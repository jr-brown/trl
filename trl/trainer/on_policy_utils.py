import torch
import torch.utils.data
import torch.nn.functional as F

from ..trainer.utils import (
    first_true_indices,
    forward,
    retokenize,
    truncate_response,
    truncate_response_from_sequences,
)


def get_just_value(
    model: torch.nn.Module, query_responses: torch.Tensor, pad_token_id: int
) -> torch.Tensor:
    attention_mask = query_responses != pad_token_id
    lm_backbone = getattr(model, model.base_model_prefix)
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)

    # Wrapping to allow for models which don't use position_ids, e.g. DistilBert
    # These models also don't have the use_cache kwarg
    if not hasattr(model, "use_position_ids") or model.use_position_ids:
        position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
        output = lm_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,  # otherwise mistral-based RM would error out
        )
    else:
        output = lm_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )

    return model.score(output.hidden_states[-1])


def get_just_reward(
    model: torch.nn.Module, query_responses: torch.Tensor, pad_token_id: int, query_length: int
) -> torch.Tensor:
    """
    3LM wrapper function for getting the reward from a model.
    Passes query response pairs into the reward model and extracts the rewards for the responses.

    Args:
        model: The reward model.
        query_responses: The responses to get the reward for.

    Returns:

    """
    attention_mask = query_responses != pad_token_id
    lm_backbone = getattr(model, model.base_model_prefix)
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)

    # Wrapping to allow for models which don't use position_ids, e.g. DistilBert
    # These models also don't have the use_cache kwarg
    if not hasattr(model, "use_position_ids") or model.use_position_ids:
        position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
        output = lm_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,  # otherwise mistral-based RM would error out
        )
    else:
        output = lm_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )

    full_rewards = model.score(output.hidden_states[-1]).squeeze(-1)

    # Check if we've produced reward for whole sequence, or value for each token
    if len(full_rewards.shape) == 1:
        return full_rewards

    # In latter case we need to get the final value of the sequence
    else:
        last_non_pad_tkn_idxs = first_true_indices(
            query_responses[:, query_length:] == pad_token_id
        ) - 1 + query_length
        return full_rewards[
            torch.arange(full_rewards.size(0), device=full_rewards.device),
            last_non_pad_tkn_idxs,
        ]


def calc_logprob(idx: int, response: torch.Tensor, logitss: torch.Tensor, local_rollout_forward_batch_size: int):
    # Get the logits for all tokens at each step of the generation.
    logits = logitss[idx : idx + local_rollout_forward_batch_size]

    # Get the log probabilities of every token at each step of the generation
    all_logprob = F.log_softmax(logits, dim=-1)

    # Get only those log-probabilities for the tokens that were actually generated.
    logprob = torch.gather(
        all_logprob, 2, response.unsqueeze(-1)
    ).squeeze(-1)

    return logprob


def calc_ref_logprob(ref_policy, query_response, response, pad_token_id: int, context_length: int, ref_temperature: float):
    # This computes the log-probabilities for the base model (reference model)
    ref_output = forward(
        ref_policy, query_response, pad_token_id
    )
    ref_logits = ref_output.logits[:, context_length - 1 : -1]
    ref_logits /= ref_temperature + 1e-7
    ref_all_logprob = F.log_softmax(ref_logits, dim=-1)

    # Get only those log-probabilities for the tokens that were actually generated.
    ref_logprob = torch.gather(
        ref_all_logprob, 2, response.unsqueeze(-1)
    ).squeeze(-1)

    return ref_logprob


def rollouts_to_loss_variables(
    queries,
    query_responses,
    logitss,
    ref_policy,
    unwrapped_value_model,
    reward_model,
    processing_class,
    reward_model_processing_class,
    context_length,
    stop_token_id,
    response_truncation_sequences: list[list[int]],
    local_rollout_forward_batch_size,
    ref_temperature,
    device,
):
    """
    Takes in an iteration batch of queries and responses, and computes various variables needed for defining the loss function. 

    Args:
        queries (torch.Tensor): The queries to the model.
        query_responses (torch.Tensor): The queries concatenated with the responses.
        logitss (torch.Tensor): The full collection of logits at each step of the response generation.
        ref_policy (AutoModelForCausalLM): The reference model.
        unwrapped_value_model (AutoModelForSequenceClassification): The value model.
        reward_model (AutoModelForSequenceClassification): The reward model.
        processing_class (AutoTokenizer): The tokenizer class for the model.
        reward_model_processing_class (AutoTokenizer): The tokenizer class for the reward model.
        context_length (int): The length of the context.
        stop_token_id (Optional[int]): The stop_token_id.
        response_truncation_sequences (list[list[int]]): Token id sequences at which to truncate the responses (inclusive).
        local_rollout_forward_batch_size (int): The number of rollouts to process simultaneously 
        ref_temperature (float): The ref_temperature to use for the reference model.
        device (torch.device): The device to run the model on.

    Returns:
        responses (torch.Tensor): The responses generated by the model
        postprocessed_responses (torch.Tensor): The responses generated by the model, truncated after the first stop token
        logprobs (torch.Tensor): The log-probabilities of the generated responses
        ref_logprobs (torch.Tensor): The log-probabilities of the generated responses according to the reference model
        sequence_lengths (torch.Tensor): The length of the generated responses
        scores (torch.Tensor): The reward of the generated responses
        state_values (torch.Tensor): The value of the generated responses
    """
    responses = []
    postprocessed_responses = []
    logprobs = []
    ref_logprobs = []
    scores = []
    sequence_lengths = []
    state_values = []

    pad_token_id = processing_class.pad_token_id

    # Note: local_rollout_forward_batch_size gives how many token generation steps we chunk trajectories into.
    # Iterate through chunks of the queries, moving along by the local_rollout_forward_batch_size each time.
    for i in range(
        0, queries.shape[0], local_rollout_forward_batch_size
    ):
        # Extract the query
        query = queries[i : i + local_rollout_forward_batch_size]
        # Extract the corresponding response
        query_response = query_responses[
            i : i + local_rollout_forward_batch_size
        ]
        # Discard the first context_length tokens.
        response = query_response[:, context_length:]

        logprob = calc_logprob(i, response, logitss, local_rollout_forward_batch_size)
        torch.cuda.empty_cache()

        ref_logprob = calc_ref_logprob(
            ref_policy, query_response, response, pad_token_id, context_length, ref_temperature
        )
        torch.cuda.empty_cache()

        # Response Processing 1. truncate response after the first occurrence of `stop_token_id` and before the occurance of any token sequence in `response_truncation_sequences`
        postprocessed_response = response
        if (
            stop_token_id is not None
        ):  # handle the edge case when stop_token_id exists but is 0
            postprocessed_response = truncate_response(
                stop_token_id, pad_token_id, response
            )

        if (
            response_truncation_sequences is not None
        ):  
            postprocessed_response = truncate_response_from_sequences(
                response_truncation_sequences, pad_token_id, response
            )
            
        # Response Processing 2. run reward model on the truncated responses

        # FLAG - This feels inefficient for when the action-value function is a head on the model doing the generation
        # Maybe there's a way to get around this. For now, we might just want separate models.
        postprocessed_query_response = torch.cat(
            (query, postprocessed_response), 1
        )
        sequence_length = (
            first_true_indices(
                postprocessed_response == pad_token_id
            )
            - 1
        )
        # It looks like TRL treates rewards and values the same way.
        # We might need our own class for action-value functions.
        # full_value has shape [batch, query_length, 1]

        full_value = get_just_value(
            unwrapped_value_model,
            query_response,
            pad_token_id,
        )
        # Extract only the value estimates for the completion
        state_value = full_value[:, context_length - 1 : -1].squeeze(-1)
        # The score is the reward at the end of each query sequence.
        # score has shape [batch]
        reward_model_inputs, reward_model_pad_token = retokenize(
            postprocessed_query_response,
            device,
            processing_class,
            reward_model_processing_class,
        )
        score = get_just_reward(
            reward_model,
            reward_model_inputs,
            reward_model_pad_token,
            context_length,
        )

        # This is just a bunch of logging stuff
        responses.append(response)
        postprocessed_responses.append(postprocessed_response)
        logprobs.append(logprob)
        ref_logprobs.append(ref_logprob)
        sequence_lengths.append(sequence_length)
        scores.append(score)
        state_values.append(state_value)

    # Now we stack all these sub-chunks together so we can pass them through as one batch.
    responses = torch.cat(responses, 0)
    postprocessed_responses = torch.cat(postprocessed_responses, 0)
    logprobs = torch.cat(logprobs, 0)
    ref_logprobs = torch.cat(ref_logprobs, 0)
    sequence_lengths = torch.cat(sequence_lengths, 0)
    scores = torch.cat(scores, 0)
    state_values = torch.cat(state_values, 0)

    return (
        responses,
        postprocessed_responses,
        logprobs,
        ref_logprobs,
        sequence_lengths,
        scores,
        state_values,
    )

