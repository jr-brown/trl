import torch
import torch.utils.data
import torch.nn.functional as F
import numpy as np

from typing import List, Literal, Optional
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase, TrainingArguments

from ..trainer.utils import (
    first_true_indices,
    forward,
    truncate_response,
    truncate_response_from_sequences,
)


def retokenize(
    input_ids: torch.Tensor,
    device,
    source_processing_class: PreTrainedTokenizerBase,
    target_processing_class: PreTrainedTokenizerBase | None = None,
) -> tuple[torch.Tensor, int | None]:

    if target_processing_class is None:
        return input_ids, source_processing_class.pad_token_id

    else:
        decoded_batch = source_processing_class.batch_decode(
            input_ids, skip_special_tokens=True
        )
        new_inputs = target_processing_class(
            decoded_batch,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
        )["input_ids"]
        new_inputs = new_inputs.to(device)
        return new_inputs, target_processing_class.pad_token_id


def get_just_value(
    model: torch.nn.Module, query_responses: torch.Tensor, pad_token_id: int
) -> torch.Tensor:
    attention_mask = query_responses != pad_token_id
    lm_backbone = getattr(model, model.base_model_prefix)
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)

    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    output = lm_backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
        use_cache=False,  # otherwise mistral-based RM would error out
    )

    return model.score(output.hidden_states[-1])


def get_just_reward(
    model: torch.nn.Module, query_responses: torch.Tensor, pad_token_id: int
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
        position_ids = (
            attention_mask.cumsum(1) - attention_mask.long()
        )  # exclusive cumsum
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

    full_rewards = model.score(output.hidden_states[-1])

    # Check if we've produced reward for whole sequence, or value for each token
    if full_rewards.ndim == 1:
        return full_rewards

    # In latter case we need to get the final value of the sequence
    else:
        # Remove excess dimensions
        # If we do this earlier, bs=1 and distilbert causes shape (1,) -> (), leading to error
        full_rewards = full_rewards.squeeze(-1)

        # Will error if query_responses is entirely pad tokens as last_non_pad_tkn_idxs will be -1
        last_non_pad_tkn_idxs = (
            query_responses.size(-1)
            - first_true_indices(
                torch.flip(query_responses, dims=(-1,)) == pad_token_id
            )
            - 1
        )
        return full_rewards[
            torch.arange(full_rewards.size(0), device=full_rewards.device),
            last_non_pad_tkn_idxs,
        ]


class ScheduledParameter:
    def __init__(
        self,
        initial_value: float,
        final_value: float,
        batch_schedule_length: int,
        schedule_type: str = "linear",
    ):
        self.initial_value = initial_value
        self.final_value = final_value
        self.batch_schedule_length = batch_schedule_length
        assert schedule_type in [
            "linear",
            "cosine",
        ], f"schedule_type must be 'linear' or 'cosine', got {schedule_type}"
        self.schedule_type = schedule_type

        self.current_step = 0
        self.current_value = initial_value

    def step(self) -> float:
        self.current_step += 1
        progress = min(self.current_step / self.batch_schedule_length, 1)
        if self.schedule_type == "linear":
            self.current_value = (
                self.initial_value + (self.final_value - self.initial_value) * progress
            )
        elif self.schedule_type == "cosine":
            self.current_value = self.final_value + 0.5 * (
                self.initial_value - self.final_value
            ) * (1 + torch.cos(progress * np.pi))
        else:
            raise ValueError(
                f"schedule_type must be 'linear' or 'cosine', got {self.schedule_type}"
            )

    def get(self) -> float:
        return self.current_value


def calc_logprob(
    idx: int,
    response: torch.Tensor,
    logitss: torch.Tensor,
    local_rollout_forward_batch_size: int,
):
    # Get the logits for all tokens at each step of the generation.
    logits = logitss[idx : idx + local_rollout_forward_batch_size]

    # Get the log probabilities of every token at each step of the generation
    all_logprob = F.log_softmax(logits, dim=-1)

    # Get only those log-probabilities for the tokens that were actually generated.
    logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)

    return logprob


def calc_ref_logprob(
    ref_policy,
    query_response,
    response,
    pad_token_id: int,
    context_length: int,
    ref_temperature: float,
):
    with torch.no_grad():
        # This computes the log-probabilities for the base model (reference model)
        ref_output = forward(ref_policy, query_response, pad_token_id)
        ref_logits = ref_output.logits[:, context_length - 1 : -1]
        ref_logits /= ref_temperature + 1e-7
        ref_all_logprob = F.log_softmax(ref_logits, dim=-1)

        # Get only those log-probabilities for the tokens that were actually generated.
        ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(
            -1
        )

    return ref_logprob


def rollouts_to_loss_variables(
    queries,
    query_responses,
    maybe_answer_ids,
    logitss,
    ref_policy,
    unwrapped_value_model,
    processing_class,
    context_length,
    stop_token_id,
    response_truncation_sequences: list[list[int]],
    local_rollout_forward_batch_size,
    ref_temperature,
    scoring_function,
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
    for i in range(0, queries.shape[0], local_rollout_forward_batch_size):
        # Extract the query
        query = queries[i : i + local_rollout_forward_batch_size]
        # Extract the corresponding response
        query_response = query_responses[i : i + local_rollout_forward_batch_size]
        # Discard the first context_length tokens.
        response = query_response[:, context_length:]

        logprob = calc_logprob(i, response, logitss, local_rollout_forward_batch_size)
        torch.cuda.empty_cache()

        ref_logprob = calc_ref_logprob(
            ref_policy,
            query_response,
            response,
            pad_token_id,
            context_length,
            ref_temperature,
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

        if response_truncation_sequences is not None:
            postprocessed_response = truncate_response_from_sequences(
                response_truncation_sequences, pad_token_id, postprocessed_response
            )

        # Response Processing 2. run reward model on the truncated responses

        # FLAG - This feels inefficient for when the action-value function is a head on the model doing the generation
        # Maybe there's a way to get around this. For now, we might just want separate models.
        postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
        sequence_length = first_true_indices(postprocessed_response == pad_token_id) - 1
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

        score = scoring_function(postprocessed_query_response, maybe_answer_ids)

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


@dataclass
class OnPolicyConfig(TrainingArguments):
    r"""
    Base configuration class for on-policy trainers.

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        run_name (`Optional[str]`, *optional*, defaults to `None`):
            Name of the run.
        dataset_num_proc (`Optional[int]`, *optional*, defaults to `None`):
            Number of processes to use for processing the dataset.
        num_mini_batches (`int`, *optional*, defaults to `1`):
            Number of minibatches to split a batch into.
        total_episodes (`Optional[int]`, *optional*, defaults to `None`):
            Total number of episodes in the dataset.
        local_rollout_forward_batch_size (`int`, *optional*, defaults to `64`):
            Per rank no grad forward pass in the rollout phase.
        num_sample_generations (`int`, *optional*, defaults to `10`):
            Number of debugging samples generations (i.e., `generate_eval_completions` calls) throughout training.
        max_num_eval_batches (`Optional[int]`, defaults to `1`):
            Maximum number of batches to be used when sampling eval generations. If set to None, will iterate through the entire eval dataset.
        response_length (`int`, *optional*, defaults to `53`):
            Length of the response.
        stop_token (`Optional[str]`, *optional*, defaults to `None`):
            Stop token.
        stop_token_id (`Optional[int]`, *optional*, defaults to `None`):
            Truncation token id.
        response_truncation_sequences (`[List[List[int]]]`, *optional*, defaults to `None`):
            Stop strings for generations from the model.
        train_temperature (`float`, *optional*, defaults to `1.0`):
            Used to divide logits when training model.
        train_rollout_temperature (`float`, *optional*, defaults to `1.0`):
            Sampling temperature.
        eval_rollout_temperature (`float`, *optional*, defaults to `1.0`):
            Sampling temperature for evaluation.
        missing_eos_penalty (`Optional[float]`, *optional*, defaults to `None`):
            Penalty applied to the score when the model fails to generate an EOS token. This is useful to encourage
            to generate completions shorter than the maximum length (`max_new_tokens`). The penalty must be a positive
            value.
        sft_model_path (`str`, *optional*, defaults to `"EleutherAI/pythia-160m"`):
            Path to the SFT model.
        world_size (`Optional[int]`, *optional*, defaults to `None`):
            Number of processes (GPUs) to use for the training.
        num_total_batches (`Optional[int]`, *optional*, defaults to `None`):
            Number of total batches to train.
        micro_batch_size (`Optional[int]`, *optional*, defaults to `None`):
            Micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`).
        local_batch_size (`Optional[int]`, *optional*, defaults to `None`):
            Batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`).
        batch_size (`Optional[int]`, *optional*, defaults to `None`):
            Batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`).
        local_mini_batch_size (`Optional[int]`, *optional*, defaults to `None`):
            Mini batch size per GPU.
        mini_batch_size (`Optional[int]`, *optional*, defaults to `None`):
            Mini batch size across GPUs.
        push_to_hub (`bool`, *optional*, defaults to `False`):
            Whether to push the model to the Hub after training.
    """

    run_name: Optional[str] = None
    dataset_num_proc: Optional[int] = None
    num_mini_batches: int = 1
    total_episodes: Optional[int] = None
    local_rollout_forward_batch_size: int = 64
    num_sample_generations: int = 10
    max_num_eval_batches: Optional[int] = 1
    response_length: int = 53
    stop_token: Optional[Literal["eos"]] = None
    stop_token_id: Optional[int] = None
    response_truncation_sequences: Optional[List[List[int]]] = None
    train_temperature: float = 1.0
    train_rollout_temperature: float = 1.0
    eval_rollout_temperature: float = 1.0
    missing_eos_penalty: Optional[float] = None
    sft_model_path: str = "EleutherAI/pythia-160m"
    reward_model_path: str | None = None
    world_size: Optional[int] = None
    num_total_batches: Optional[int] = None
    micro_batch_size: Optional[int] = None
    local_batch_size: Optional[int] = None
    batch_size: Optional[int] = None
    local_mini_batch_size: Optional[int] = None
    mini_batch_size: Optional[int] = None
    push_to_hub: bool = False
    time_limit_mins: Optional[float] = None
    # Parameters
    lam: float = 0.95
    final_lam: Optional[float] = None
    lam_episode_length: Optional[int] = None
    lam_schedule_type: str = "linear"
    final_answer_split_str: str | None = None

