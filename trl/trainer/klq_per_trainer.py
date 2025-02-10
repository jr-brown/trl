# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
from typing import Dict, List, Optional, Tuple, Union, Callable
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor  # for type hinting
from dataclasses import dataclass

from datasets import Dataset
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    TrainerCallback,
)

from ..core import masked_mean, masked_whiten
from ..models.utils import unwrap_model_for_generation
from ..trainer.utils import (
    batch_generation,
    forward,
)
from .klq_config import KLQConfig
from .on_policy_utils import (
    OnPolicyConfig,
    forward_pass_on_rollouts,
    rollouts_to_loss_variables,
    ScheduledParameter,
    INVALID_LOGPROB,
)
from .on_policy_trainer import OnPolicyTrainer

# new imports for PER file
from .klq_trainer import loss_function_map, LossFunctionTensors

ProcessingClass = PreTrainedTokenizerBase


@dataclass
class KLQPERConfig(KLQConfig):
    """Config for KLQ with Prioritised Experience Replay.

    Note:
        The replay number is equal to the replay rate times the local batch size.
        The learning_starts parameters should *always* be greater than the replay number.
    """

    local_replay_buffer_capacity: int = 250
    replay_rate: float = 0.5
    local_learning_starts: int = 100
    retrace: bool = True
    retrace_clip: float = 1.0
    expert_retrace_clip: float = 0.95


# New code at top of file
def prioritise_batch(
    config: KLQPERConfig,
    responses,
    state_values,
    lam,
    ref_logprobs,
    new_logprobs,
    gen_logprobs,
    rewards,
    padding_mask_plus_one,
):
    """Take in the batch that we have trained on, and the current policy.
    Run a forward pass to recompute the Deltas.
    Deltas are computed using the backwards recursion relationship for retrace(\lambda) if config.retrace is True.
    Else, we use the Q^*(\lambda) estimator.
    From Deltas recompute priorities.
    Currently priorities are computed using the mean square of the Deltas.
    Specifically, need to take in: x (query), y (completion), reference probabilities (to obtain Q from policy),
    behavioural probabilities, and reward model scores.

    Only need logprobs and state_values
    """
    # Initial cheap computation to get action values
    action_values = config.kl_coef * (new_logprobs - ref_logprobs) + state_values
    action_values = torch.masked_fill(action_values, padding_mask_plus_one, 0)

    if config.retrace:
        traces = torch.clamp(
            torch.exp(new_logprobs - gen_logprobs), max=config.retrace_clip
        )
    else:
        traces = torch.ones_like(new_logprobs)

    last_Delta = 0
    Deltas_reversed = []
    gen_length = responses.shape[1]  # This is the length of the responses.
    for t in reversed(range(gen_length)):
        # Extract the next token state-values
        next_state_values = state_values[:, t + 1] if t < gen_length - 1 else 0.0
        trace = (
            traces[:, t + 1] if t < gen_length - 1 else 1.0
        )  # BUGHOTSPOT - Edward: I'm pretty sure this is correct but it could be t instead of t + 1
        # Compute the TD-error
        delta = rewards[:, t] + config.gamma * next_state_values - action_values[:, t]
        # Use the retrace backwards recursion relationship
        last_Delta = delta + config.gamma * lam * trace * last_Delta
        Deltas_reversed.append(last_Delta)
    # Create the Delta estimates by reversing the lambda-return backward recursion
    Deltas = torch.stack(Deltas_reversed[::-1], dim=1)
    # Set the return estimates to be the Delta estimates
    returns = config.alpha * Deltas + action_values  # This used to be state_values
    returns = torch.masked_fill(returns, padding_mask_plus_one, 0)  # BUGHOTSPOT

    # Priority computation
    from trl.core import masked_mean

    priorities = masked_mean(Deltas**2, ~padding_mask_plus_one, axis=1)
    # Take mean over the length dimension, not batch dimension
    return priorities, returns


class PERBuffer:
    """Prioritised Experience Replay.
    gen_logprobs are logprobs for model that generated the responses."""

    def __init__(
        self,
        capacity: int,
        query_length: int,
        response_length: int,
        device: torch.device,
    ):
        self.capacity = capacity
        self.device = device
        self.row_available = torch.ones(capacity, dtype=torch.bool, device=device)
        qr_length = query_length + response_length
        self.qr_length = qr_length
        self.query_responses = torch.zeros(
            (capacity, qr_length), dtype=torch.long, device=device
        )
        self.ref_logprobs = torch.zeros((capacity, response_length), device=device)
        self.gen_logprobs = torch.zeros((capacity, response_length), device=device)
        self.rewards = torch.zeros((capacity, response_length), device=device)
        self.returns = torch.zeros((capacity, response_length), device=device)
        self.priorities = torch.zeros(capacity, device=device)
        self.padding_mask = torch.zeros(
            (capacity, response_length), dtype=torch.bool, device=device
        )
        self.padding_mask_plus_one = torch.zeros(
            (capacity, response_length), dtype=torch.bool, device=device
        )

    @property
    def num_samples(self):
        """Number of rows corresponding to previous experiences."""
        return torch.sum(~self.row_available)

    def add_batch(
        self,
        query_responses,
        ref_logprobs,
        gen_logprobs,
        rewards,
        returns,
        priorities,
        padding_mask,
        padding_mask_plus_one,
    ):
        """Add a batch of samples to the buffer."""
        # NOTE: could read device from query_responses etc also
        # if there are insufficient empty rows, mark the lowest priority rows for overwrite
        num_samples_in = query_responses.shape[0]
        if torch.sum(self.row_available) < num_samples_in:
            idxs_to_overwrite = torch.argsort(self.priorities)[:num_samples_in]
            self.row_available[idxs_to_overwrite] = True
        # get first available rows now have now guaranteed that there are enough
        idxs_available = torch.arange(self.capacity, device=self.device)[
            self.row_available
        ]
        idxs_to_write = idxs_available[:num_samples_in]

        self.query_responses[idxs_to_write] = query_responses
        self.ref_logprobs[idxs_to_write] = ref_logprobs
        self.gen_logprobs[idxs_to_write] = gen_logprobs
        self.rewards[idxs_to_write] = rewards
        self.returns[idxs_to_write] = returns
        self.priorities[idxs_to_write] = priorities
        self.padding_mask[idxs_to_write] = padding_mask
        self.padding_mask_plus_one[idxs_to_write] = padding_mask_plus_one
        self.row_available[idxs_to_write] = False

    def pop_batch(self, pop_batch_size: int):
        """Extracts the samples with highest priority from the buffer."""
        idxs_to_pop = torch.argsort(self.priorities, descending=True)[:pop_batch_size]
        # TODO: Implement sampling with probabilities proportional to priority to the alpha.
        query_responses = self.query_responses[idxs_to_pop]
        ref_logprobs = self.ref_logprobs[idxs_to_pop]
        gen_logprobs = self.gen_logprobs[idxs_to_pop]
        rewards = self.rewards[idxs_to_pop]
        returns = self.returns[idxs_to_pop]
        priorities = self.priorities[idxs_to_pop]
        padding_mask = self.padding_mask[idxs_to_pop]
        padding_mask_plus_one = self.padding_mask_plus_one[idxs_to_pop]
        self.row_available[idxs_to_pop] = True
        return (
            query_responses,
            ref_logprobs,
            gen_logprobs,
            rewards,
            returns,
            priorities,
            padding_mask,
            padding_mask_plus_one,
        )


class KLQPERStats:
    def __init__(self, stats_shape: Tuple[int, int, int], device: torch.device) -> None:
        self.loss_function_stats = torch.zeros(stats_shape, device=device)
        self.action_value_stats = torch.zeros(stats_shape, device=device)
        self.state_value_stats = torch.zeros(stats_shape, device=device)
        self.advantage_stats = torch.zeros(stats_shape, device=device)
        # self.log_ratio_new_ref_stats = torch.zeros(stats_shape, device=device)
        self.entropy_stats = torch.zeros(stats_shape, device=device)
        self.prev_ref_log_ratio_stats = torch.zeros(stats_shape, device=device)
        # self.kl_new_ref_stats = torch.zeros(stats_shape, device=device)
        self.prev_new_log_ratio_stats = torch.zeros(stats_shape, device=device)
        # self.kl_new_prev_stats = torch.zeros(stats_shape, device=device)

    def update(
        self,
        update_location: Tuple[int, int, int],
        action_value_function_loss,
        action_value_prediction,
        state_value_prediction,
        # new_ref_log_ratio,
        entropy,
        prev_ref_log_ratio,
        # kl_new_ref,
        prev_new_log_ratio,
        # kl_new_prev,
    ):
        self.loss_function_stats[update_location] = action_value_function_loss
        self.action_value_stats[update_location] = action_value_prediction
        self.state_value_stats[update_location] = state_value_prediction
        self.advantage_stats[update_location] = (
            action_value_prediction - state_value_prediction
        )
        # self.log_ratio_new_ref_stats[update_location] = new_ref_log_ratio
        self.entropy_stats[update_location] = entropy
        self.prev_ref_log_ratio_stats[update_location] = prev_ref_log_ratio
        # self.kl_new_ref_stats[update_location] = kl_new_ref
        self.prev_new_log_ratio_stats[update_location] = prev_new_log_ratio
        # self.kl_new_prev_stats[update_location] = kl_new_prev


def klq_per_micro_batch_updates(
    config: KLQConfig,
    # integers
    epoch_idx: int,
    minibatch_idx: int,
    mini_batch_start: int,
    context_length: int,
    pad_token_id: int,
    # tensors
    batch_inds,
    responses,
    query_responses,
    logprobs,
    ref_logprobs,
    returns,
    state_values,
    padding_mask,
    padding_mask_plus_one,
    # Stateful parameters that get updated
    model,
    accelerator,
    optimizer,
    stats,
):
    """Currently simply the same as KLQ micro batch updates"""
    mini_batch_end = mini_batch_start + config.local_mini_batch_size
    mini_batch_inds = batch_inds[mini_batch_start:mini_batch_end]
    gradient_accumulation_idx = 0
    for micro_batch_start in range(
        0, config.local_mini_batch_size, config.per_device_train_batch_size
    ):
        # I think that micro-batches are minibatches divided between machines.
        # accelerator is initialised precisely with config.gradient_accumulation_steps
        # so knows how many gradients to accumulate before updating weights
        with accelerator.accumulate(model):
            micro_batch_end = micro_batch_start + config.per_device_train_batch_size
            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
            # Retrieve the relevant variables for this microbatch
            micro_batch_responses = responses[micro_batch_inds]
            micro_batch_query_responses = query_responses[micro_batch_inds]

            # WARNING: This is not actually the log probabilities for the previous policy.
            # Instead, at present it is the generative log probabilities.
            # This is equal to the prev log probs for the samples generated from rollouts.
            # However, it is not equal to the prev log probs for the samples from the buffer.
            micro_batch_prev_logprobs = logprobs[micro_batch_inds]

            micro_batch_ref_logprobs = ref_logprobs[micro_batch_inds]
            micro_batch_return = returns[micro_batch_inds]
            micro_batch_prev_state_values = state_values[micro_batch_inds]

            output, state_value_prediction_temporary = forward(
                model, micro_batch_query_responses, pad_token_id
            )
            logits = output.logits[:, context_length - 1 : -1]
            logits /= config.train_temperature + 1e-7
            new_all_logprobs = F.log_softmax(logits, dim=-1)
            new_logprobs = torch.gather(
                new_all_logprobs, 2, micro_batch_responses.unsqueeze(-1)
            ).squeeze(-1)
            new_logprobs = torch.masked_fill(
                new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
            )

            # Compute inputs to loss function
            state_value_prediction = state_value_prediction_temporary[
                :, context_length - 1 : -1
            ].squeeze(-1)
            state_value_prediction = torch.masked_fill(
                state_value_prediction, padding_mask_plus_one[micro_batch_inds], 0
            )
            # This is for logging
            mean_state_value_prediction = masked_mean(
                state_value_prediction, ~padding_mask_plus_one[micro_batch_inds]
            )

            new_ref_log_ratio = new_logprobs - micro_batch_ref_logprobs
            prev_ref_log_ratio = micro_batch_prev_logprobs - micro_batch_ref_logprobs
            action_value_prediction = (
                config.kl_coef * (new_ref_log_ratio) + state_value_prediction
            )
            action_value_prediction = torch.masked_fill(
                action_value_prediction, padding_mask_plus_one[micro_batch_inds], 0
            )  # TODO: I'm quite unsure about whether this should be padding mask or padding mask plus one.

            # This is for logging
            mean_action_value_prediction = masked_mean(
                action_value_prediction, ~padding_mask_plus_one[micro_batch_inds]
            )  # TODO: Again, unsure about pm or pm+1

            # Compute loss
            action_value_function_losses = loss_function_map[config.loss_function](
                config,
                LossFunctionTensors(
                    action_value_prediction,
                    state_value_prediction,
                    new_ref_log_ratio,
                    micro_batch_prev_state_values,
                    prev_ref_log_ratio,
                    micro_batch_return,
                ),
            )
            action_value_function_loss = masked_mean(
                action_value_function_losses, ~padding_mask_plus_one[micro_batch_inds]
            )
            # Perform the update step.
            accelerator.backward(action_value_function_loss)
            optimizer.step()
            optimizer.zero_grad()

            # This is all just for logging.
            with torch.no_grad():

                # Logits has shape [batch, response_length, vocab_size]
                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                # prob_dist has shape [batch, response_length, vocab_size]
                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(
                    prob_dist * logits, dim=-1
                )
                # entropy has shape [batch, response_length]
                avg_entropy = masked_mean(entropy, ~padding_mask[micro_batch_inds])
                # avg_entropy is a scalar.
                # avg_entropy is the average entropy of the probability distribution at each non-padding token in the completion.

                prev_new_log_ratio = micro_batch_prev_logprobs - new_logprobs
                prev_new_log_ratio = torch.masked_fill(
                    prev_new_log_ratio, padding_mask[micro_batch_inds], 0
                )
                prev_ref_log_ratio = (
                    micro_batch_prev_logprobs - micro_batch_ref_logprobs
                )
                prev_ref_log_ratio = torch.masked_fill(
                    prev_ref_log_ratio, padding_mask[micro_batch_inds], 0
                )  # NOTE: This computation is not meaningful at the moment, owing to the conflation of prev log probs with gen log probs.
                # new_prev_log_ratio = -prev_new_log_ratio
                # new_prev_ratio = torch.exp(new_prev_log_ratio)

                update_location = (
                    epoch_idx,
                    minibatch_idx,
                    gradient_accumulation_idx,
                )
                stats.update(
                    update_location,
                    action_value_function_loss=action_value_function_loss,
                    action_value_prediction=mean_action_value_prediction,
                    state_value_prediction=mean_state_value_prediction,
                    # new_ref_log_ratio=new_ref_log_ratio.sum(1).mean(),
                    entropy=avg_entropy,
                    prev_ref_log_ratio=prev_ref_log_ratio.sum(
                        1
                    ).mean(),  # WARNING: not meaningful at the moment
                    # kl_new_ref=(new_prev_ratio * new_ref_log_ratio).sum(1).mean(),
                    prev_new_log_ratio=prev_new_log_ratio.sum(
                        1
                    ).mean(),  # WARNING: not meaningful at the moment
                    # kl_new_prev=(new_prev_ratio * new_prev_log_ratio).sum(1).mean(),
                )
        gradient_accumulation_idx += 1
    minibatch_idx += 1


def klq_per_batch_update(
    config: KLQPERConfig,
    generation_config,
    processing_class,
    reward_model_processing_class,
    # optimisation / performance
    optimizer,
    accelerator,
    device: torch.device,
    # stateful parameters to be updated
    model,
    ref_policy,
    reward_model,
    klq_stats: KLQPERStats,
    # data for the this batch!
    buffer: PERBuffer,
    data: Dict[
        str, Tensor
    ],  # Need to change this to the dataloader so that we can draw arbitrary samples from the loader.
    # Scheduled parameters
    lam: float,
):

    start_time = time.perf_counter()

    ### PER-NEW ->
    """
    When the number of samples in the buffer exceeds the learning starts, we draw
    replay_rate * local_batch_size samples from the buffer. 
    These are then combined with the samples generated from rollouts to form the training batch.
    """
    per_training = False

    if buffer.num_samples >= config.local_learning_starts:
        per_training = True
        replay_number = int(config.replay_rate * config.local_batch_size)
        assert replay_number <= buffer.num_samples, "Not enough samples in buffer."
        (
            replayed_query_responses,
            replayed_ref_logprobs,
            replayed_gen_logprobs,
            replayed_rewards,
            replayed_returns,
            replayed_priorities,
            replayed_padding_mask,
            replayed_padding_mask_plus_one,
        ) = buffer.pop_batch(replay_number)

    ### PER-NEW <-

    with torch.no_grad():
        queries = data["input_ids"].to(device)
        if per_training:
            rollout_number = config.local_batch_size - replay_number
            queries = queries[:rollout_number]

        context_length = queries.shape[1]
        with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
            # query_respones and logitss are both torch Tensors.
            # query_responses has shape [batch, query_length]
            # logitss has shape [batch, response_length, vocabulary size]

            query_responses, logitss = batch_generation(
                unwrapped_model.policy,
                queries,
                config.local_rollout_forward_batch_size,
                processing_class.pad_token_id,
                generation_config,
            )
        generation_stop_time = time.perf_counter()
        generation_time = generation_stop_time - start_time
        (
            responses,
            postprocessed_responses,
            gen_logprobs,
            ref_logprobs,
            sequence_lengths,
            scores,
            state_values,
        ) = rollouts_to_loss_variables(
            queries=queries,
            query_responses=query_responses,
            logitss=logitss,
            ref_policy=ref_policy,
            unwrapped_value_model=accelerator.unwrap_model(model).value_model,
            reward_model=reward_model,
            processing_class=processing_class,
            reward_model_processing_class=reward_model_processing_class,
            context_length=context_length,
            stop_token_id=config.stop_token_id,
            response_truncation_sequences=config.response_truncation_sequences,
            local_rollout_forward_batch_size=config.local_rollout_forward_batch_size,
            ref_temperature=config.train_temperature,
            device=device,
        )
        action_values = config.kl_coef * (gen_logprobs - ref_logprobs) + state_values
        torch.cuda.empty_cache()
        gc.collect()

        # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
        # Completions not passing that filter will receive a lower score.
        contain_eos_token = torch.any(
            postprocessed_responses == processing_class.eos_token_id, dim=-1
        )
        if config.missing_eos_penalty is not None:
            scores[~contain_eos_token] -= config.missing_eos_penalty
        accelerator.print(
            f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}"
        )

        # be very careful with `padding_mask_plus_one`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
        response_idxs = torch.arange(
            responses.shape[1], device=responses.device
        ).repeat(responses.shape[0], 1)
        padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
        gen_logprobs = torch.masked_fill(gen_logprobs, padding_mask, INVALID_LOGPROB)
        ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
        sequence_lengths_plus_one = sequence_lengths + 1
        padding_mask_plus_one = response_idxs > (sequence_lengths_plus_one.unsqueeze(1))
        state_values = torch.masked_fill(state_values, padding_mask_plus_one, 0)
        action_values = torch.masked_fill(action_values, padding_mask_plus_one, 0)

        # 4. compute rewards
        prev_ref_log_ratio = gen_logprobs - ref_logprobs  # This is just for logging!
        prev_ref_log_ratio = torch.masked_fill(
            prev_ref_log_ratio, padding_mask, 0
        )  # Set the log-ratio to 0 for padding tokens.
        non_score_reward = (
            -config.kl_coef * prev_ref_log_ratio
        )  # This is just for logging!

        # rewards has shape [batch, response_length]
        rewards = torch.zeros_like(gen_logprobs)
        batch_indices = torch.arange(
            rewards.size(0), device=rewards.device
        )  # [0, 1, 2, ..., batch_size - 1]
        sequence_end_indices = torch.where(
            sequence_lengths_plus_one < rewards.size(1),
            sequence_lengths_plus_one,
            sequence_lengths,
        )
        rewards[[batch_indices, sequence_end_indices]] += scores

        # 5. whiten rewards
        if config.whiten_rewards:
            rewards = masked_whiten(
                rewards, mask=~padding_mask_plus_one, shift_mean=False
            )
            rewards = torch.masked_fill(rewards, padding_mask_plus_one, 0)

        # 6. compute Deltas and returns
        # Initialise the GAE at 0 for the last time step.
        last_Delta = 0
        Deltas_reversed = []
        gen_length = responses.shape[1]  # This is the length of the responses.
        for t in reversed(range(gen_length)):
            # Extract the next token state-values
            next_state_values = state_values[:, t + 1] if t < gen_length - 1 else 0.0
            # Compute the TD-error
            delta = (
                rewards[:, t] + config.gamma * next_state_values - action_values[:, t]
            )
            # Use the GAE backwards recursion relationship
            last_Delta = delta + config.gamma * lam * last_Delta
            Deltas_reversed.append(last_Delta)
        # Create the Delta estimates by reversing the lambda-return backward recursion
        Deltas = torch.stack(Deltas_reversed[::-1], dim=1)
        # Set the return estimates to be the Delta estimates
        returns = config.alpha * Deltas + action_values  # This used to be state_values
        returns = torch.masked_fill(returns, padding_mask_plus_one, 0)  # BUGHOTSPOT

        # We only want the returns and the rewards, so delete all other variables.
        del (
            last_Delta,
            Deltas_reversed,
            delta,
            next_state_values,
            Deltas,
        )
        torch.cuda.empty_cache()
    processing_stop_time = time.perf_counter()
    processing_time = processing_stop_time - generation_stop_time

    ### PER-NEW ->
    # Combine the samples from the rollout with samples from the buffer

    if per_training:
        query_responses = torch.cat([query_responses, replayed_query_responses], dim=0)
        ref_logprobs = torch.cat([ref_logprobs, replayed_ref_logprobs], dim=0)
        gen_logprobs = torch.cat([gen_logprobs, replayed_gen_logprobs], dim=0)
        rewards = torch.cat([rewards, replayed_rewards], dim=0)
        returns = torch.cat([returns, replayed_returns], dim=0)
        padding_mask = torch.cat([padding_mask, replayed_padding_mask], dim=0)
        padding_mask_plus_one = torch.cat(
            [padding_mask_plus_one, replayed_padding_mask_plus_one], dim=0
        )

    ### PER-NEW <-

    # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
    # num_epochs_per_batch_update specifies how many times to loop over the PPO dataset.
    for epoch_idx in range(config.num_epochs_per_batch_update):
        # Draw a random permutation
        batch_inds = np.random.permutation(config.local_batch_size)
        minibatch_idx = 0
        for mini_batch_start in range(
            0, config.local_batch_size, config.local_mini_batch_size
        ):
            klq_per_micro_batch_updates(
                config=config,
                # integers
                epoch_idx=epoch_idx,
                minibatch_idx=minibatch_idx,
                mini_batch_start=mini_batch_start,
                context_length=context_length,
                pad_token_id=processing_class.pad_token_id,
                # tensors
                batch_inds=batch_inds,
                responses=responses,
                query_responses=query_responses,
                logprobs=gen_logprobs,  # WARNING:
                ref_logprobs=ref_logprobs,
                returns=returns,
                state_values=state_values,
                padding_mask=padding_mask,
                padding_mask_plus_one=padding_mask_plus_one,
                # Stateful parameters that get updated
                model=model,
                accelerator=accelerator,
                optimizer=optimizer,
                stats=klq_stats,
            )
            torch.cuda.empty_cache()

    # We recompute the Deltas and returns here

    with torch.no_grad():
        # Do a whole forward pass with new weights to get the new logprobs and state values.
        new_logprobs, state_values = forward_pass_on_rollouts(
            config=config,
            model=model,
            query_responses=query_responses,
            local_rollout_forward_batch_size=config.local_rollout_forward_batch_size,
            context_length=context_length,
            pad_token_id=processing_class.pad_token_id,
            padding_mask=padding_mask,
            padding_mask_plus_one=padding_mask_plus_one,
            ref_logprobs=ref_logprobs,
        )
        priorities, returns = prioritise_batch(
            config=config,
            responses=responses,
            state_values=state_values,
            lam=lam,
            ref_logprobs=ref_logprobs,
            new_logprobs=new_logprobs,
            gen_logprobs=gen_logprobs,
            rewards=rewards,
            padding_mask_plus_one=padding_mask_plus_one,
        )
        buffer.add_batch(
            query_responses=query_responses,
            ref_logprobs=ref_logprobs,
            gen_logprobs=gen_logprobs,
            rewards=rewards,
            returns=returns,
            priorities=priorities,
            padding_mask=padding_mask,
            padding_mask_plus_one=padding_mask_plus_one,
        )

    training_stop_time = time.perf_counter()
    training_time = training_stop_time - processing_stop_time

    # At the end of training, log a bunch of statistics in the metrics dictionary.
    with torch.no_grad():
        # The sum the non-score reward over the response length, and then take the mean over the batch.
        mean_non_score_reward = non_score_reward.sum(1).mean()
        # Compute the RLHF reward by adding the mean non-score reward to the mean score.
        mean_rlhf_reward = mean_non_score_reward + scores.mean()

        metrics = {}
        s = klq_stats
        metrics_gathered_and_meaned = {
            "objective/traj/non_score_reward": mean_non_score_reward,
            "objective/traj/rlhf_reward": mean_rlhf_reward,
            "objective/traj/scores": scores.mean(),
            #
            "policy/token/entropy": s.entropy_stats,
            "policy/traj/prev_ref_log_ratio": s.prev_ref_log_ratio_stats,
            "policy/traj/prev_new_log_ratio": s.prev_new_log_ratio_stats,
            #
            "loss/token/action_value_loss": s.loss_function_stats,
            #
            "value_function/token/state_value": s.state_value_stats,
            "value_function/token/action_value": s.action_value_stats,
            "value_function/token/advantage": s.advantage_stats,
        }
        for m_name, m_tensor in metrics_gathered_and_meaned.items():
            metrics[m_name] = accelerator.gather(m_tensor).mean().item()

        # Metrics that do not need gathering
        metrics["objective/traj/num_eos_tokens"] = (
            (responses == processing_class.eos_token_id).sum().item()
        )
        metrics["loss/token/target_value"] = returns.mean().item()
        metrics["loss/token/target_value_var"] = returns.var().item()

    metrics["time/iteration/generation"] = generation_time
    metrics["time/iteration/processing"] = processing_time
    metrics["time/iteration/training"] = training_time

    return metrics


class KLQPERTrainer(OnPolicyTrainer):
    _tag_names = ["trl", "klq"]

    def __init__(
        self,
        # config and tokenizers
        config: KLQPERConfig,
        processing_class: ProcessingClass,
        reward_model_processing_class: ProcessingClass,
        # models
        policy: nn.Module,
        ref_policy: nn.Module,
        reward_model: nn.Module,
        value_model: nn.Module,
        # datasets and loaders
        train_dataset: Dataset,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        callbacks: Optional[List[TrainerCallback]] = None,
    ) -> None:

        super().__init__(
            config=config,
            processing_class=processing_class,
            reward_model_processing_class=reward_model_processing_class,
            policy=policy,
            ref_policy=ref_policy,
            reward_model=reward_model,
            value_model=value_model,
            train_dataset=train_dataset,
            data_collator=data_collator,
            eval_dataset=eval_dataset,
            optimizers=optimizers,
            callbacks=callbacks,
        )
        self.buffer = None

    def _initialise_stats(self) -> KLQPERStats:
        stats_shape = (
            self.args.num_epochs_per_batch_update,
            self.args.local_batch_size,
            self.args.local_mini_batch_size,
        )
        return KLQPERStats(stats_shape, self.accelerator.device)

    def _batch_update(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        if self.buffer is None:
            queries = data["input_ids"]
            query_length = queries.shape[1]

            self.buffer = PERBuffer(
                capacity=self.args.local_replay_buffer_capacity,
                query_length=query_length,
                response_length=self.args.response_length,
                device=self.accelerator.device,
            )

        return klq_per_batch_update(
            # config-like and tokenizers
            config=self.args,
            generation_config=self.train_generation_config,
            processing_class=self.processing_class,
            reward_model_processing_class=self.reward_model_processing_class,
            # optimisation / performance
            optimizer=self.optimizer,
            accelerator=self.accelerator,
            device=self.accelerator.device,
            # stateful models and stats to be updated
            model=self.model,
            ref_policy=self.ref_policy,
            reward_model=self.reward_model,
            klq_stats=self.stats,
            # data for this batch!
            buffer=self.buffer,
            data=data,
            # Scheduled parameters
            lam=self.lambda_scheduler.get(),
        )
