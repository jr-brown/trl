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
from accelerate import Accelerator
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
from .on_policy_trainer import OnPolicyTrainer
from .ppo_config import PPOConfig
from .on_policy_utils import rollouts_to_loss_variables

# To actually make type checking helpful and not throw errors everywhere
ProcessingClass = PreTrainedTokenizerBase


INVALID_LOGPROB = 1.0


class PPOStats:
    def __init__(self, stats_shape: Tuple[int, int, int], device: torch.device):
        self.approximate_kl_stats = torch.zeros(stats_shape, device=device)
        self.policy_gradient_clipfrac_stats = torch.zeros(stats_shape, device=device)
        self.policy_gradient_loss_stats = torch.zeros(stats_shape, device=device)
        self.value_function_loss_stats = torch.zeros(stats_shape, device=device)
        self.value_function_clipfrac_stats = torch.zeros(stats_shape, device=device)
        self.total_loss_stats = torch.zeros(stats_shape, device=device)
        self.entropy_stats = torch.zeros(stats_shape, device=device)
        self.ratio_stats = torch.zeros(stats_shape, device=device)

    def update(
        self,
        update_location: Tuple[int, int, int],
        approximate_kl_mean: torch.Tensor,
        policy_gradient_clipfrac: torch.Tensor,
        policy_gradient_loss: torch.Tensor,
        state_value_function_loss: torch.Tensor,
        state_value_function_clipfrac: torch.Tensor,
        total_loss: torch.Tensor,
        entropy_mean: torch.Tensor,
        new_prev_prob_ratio_mean: torch.Tensor,
    ):
        self.approximate_kl_stats[update_location] = approximate_kl_mean
        self.policy_gradient_clipfrac_stats[update_location] = policy_gradient_clipfrac
        self.policy_gradient_loss_stats[update_location] = policy_gradient_loss
        self.value_function_loss_stats[update_location] = state_value_function_loss
        self.value_function_clipfrac_stats[update_location] = (
            state_value_function_clipfrac
        )
        self.total_loss_stats[update_location] = total_loss
        self.entropy_stats[update_location] = entropy_mean
        self.ratio_stats[update_location] = new_prev_prob_ratio_mean


def ppo_micro_batch_updates(
    config: PPOConfig,
    epoch_idx,
    minibatch_idx,
    mini_batch_start: int,
    model,
    advantages,
    responses,
    query_responses,
    logprobs,
    returns,
    state_values,
    padding_mask,
    padding_mask_plus_one,
    context_length: int,
    pad_token_id: int,
    batch_inds: np.ndarray,
    accelerator: Accelerator,
    optimizer,
    stats: PPOStats,
):
    mini_batch_end = mini_batch_start + config.local_mini_batch_size
    mini_batch_inds = batch_inds[mini_batch_start:mini_batch_end]
    gradient_accumulation_idx = 0

    for micro_batch_start in range(
        0, config.local_mini_batch_size, config.per_device_train_batch_size
    ):
        # I think that micro-batches are minibatches divided between machines.
        with accelerator.accumulate(model):
            micro_batch_end = micro_batch_start + config.per_device_train_batch_size
            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
            # Retrieve the relevant variables for this microbatch
            micro_batch_advantage = advantages[micro_batch_inds]
            micro_batch_responses = responses[micro_batch_inds]
            micro_batch_query_responses = query_responses[micro_batch_inds]
            micro_batch_logprobs = logprobs[micro_batch_inds]
            micro_batch_return = returns[micro_batch_inds]
            micro_batch_state_values = state_values[micro_batch_inds]

            output, state_value_prediction_temp = forward(
                model,
                micro_batch_query_responses,
                pad_token_id,
            )
            logits = output.logits[:, context_length - 1 : -1]
            logits /= config.train_temperature + 1e-7
            new_all_logprobs = F.log_softmax(logits, dim=-1)
            new_logprobs = torch.gather(
                new_all_logprobs, 2, micro_batch_responses.unsqueeze(-1)
            ).squeeze(-1)
            new_logprobs = torch.masked_fill(
                new_logprobs,
                padding_mask[micro_batch_inds],
                INVALID_LOGPROB,
            )

            # Compute the value loss term
            state_value_prediction = state_value_prediction_temp[
                :, context_length - 1 : -1
            ].squeeze(-1)
            state_value_prediction = torch.masked_fill(
                state_value_prediction, padding_mask_plus_one[micro_batch_inds], 0
            )
            state_value_prediction_clipped = torch.clamp(
                state_value_prediction,
                micro_batch_state_values - config.cliprange_value,
                micro_batch_state_values + config.cliprange_value,
            )
            state_value_function_losses_unclipped = torch.square(
                state_value_prediction - micro_batch_return
            )
            state_value_function_losses_clipped = torch.square(
                state_value_prediction_clipped - micro_batch_return
            )
            state_value_function_loss_max = torch.max(
                state_value_function_losses_unclipped,
                state_value_function_losses_clipped,
            )
            state_value_function_loss = 0.5 * masked_mean(
                state_value_function_loss_max,
                ~padding_mask_plus_one[micro_batch_inds],
            )
            state_value_function_clipfrac = masked_mean(
                (
                    state_value_function_losses_clipped
                    > state_value_function_losses_unclipped
                ).float(),
                ~padding_mask_plus_one[micro_batch_inds],
            )

            # Compute the policy gradient loss term.
            logprobs_diff = new_logprobs - micro_batch_logprobs
            new_prev_prob_ratio = torch.exp(logprobs_diff)
            policy_gradient_losses_unclipped = (
                -micro_batch_advantage * new_prev_prob_ratio
            )
            policy_gradient_losses_clipped = -micro_batch_advantage * torch.clamp(
                new_prev_prob_ratio, 1.0 - config.cliprange, 1.0 + config.cliprange
            )
            policy_gradient_losses_max = torch.max(
                policy_gradient_losses_unclipped,
                policy_gradient_losses_clipped,
            )
            policy_gradient_loss = masked_mean(
                policy_gradient_losses_max,
                ~padding_mask[micro_batch_inds],
            )
            total_loss = (
                policy_gradient_loss + config.vf_coef * state_value_function_loss
            )

            # Perform the update step.
            accelerator.backward(total_loss)
            optimizer.step()
            optimizer.zero_grad()

            # This is all just for logging.
            with torch.no_grad():
                policy_gradient_clipfrac = masked_mean(
                    (
                        policy_gradient_losses_clipped
                        > policy_gradient_losses_unclipped
                    ).float(),
                    ~padding_mask[micro_batch_inds],
                )

                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(
                    prob_dist * logits, dim=-1
                )
                avg_entropy = masked_mean(entropy, ~padding_mask[micro_batch_inds])

                approximate_kl = 0.5 * (logprobs_diff**2).mean()
                # create a slice object for the indices that we want to populate
                update_location = (
                    epoch_idx,
                    minibatch_idx,
                    gradient_accumulation_idx,
                )
                stats.update(
                    update_location,
                    approximate_kl,
                    policy_gradient_clipfrac,
                    policy_gradient_loss,
                    state_value_function_loss,
                    state_value_function_clipfrac,
                    total_loss,
                    avg_entropy,
                    new_prev_prob_ratio.mean(),
                )
        gradient_accumulation_idx += 1


def ppo_batch_update(
    config,
    generation_config,
    scoring_function,
    processing_class,
    # GPU related things
    device: torch.device,
    accelerator,
    optimizer,
    # stateful parameters
    model,
    ref_policy,
    ppo_stats,
    # data for this batch
    data,
    # Scheduled parameters
    lam,
):

    start_time = time.perf_counter()

    # Generate a PPO dataset for this update phase
    with torch.no_grad():
        queries = data["input_ids"].to(device)
        maybe_answer_ids = data.get("answer_ids")

        if maybe_answer_ids is not None:
            maybe_answer_ids = maybe_answer_ids.to(device)

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
            logprobs,
            ref_logprobs,
            sequence_lengths,
            scores,
            state_values,
        ) = rollouts_to_loss_variables(
            queries=queries,
            query_responses=query_responses,
            maybe_answer_ids=maybe_answer_ids,
            logitss=logitss,
            ref_policy=ref_policy,
            unwrapped_value_model=accelerator.unwrap_model(model).value_model,
            processing_class=processing_class,
            context_length=context_length,
            stop_token_id=config.stop_token_id,
            response_truncation_sequences=config.response_truncation_sequences,
            local_rollout_forward_batch_size=config.local_rollout_forward_batch_size,
            ref_temperature=config.train_temperature,
            scoring_function=scoring_function,
        )
        torch.cuda.empty_cache()
        gc.collect()

        # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
        # Completions not passing that filter will receive a lower score.
        contain_eos_token = torch.any(
            postprocessed_responses == processing_class.eos_token_id,
            dim=-1,
        )
        if config.missing_eos_penalty is not None:
            scores[~contain_eos_token] -= config.missing_eos_penalty
        # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

        # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
        response_idxs = torch.arange(
            responses.shape[1], device=responses.device
        ).repeat(responses.shape[0], 1)
        padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
        logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
        ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
        sequence_lengths_plus_one = sequence_lengths + 1
        padding_mask_plus_one = response_idxs > (sequence_lengths_plus_one.unsqueeze(1))
        state_values = torch.masked_fill(state_values, padding_mask_plus_one, 0)

        # 4. compute rewards
        prev_ref_log_ratio = logprobs - ref_logprobs
        prev_ref_log_ratio = torch.masked_fill(
            prev_ref_log_ratio, padding_mask, 0
        )  # Set the log-ratio to 0 for padding tokens.
        non_score_reward = -config.kl_coef * prev_ref_log_ratio
        # rewards has shape [batch, response_length]
        rewards = non_score_reward.clone()
        batch_indices = torch.arange(rewards.size(0), device=rewards.device)
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

        # 6. compute advantages and returns
        # Initialise the GAE at 0 for the last time step.
        last_gae = 0
        advantages_reversed = []
        gen_length = responses.shape[1]

        for t in reversed(range(gen_length)):
            nextvalues = state_values[:, t + 1] if t < gen_length - 1 else 0.0
            # Compute the TD-error
            delta = rewards[:, t] + config.gamma * nextvalues - state_values[:, t]
            # Use the GAE backwards recursion relationship
            last_gae = delta + config.gamma * lam * last_gae
            advantages_reversed.append(last_gae)

        # Create the advantage estimates by reversing the GAE backward recursion
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        # Set the return estimates to be the advantage estimates
        returns = advantages + state_values
        # Whiten the advantages. Note that this is *non-optional* and *done at the entire batch level*
        advantages = masked_whiten(advantages, ~padding_mask)
        advantages = torch.masked_fill(advantages, padding_mask, 0)
        torch.cuda.empty_cache()

    processing_stop_time = time.perf_counter()
    processing_time = processing_stop_time - generation_stop_time

    # Do multiple epochs of PPO training, using the dataset for this update phase
    # use a fresh random shuffle in each epoch
    # num_epochs_per_batch_update specifies how many times to loop over the PPO dataset.
    for epoch_idx in range(config.num_epochs_per_batch_update):
        # Draw a random permutation
        batch_inds = np.random.permutation(config.local_batch_size)
        for minibatch_idx, mini_batch_start in enumerate(
            range(0, config.local_batch_size, config.local_mini_batch_size)
        ):
            ppo_micro_batch_updates(
                # config!
                config=config,
                # integers
                epoch_idx=epoch_idx,
                minibatch_idx=minibatch_idx,
                mini_batch_start=mini_batch_start,
                context_length=context_length,
                pad_token_id=processing_class.pad_token_id,
                # tensors
                batch_inds=batch_inds,
                advantages=advantages,
                responses=responses,
                query_responses=query_responses,
                logprobs=logprobs,
                returns=returns,
                state_values=state_values,
                padding_mask=padding_mask,
                padding_mask_plus_one=padding_mask_plus_one,
                # Stateful parameters that get updated
                model=model,
                accelerator=accelerator,
                optimizer=optimizer,
                stats=ppo_stats,
            )
            torch.cuda.empty_cache()

    training_stop_time = time.perf_counter()
    training_time = training_stop_time - processing_stop_time

    # At the end of the update phase, log a bunch of statistics in the metrics dictionary.
    with torch.no_grad():
        # Depreciated. This was just the sampled log ratio.
        # mean_kl = kl.sum(1).mean()

        # This *does not* compute the entropy.
        # logprobs has shape [batch, response_length]
        # So this computes the total negative log likelihood of the responses.
        # Additionally, because the logprobs are filled with INVALID_LOGPROB, the sum will be invalid.
        # As a result, this is pretty meaningless.
        # mean_entropy = (-logprobs).sum(1).mean()
        mean_non_score_reward = non_score_reward.sum(1).mean()
        mean_rlhf_reward = mean_non_score_reward + scores.mean()

        metrics = {}

        # Refactor the metric caching to make it easier to parse
        s = ppo_stats
        metrics_gathered_and_meaned = {
            # "objective/kl": mean_kl, # no longer logging this, because it's badly named.
            # "objective/entropy": mean_entropy, # no longer logging this, because it's meaningless.
            "objective/traj/non_score_reward": mean_non_score_reward,
            "objective/traj/rlhf_reward": mean_rlhf_reward,
            "objective/traj/scores": scores.mean(),
            #
            "policy/token/entropy": s.entropy_stats,
            "policy/traj/prev_ref_log_ratio": prev_ref_log_ratio.sum(1).mean(),
            "policy/approximate_kl_avg": s.approximate_kl_stats,
            "policy/token/clipfrac_avg": s.policy_gradient_clipfrac_stats,
            #
            "loss/token/policy_loss": s.policy_gradient_loss_stats,
            "loss/token/state_value_loss": s.value_function_loss_stats,
            "loss/token/total_loss": s.total_loss_stats,
            #
            "val/clipfrac_avg": s.value_function_clipfrac_stats,
            "val/new_prev_prob_ratio": s.ratio_stats,
        }
        for m_name, m_tensor in metrics_gathered_and_meaned.items():
            metrics[m_name] = accelerator.gather(m_tensor).mean().item()

        metrics["val/new_prev_prob_ratio_var"] = (
            accelerator.gather(s.ratio_stats).var().item()
        )
        metrics["objective/traj/num_eos_tokens"] = (
            (responses == processing_class.eos_token_id).sum().item()
        )

    metrics["time/iteration/generation"] = generation_time
    metrics["time/iteration/processing"] = processing_time
    metrics["time/iteration/training"] = training_time

    return metrics


class PPOTrainer(OnPolicyTrainer):
    _tag_names = ["trl", "ppo"]

    def __init__(
        self,
        # config and tokenizers
        config: PPOConfig,
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
        assert (
            config.train_temperature == config.train_rollout_temperature
        ), "PPO does not support these being inequal."

    def _initialise_stats(self) -> PPOStats:
        stats_shape = (
            self.args.num_epochs_per_batch_update,
            self.args.local_batch_size,
            self.args.local_mini_batch_size,
        )
        return PPOStats(stats_shape, self.accelerator.device)

    def _batch_update(
        self,
        data: Dict[str, torch.Tensor],
        scoring_function: Callable,
    ) -> Dict[str, float]:
        return ppo_batch_update(
            # config-like and tokenizers
            config=self.args,
            generation_config=self.train_generation_config,
            scoring_function=scoring_function,
            processing_class=self.processing_class,
            # optimisation / performance
            optimizer=self.optimizer,
            accelerator=self.accelerator,
            device=self.accelerator.device,
            # stateful models and stats to be updated
            model=self.model,
            ref_policy=self.ref_policy,
            ppo_stats=self.stats,
            # data for this batch!
            data=data,
            # Scheduled parameters
            lam=self.lambda_scheduler.get(),
        )
