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
from .on_policy_utils import rollouts_to_loss_variables
from .on_policy_trainer import OnPolicyTrainer


ProcessingClass = PreTrainedTokenizerBase


INVALID_LOGPROB = 1.0


@dataclass
class LossFunctionTensors:
    action_value_prediction: torch.Tensor
    state_value_prediction: torch.Tensor
    new_ref_log_ratio: torch.Tensor
    micro_batch_prev_state_values: torch.Tensor
    prev_ref_log_ratio: torch.Tensor
    micro_batch_return: torch.Tensor


def l2_loss(
    config,
    tensors: LossFunctionTensors,
):
    return torch.square(tensors.action_value_prediction - tensors.micro_batch_return)


def huber_loss(
    config,
    tensors: LossFunctionTensors,
):
    huber_cutoff = config.loss_kwargs["huber_cutoff"]
    return F.huber_loss(
        tensors.action_value_prediction,
        tensors.micro_batch_return,
        reduction="none",
        delta=huber_cutoff,
    )


def value_clipped_loss(
    config,
    tensors: LossFunctionTensors,
):
    # Unload arguments
    kl_coef = config.kl_coef
    action_value_clip_bound = config.loss_kwargs["action_value_clip_bound"]

    # Compute the standard l2 loss
    unclipped_value_loss = torch.square(
        tensors.action_value_prediction - tensors.micro_batch_return
    )

    # Compute the clipped loss
    prev_action_value_prediction = (
        kl_coef * (tensors.prev_ref_log_ratio) + tensors.micro_batch_prev_state_values
    )
    clipped_aciton_value_prediction = torch.clamp(
        tensors.action_value_prediction,
        prev_action_value_prediction - action_value_clip_bound,
        prev_action_value_prediction + action_value_clip_bound,
    )
    clipped_value_loss = torch.square(
        clipped_aciton_value_prediction - tensors.micro_batch_return
    )

    # Return the maximum of the two losses
    return torch.max(unclipped_value_loss, clipped_value_loss)


def ratio_clipped_loss(
    config,
    tensors: LossFunctionTensors,
):
    # Unload arguments
    kl_coef = config.kl_coef
    log_ratio_clip_bound = config.loss_kwargs["log_ratio_clip_bound"]

    # Compute the standard l2 loss
    unclipped_value_loss = torch.square(
        tensors.action_value_prediction - tensors.micro_batch_return
    )

    # Compute the clipped loss
    clipped_log_ratio = torch.clamp(
        tensors.new_ref_log_ratio,
        tensors.prev_ref_log_ratio - log_ratio_clip_bound,
        tensors.prev_ref_log_ratio + log_ratio_clip_bound,
    )
    clipped_action_value_prediction = (
        kl_coef * (clipped_log_ratio) + tensors.state_value_prediction
    )
    clipped_value_loss = torch.square(
        clipped_action_value_prediction - tensors.micro_batch_return
    )

    # Return the maximum of the two losses
    return torch.max(unclipped_value_loss, clipped_value_loss)


def double_clipped_loss(
    config,
    tensors: LossFunctionTensors,
):
    # unload arguments
    kl_coef = config.kl_coef
    action_value_clip_bound = config.loss_kwargs["action_value_clip_bound"]
    log_ratio_clip_bound = config.loss_kwargs["log_ratio_clip_bound"]

    # Compute the standard l2 loss
    unclipped_value_loss = torch.square(
        tensors.action_value_prediction - tensors.micro_batch_return
    )

    # Compute the action-value clipped loss
    prev_action_value_prediction = (
        kl_coef * (tensors.prev_ref_log_ratio) + tensors.micro_batch_prev_state_values
    )
    action_value_clipped_aciton_value_prediction = torch.clamp(
        tensors.action_value_prediction,
        prev_action_value_prediction - action_value_clip_bound,
        prev_action_value_prediction + action_value_clip_bound,
    )
    action_value_clipped_value_loss = torch.square(
        action_value_clipped_aciton_value_prediction - tensors.micro_batch_return
    )

    # Compute the log-ratio clipped loss
    clipped_log_ratio = torch.clamp(
        tensors.new_ref_log_ratio,
        tensors.prev_ref_log_ratio - log_ratio_clip_bound,
        tensors.prev_ref_log_ratio + log_ratio_clip_bound,
    )
    log_ratio_clipped_action_value_prediction = (
        kl_coef * (clipped_log_ratio) + tensors.state_value_prediction
    )
    log_ratio_clipped_value_loss = torch.square(
        log_ratio_clipped_action_value_prediction - tensors.micro_batch_return
    )

    # Return the maximum of the three losses
    return torch.max(
        torch.max(unclipped_value_loss, action_value_clipped_value_loss),
        log_ratio_clipped_value_loss,
    )


# This type checking is fucked.
loss_function_map: dict[
    str,
    Callable[
        [
            KLQConfig,
            LossFunctionTensors,
        ],
        torch.Tensor,
    ],
] = {
    "l2_loss": l2_loss,
    "huber": huber_loss,
    "value_clipped": value_clipped_loss,
    "ratio_clipped": ratio_clipped_loss,
    "double_clipped": double_clipped_loss,
}


class KLQStats:
    def __init__(self, stats_shape: Tuple[int, int, int], device: torch.device) -> None:
        self.loss_function_stats = torch.zeros(stats_shape, device=device)
        self.action_value_stats = torch.zeros(stats_shape, device=device)
        self.state_value_stats = torch.zeros(stats_shape, device=device)
        self.log_ratio_new_ref_stats = torch.zeros(stats_shape, device=device)
        self.entropy_stats = torch.zeros(stats_shape, device=device)
        self.kl_prev_ref_stats = torch.zeros(stats_shape, device=device)
        self.kl_new_ref_stats = torch.zeros(stats_shape, device=device)
        self.kl_prev_new_stats = torch.zeros(stats_shape, device=device)
        self.kl_new_prev_stats = torch.zeros(stats_shape, device=device)

    def update(
        self,
        update_location: Tuple[int, int, int],
        action_value_function_loss,
        action_value_prediction,
        state_value_prediction,
        new_ref_log_ratio,
        entropy,
        kl_prev_ref,
        kl_new_ref,
        kl_prev_new,
        kl_new_prev,
    ):
        self.loss_function_stats[update_location] = action_value_function_loss
        self.action_value_stats[update_location] = action_value_prediction
        self.state_value_stats[update_location] = state_value_prediction
        self.log_ratio_new_ref_stats[update_location] = new_ref_log_ratio
        self.entropy_stats[update_location] = entropy
        self.kl_prev_ref_stats[update_location] = kl_prev_ref
        self.kl_new_ref_stats[update_location] = kl_new_ref
        self.kl_prev_new_stats[update_location] = kl_prev_new
        self.kl_new_prev_stats[update_location] = kl_new_prev


def micro_batch_updates(
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
            micro_batch_responses = responses[micro_batch_inds]
            micro_batch_query_responses = query_responses[micro_batch_inds]
            micro_batch_prev_logprobs = logprobs[micro_batch_inds]
            micro_batch_ref_logprobs = ref_logprobs[micro_batch_inds]
            micro_batch_return = returns[micro_batch_inds]
            micro_batch_prev_state_values = state_values[micro_batch_inds]

            output, state_value_prediction_temporary = forward(
                model, micro_batch_query_responses, pad_token_id
            )
            logits = output.logits[:, context_length - 1 : -1]
            logits /= config.temperature + 1e-7
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
            new_ref_log_ratio = new_logprobs - micro_batch_ref_logprobs
            prev_ref_log_ratio = micro_batch_prev_logprobs - micro_batch_ref_logprobs
            action_value_prediction = (
                config.kl_coef * (new_ref_log_ratio) + state_value_prediction
            )

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
                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(
                    prob_dist * logits, dim=-1
                )
                prev_new_log_ratio = micro_batch_prev_logprobs - new_logprobs
                prev_ref_log_ratio = micro_batch_prev_logprobs - ref_logprobs
                new_prev_log_ratio = -prev_new_log_ratio
                new_prev_ratio = torch.exp(new_prev_log_ratio)

                update_location = (
                    epoch_idx,
                    minibatch_idx,
                    gradient_accumulation_idx,
                )
                stats.update(
                    update_location,
                    action_value_function_loss=action_value_function_loss.mean(),
                    action_value_prediction=action_value_prediction.mean(),
                    state_value_prediction=state_value_prediction.mean(),
                    new_ref_log_ratio=new_ref_log_ratio.mean(),
                    entropy=entropy.mean(),
                    kl_prev_ref=prev_ref_log_ratio.mean(),
                    kl_new_ref=(new_prev_ratio * new_ref_log_ratio).mean(),
                    kl_prev_new=prev_new_log_ratio.mean(),
                    kl_new_prev=(new_prev_ratio * new_prev_log_ratio).mean(),
                )
        gradient_accumulation_idx += 1
    minibatch_idx += 1


def klq_batch_update(
    config: KLQConfig,
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
    klq_stats: KLQStats,
    # data for the this batch!
    data: Dict[str, Tensor],
):
    with torch.no_grad():
        queries = data["input_ids"].to(device)
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
            ref_temperature=config.ref_temperature,
            device=device,
        )
        action_values = config.kl_coef * (logprobs - ref_logprobs) + state_values
        torch.cuda.empty_cache()
        gc.collect()

        # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
        # Completions not passing that filter will receive a lower score.
        contain_eos_token = torch.any(
            postprocessed_responses == processing_class.eos_token_id, dim=-1
        )
        if config.missing_eos_penalty is not None:
            scores[~contain_eos_token] -= config.missing_eos_penalty
        # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

        # be very careful with `padding_mask_plus_one`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
        response_idxs = torch.arange(
            responses.shape[1], device=responses.device
        ).repeat(responses.shape[0], 1)
        padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
        logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
        ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
        sequence_lengths_plus_one = sequence_lengths + 1
        padding_mask_plus_one = response_idxs > (sequence_lengths_plus_one.unsqueeze(1))
        state_values = torch.masked_fill(state_values, padding_mask_plus_one, 0)
        action_values = torch.masked_fill(action_values, padding_mask_plus_one, 0)

        # 4. compute rewards
        # rewards has shape [batch, response_length]
        rewards = torch.zeros_like(logprobs)
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

        # 6. compute advantages and returns
        # Initialise the GAE at 0 for the last time step.
        last_gae = 0
        advantages_reversed = []
        gen_length = responses.shape[1]  # This is the length of the responses.
        for t in reversed(range(gen_length)):
            # Extract the next token state-values
            next_state_values = state_values[:, t + 1] if t < gen_length - 1 else 0.0
            # Compute the TD-error
            delta = (
                rewards[:, t] + config.gamma * next_state_values - action_values[:, t]
            )
            # Use the GAE backwards recursion relationship
            last_gae = delta + config.gamma * config.lam * last_gae
            advantages_reversed.append(last_gae)
        # Create the advantage estimates by reversing the GAE backward recursion
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        # Set the return estimates to be the advantage estimates
        returns = advantages + action_values  # This used to be state_values
        returns = torch.masked_fill(returns, padding_mask_plus_one, 0)  # BUGHOTSPOT

        # Whiten the advantages. Note that this is *non-optional* and *done at the entire batch level*
        # advantages = masked_whiten(advantages, ~padding_mask)
        # advantages = torch.masked_fill(advantages, padding_mask, 0)

        # We only want the returns, so delete all other variables.
        del (
            rewards,
            last_gae,
            advantages_reversed,
            delta,
            next_state_values,
            advantages,
        )
        torch.cuda.empty_cache()

    # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
    # num_epochs_per_batch_update specifies how many times to loop over the PPO dataset.
    for epoch_idx in range(config.num_epochs_per_batch_update):
        # Draw a random permutation
        batch_inds = np.random.permutation(config.local_batch_size)
        minibatch_idx = 0
        for mini_batch_start in range(
            0, config.local_batch_size, config.local_mini_batch_size
        ):
            micro_batch_updates(
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
                logprobs=logprobs,
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

    # At the end of training, log a bunch of statistics in the metrics dictionary.
    with torch.no_grad():
        mean_entropy = (-logprobs).sum(1).mean()

        metrics = {}
        s = klq_stats
        metrics_gathered_and_meaned = {
            "objective/scores": scores.mean(),
            "objective/kl_prev_ref": s.kl_prev_ref_stats,
            "loss/avg_value_loss": s.loss_function_stats,
            "value_function/avg_state_value": s.state_value_stats,
            "value_function/avg_action_value": s.action_value_stats,
            "value_function/avg_log_ratio": s.log_ratio_new_ref_stats,
            "policy/entropy": s.entropy_stats,
            "policy/kl_new_prev": s.kl_new_prev_stats,
            "policy/kl_prev_new": s.kl_prev_new_stats,
            "policy/kl_new_ref": s.kl_new_ref_stats,
        }
        for m_name, m_tensor in metrics_gathered_and_meaned.items():
            metrics[m_name] = accelerator.gather(m_tensor).mean().item()

        # Metrics that do not need gathering
        metrics["objective/num_eos_tokens"] = (
            (responses == processing_class.eos_token_id).sum().item()
        )
        metrics["loss/avg_target_values"] = returns.mean().item()
        metrics["loss/var_target_values"] = returns.var().item()

    return metrics


class KLQTrainer(OnPolicyTrainer):
    _tag_names = ["trl", "klq"]

    def __init__(
        self,
        # config and tokenizers
        config: KLQConfig,
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

    def _initialise_stats(self) -> KLQStats:
        stats_shape = (
            self.args.num_epochs_per_batch_update,
            self.args.local_batch_size,
            self.args.local_mini_batch_size,
        )
        return KLQStats(stats_shape, self.accelerator.device)

    def _batch_update(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        return klq_batch_update(
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
            data=data,
        )
