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
import math
import os
import textwrap
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor # for type hinting
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    GenerationConfig,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback

from ..core import masked_mean, masked_whiten
from ..models.utils import unwrap_model_for_generation
from ..trainer.utils import (
    OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    forward,
    retokenize,
    prepare_deepspeed,
    print_rich_table,
    truncate_response,
)
from .klq_config import KLQConfig
from .utils import generate_model_card
from .on_policy_utils import get_just_reward, forward_rollout


if is_wandb_available():
    import wandb



ProcessingClass = PreTrainedTokenizerBase


INVALID_LOGPROB = 1.0


def l2_loss(config, action_value_prediction, state_value_prediction, new_ref_log_ratio, micro_batch_prev_state_values, prev_ref_log_ratio, micro_batch_return):
    return torch.square(action_value_prediction - micro_batch_return)

def huber_loss(config, action_value_prediction, state_value_prediction, new_ref_log_ratio, micro_batch_prev_state_values, prev_ref_log_ratio, micro_batch_return):
    huber_cutoff = config.loss_kwargs["huber_cutoff"]
    return F.huber_loss(action_value_prediction, micro_batch_return, reduction='none', delta=huber_cutoff)

def value_clipped_loss(config, action_value_prediction, state_value_prediction, new_ref_log_ratio, micro_batch_prev_state_values, prev_ref_log_ratio, micro_batch_return):
    # Unload arguments
    kl_coef = config.kl_coef
    action_value_clip_bound = config.loss_kwargs["action_value_clip_bound"]

    # Compute the standard l2 loss
    unclipped_value_loss = torch.square(action_value_prediction - micro_batch_return)

    # Compute the clipped loss
    prev_action_value_prediction = kl_coef*(prev_ref_log_ratio) + micro_batch_prev_state_values
    clipped_aciton_value_prediction = torch.clamp(action_value_prediction, prev_action_value_prediction - action_value_clip_bound, prev_action_value_prediction + action_value_clip_bound)
    clipped_value_loss = torch.square(clipped_aciton_value_prediction - micro_batch_return)

    # Return the maximum of the two losses
    return torch.max(unclipped_value_loss, clipped_value_loss)

def ratio_clipped_loss(config, action_value_prediction, state_value_prediction, new_ref_log_ratio, micro_batch_prev_state_values, prev_ref_log_ratio, micro_batch_return):
    # Unload arguments
    kl_coef = config.kl_coef
    log_ratio_clip_bound = config.loss_kwargs["log_ratio_clip_bound"]

    # Compute the standard l2 loss
    unclipped_value_loss = torch.square(action_value_prediction - micro_batch_return)

    # Compute the clipped loss
    clipped_log_ratio = torch.clamp(new_ref_log_ratio, prev_ref_log_ratio - log_ratio_clip_bound, prev_ref_log_ratio + log_ratio_clip_bound)
    clipped_action_value_prediction = kl_coef*(clipped_log_ratio) + state_value_prediction
    clipped_value_loss = torch.square(clipped_action_value_prediction - micro_batch_return)

    # Return the maximum of the two losses
    return torch.max(unclipped_value_loss, clipped_value_loss)

def double_clipped_loss(config, action_value_prediction, state_value_prediction, new_ref_log_ratio, micro_batch_prev_state_values, prev_ref_log_ratio, micro_batch_return):
    # unload arguments
    kl_coef = config.kl_coef
    action_value_clip_bound = config.loss_kwargs["action_value_clip_bound"]
    log_ratio_clip_bound = config.loss_kwargs["log_ratio_clip_bound"]

    # Compute the standard l2 loss
    unclipped_value_loss = torch.square(action_value_prediction - micro_batch_return)

    # Compute the action-value clipped loss
    prev_action_value_prediction = kl_coef*(prev_ref_log_ratio) + micro_batch_prev_state_values
    action_value_clipped_aciton_value_prediction = torch.clamp(action_value_prediction, prev_action_value_prediction - action_value_clip_bound, prev_action_value_prediction + action_value_clip_bound)
    action_value_clipped_value_loss = torch.square(action_value_clipped_aciton_value_prediction - micro_batch_return)

    # Compute the log-ratio clipped loss
    clipped_log_ratio = torch.clamp(new_ref_log_ratio, prev_ref_log_ratio - log_ratio_clip_bound, prev_ref_log_ratio + log_ratio_clip_bound)
    log_ratio_clipped_action_value_prediction = kl_coef*(clipped_log_ratio) + state_value_prediction
    log_ratio_clipped_value_loss = torch.square(log_ratio_clipped_action_value_prediction - micro_batch_return)

    # Return the maximum of the three losses
    return torch.max(torch.max(unclipped_value_loss, action_value_clipped_value_loss), log_ratio_clipped_value_loss)



# This type checking is fucked.
loss_function_map: dict[
    str,
    Callable[[KLQConfig, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
] = {
    "l2_loss": l2_loss,
    "huber": huber_loss,
    "value_clipped": value_clipped_loss,
    "ratio_clipped": ratio_clipped_loss,
    "double_clipped": double_clipped_loss,
}


# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(nn.Module):
    def __init__(self, policy, value_model) -> None:
        super().__init__()
        self.policy = policy
        self.value_model = value_model
        self.critic_backbone = getattr(value_model, value_model.base_model_prefix)

    def forward(self, **kwargs):
        output = self.critic_backbone(
            **kwargs,
        )
        logits = self.value_model.score(output.hidden_states[-1])
        return self.policy(**kwargs), logits

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
    for micro_batch_start in range(0, config.local_mini_batch_size, config.per_device_train_batch_size):
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

            output, state_value_prediction_temporary = forward(model, micro_batch_query_responses, pad_token_id)
            logits = output.logits[:, context_length - 1 : -1]
            logits /= config.temperature + 1e-7
            new_all_logprobs = F.log_softmax(logits, dim=-1)
            new_logprobs = torch.gather(new_all_logprobs, 2, micro_batch_responses.unsqueeze(-1)).squeeze(-1)
            new_logprobs = torch.masked_fill(
                new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
            )

            # Compute inputs to loss function
            state_value_prediction = state_value_prediction_temporary[:, context_length - 1 : -1].squeeze(-1)
            state_value_prediction = torch.masked_fill(state_value_prediction, padding_mask_plus_one[micro_batch_inds], 0)
            new_ref_log_ratio = new_logprobs - micro_batch_ref_logprobs
            prev_ref_log_ratio = micro_batch_prev_logprobs - micro_batch_ref_logprobs
            action_value_prediction = config.kl_coef*(new_ref_log_ratio) + state_value_prediction

            # Compute loss
            action_value_function_losses = loss_function_map[config.loss_function](
                config,
                action_value_prediction,
                state_value_prediction,
                new_ref_log_ratio,
                micro_batch_prev_state_values,
                prev_ref_log_ratio,
                micro_batch_return,
            )
            action_value_function_loss = masked_mean(
                action_value_function_losses,
                ~padding_mask_plus_one[micro_batch_inds]
            )

            # Perform the update step.
            accelerator.backward(action_value_function_loss)
            optimizer.step()
            optimizer.zero_grad()

            # This is all just for logging.
            with torch.no_grad():
                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
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


def batch_update(
    config: KLQConfig,
    generation_config,
    processing_class,
    reward_model_processing_class,
    # GPU related things
    device: torch.device,
    accelerator,
    optimizer,
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
        responses = []
        postprocessed_responses = []
        logprobs = []
        ref_logprobs = []
        scores = []
        sequence_lengths = []
        state_values = []
        action_values = []
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
        ) = forward_rollout(
            queries=queries,
            query_responses=query_responses,
            logitss=logitss,
            ref_policy=ref_policy,
            unwrapped_value_model=accelerator.unwrap_model(model).value_model,
            reward_model=reward_model,
            processing_class=processing_class,
            reward_model_processing_class=reward_model_processing_class,
            context_length=context_length,
            pad_token_id=processing_class.pad_token_id,
            stop_token_id=config.stop_token_id,
            local_rollout_forward_batch_size=config.local_rollout_forward_batch_size,
            temperature=config.temperature,
            device=device,
        )
        action_values = config.kl_coef * (logprobs - ref_logprobs) + state_values
        torch.cuda.empty_cache()
        gc.collect()

        # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
        # Completions not passing that filter will receive a lower score.
        contain_eos_token = torch.any(postprocessed_responses == processing_class.eos_token_id, dim=-1)
        if config.missing_eos_penalty is not None:
            scores[~contain_eos_token] -= config.missing_eos_penalty
        # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

        # be very careful with `padding_mask_plus_one`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
        response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
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
        batch_indices = torch.arange(rewards.size(0), device=rewards.device) # [0, 1, 2, ..., batch_size - 1]
        sequence_end_indices = torch.where(sequence_lengths_plus_one < rewards.size(1), sequence_lengths_plus_one, sequence_lengths)
        rewards[[batch_indices, sequence_end_indices]] += scores

        # 5. whiten rewards
        if config.whiten_rewards:
            rewards = masked_whiten(rewards, mask=~padding_mask_plus_one, shift_mean=False)
            rewards = torch.masked_fill(rewards, padding_mask_plus_one, 0)

        # 6. compute advantages and returns
        # Initialise the GAE at 0 for the last time step.
        last_gae = 0
        advantages_reversed = []
        gen_length = responses.shape[1] # This is the length of the responses.
        for t in reversed(range(gen_length)):
            # Extract the next token state-values
            next_state_values = state_values[:, t + 1] if t < gen_length - 1 else 0.0
            # Compute the TD-error
            delta = rewards[:, t] + config.gamma * next_state_values - action_values[:, t]
            # Use the GAE backwards recursion relationship
            last_gae = delta + config.gamma * config.lam * last_gae
            advantages_reversed.append(last_gae)
        # Create the advantage estimates by reversing the GAE backward recursion
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        # Set the return estimates to be the advantage estimates
        returns = advantages + state_values
        returns = torch.masked_fill(returns, padding_mask_plus_one, 0) # BUGHOTSPOT

        # Whiten the advantages. Note that this is *non-optional* and *done at the entire batch level*
        # advantages = masked_whiten(advantages, ~padding_mask)
        # advantages = torch.masked_fill(advantages, padding_mask, 0)

        # We only want the returns, so delete all other variables.
        del (rewards, last_gae, advantages_reversed, delta, next_state_values, advantages)
        torch.cuda.empty_cache()

    # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
    # num_ppo_epochs specifies how many times to loop over the PPO dataset.
    for klq_epoch_idx in range(config.num_klq_epochs):
        # Draw a random permutation
        batch_inds = np.random.permutation(config.local_batch_size)
        minibatch_idx = 0
        for mini_batch_start in range(0, config.local_batch_size, config.local_mini_batch_size):
            micro_batch_updates(
                config=config,
                # integers
                epoch_idx=klq_epoch_idx,
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
            'objective/scores': scores.mean(),
            'objective/kl_prev_ref': s.kl_prev_ref_stats,
            'loss/avg_value_loss': s.loss_function_stats,
            'value_function/avg_state_value': s.state_value_stats,
            'value_function/avg_action_value': s.action_value_stats,
            'value_function/avg_log_ratio': s.log_ratio_new_ref_stats,
            'policy/entropy': s.entropy_stats,
            'policy/kl_new_prev': s.kl_new_prev_stats,
            'policy/kl_prev_new': s.kl_prev_new_stats,
            'policy/kl_new_ref': s.kl_new_ref_stats,
        }
        for m_name, m_tensor in metrics_gathered_and_meaned.items():
            metrics[m_name] = accelerator.gather(m_tensor).mean().item()

        # Metrics that do not need gathering
        metrics["objective/num_eos_tokens"] = (responses == processing_class.eos_token_id).sum().item()
        metrics["loss/avg_target_values"] = returns.mean().item()
        metrics["loss/var_target_values"] = returns.var().item()

class KLQTrainer(Trainer):
    _tag_names = ["trl", "klq"]

    def __init__(
        self,
        config: KLQConfig,
        processing_class: Optional[ProcessingClass],
        policy: nn.Module,
        ref_policy: nn.Module,
        reward_model: nn.Module,
        train_dataset: Dataset,
        value_model: Optional[nn.Module] = None,
        reward_model_processing_class: Optional[ProcessingClass] = None,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[List[TrainerCallback]] = None,
    ) -> None:
        if ref_policy is policy:
            raise ValueError(
                "`policy` and `ref_policy` cannot be the same object. If you want `ref_policy` to be the "
                "same as `policy`, you must mass a copy of it, or `None` if you use peft."
            )

        self.args = config
        self.processing_class = processing_class
        self.policy = policy

        self.policy.generation_config.eos_token_id = (
            None  # disable `pad_token_id` and `eos_token_id` because we just want to
        )
        self.policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding

        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.value_model = value_model
        self.reward_model_processing_class = reward_model_processing_class
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers

        #########
        # calculate various batch sizes
        #########

        # One "episode" is one prompt-completion pair
        if config.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            config.total_episodes = int(config.num_train_epochs * self.train_dataset_len)
        accelerator = Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps)
        self.accelerator = accelerator

        # The number of processes we're using
        config.world_size = accelerator.num_processes
        # per_device_train_batch_size
        # gradient_accumulation_steps
        # num_mini_batches
        config.local_batch_size = (
            config.per_device_train_batch_size * config.gradient_accumulation_steps * config.num_mini_batches
        )
        # Total batch size across all processes
        config.micro_batch_size = int(config.per_device_train_batch_size * config.world_size)
        config.batch_size = int(config.local_batch_size * config.world_size)
        config.mini_batch_size = exact_div(
            config.batch_size, config.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        config.local_mini_batch_size = exact_div(
            config.local_batch_size, config.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        if config.whiten_rewards:
            assert (
                config.local_mini_batch_size >= 8
            ), f"Per-rank minibatch size {config.local_mini_batch_size} is insufficient for whitening"
        # `per_rank_rollout_batch_size` is our `config.local_batch_size`
        # `per_rank_minibatch_size` is our `config.local_mini_batch_size`
        config.num_total_batches = math.ceil(
            config.total_episodes / config.batch_size
        )  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        config.run_name = f"{config.exp_name}__{config.seed}__{time_int}"
        self.local_seed = config.seed + accelerator.process_index * 100003  # Prime
        if config.num_sample_generations > 0:
            self.sample_generations_freq = max(1, config.num_total_batches // config.num_sample_generations)
        self.local_dataloader_batch_size = config.local_batch_size

        #########
        # setup model, optimizer, and others
        #########
        for module in [policy, ref_policy, value_model, reward_model]:
            disable_dropout_in_model(module)
        if config.stop_token and config.stop_token == "eos":
            config.stop_token_id = processing_class.eos_token_id
        self.model = PolicyAndValueWrapper(policy, value_model)
        self.model.config = policy.config  # needed for pushing to hub
        self.create_optimizer_and_scheduler(
            num_training_steps=config.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        ### trainer specifics
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(config.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if config.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )
        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if config.push_to_hub:
            self.init_hf_repo()
        if config.should_save:
            os.makedirs(config.output_dir, exist_ok=True)

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=DataCollatorWithPadding(self.processing_class),
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(config.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=config.per_device_eval_batch_size,
            collate_fn=DataCollatorWithPadding(self.processing_class),
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if self.is_deepspeed_enabled:
            self.reward_model = prepare_deepspeed(
                self.reward_model, config.per_device_train_batch_size, config.fp16, config.bf16
            )
            self.ref_policy = prepare_deepspeed(
                self.ref_policy, config.per_device_train_batch_size, config.fp16, config.bf16
            )
        else:
            self.ref_policy = self.ref_policy.to(self.accelerator.device)
            self.reward_model = self.reward_model.to(self.accelerator.device)

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        backup_model = self.model
        self.model = self.model.policy  # save only the policy

        if self.is_deepspeed_enabled:
            backup_deepspeed = self.deepspeed
            self.deepspeed = self.model

        super().save_model(output_dir, _internal_call)

        self.model = backup_model

        if self.is_deepspeed_enabled:
            self.deepspeed = backup_deepspeed

    def train(self):
        config = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        ref_policy = self.ref_policy
        reward_model = self.reward_model
        processing_class = self.processing_class
        assert processing_class is not None
        reward_model_processing_class = self.reward_model_processing_class
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=config.response_length,
            temperature=(config.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        accelerator.print("===training policy===")
        start_time = time.time()
        # num_ppo_epochs is the number of epochs for which we train on the PPO dataset for each increment of PPO
        # num_mini_batches
        # gradient_accumulation_steps

        stats_shape = (
            config.num_klq_epochs, 
            config.num_mini_batches, 
            config.gradient_accumulation_steps
        )
        klq_stats = KLQStats(stats_shape, device)
        # Define a collection of tensors which track statistics over training
        loss_function_stats = torch.zeros(stats_shape, device=device)
        action_value_stats = torch.zeros(stats_shape, device=device)
        state_value_stats = torch.zeros(stats_shape, device=device)
        log_ratio_new_ref_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        kl_prev_ref_stats = torch.zeros(stats_shape, device=device)
        kl_new_ref_stats = torch.zeros(stats_shape, device=device)
        kl_prev_new_stats = torch.zeros(stats_shape, device=device)
        kl_new_prev_stats = torch.zeros(stats_shape, device=device)

        # Put the model into train mode.
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = config.num_total_batches * config.num_mini_batches
        self.state.num_train_epochs = config.total_episodes / self.train_dataset_len
        # Compute absolute state_values for logging, eval, and save if given as ratio
        if config.logging_steps is not None:
            if config.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * config.logging_steps)
            else:
                self.state.logging_steps = config.logging_steps
        if config.eval_steps is not None:
            if config.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * config.eval_steps)
            else:
                self.state.eval_steps = config.eval_steps
        if config.save_steps is not None:
            if config.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * config.save_steps)
            else:
                self.state.save_steps = config.save_steps
        self.control = self.callback_handler.on_train_begin(config, self.state, self.control)

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        # The actual training loop
        for update in range(1, config.num_total_batches + 1):
            self.state.episode += 1 * config.batch_size
            data = next(iter_dataloader)
            model, metrics = batch_update(
                config=config,
                generation_config=generation_config,
                processing_class=processing_class,
                reward_model_processing_class=reward_model_processing_class,
                device=device,
                accelerator=accelerator,
                optimizer=optimizer,
                model=model,
                ref_policy=ref_policy,
                reward_model=reward_model,
                klq_stats=klq_stats,
                data=data,
            )

            # metrics which are cleaner to update outside of the batch_update function
            eps = int(self.state.episode / (time.time() - start_time))
            metrics["eps"] = eps
            metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
            metrics["episode"] = self.state.episode
            
            self.state.epoch = self.state.episode / self.train_dataset_len  # used by self.log
            self.state.global_step += 1
            self.log(metrics)

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(config, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None, metrics=metrics)
                self.control = self.callback_handler.on_save(config, self.state, self.control)

            torch.cuda.empty_cache()
            gc.collect()

            if config.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)
                torch.cuda.empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(config, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(config, self.state, self.control)

    def generate_completions(self, sampling: bool = False):
        config = self.args
        processing_class = self.processing_class
        reward_model_processing_class = self.reward_model_processing_class
        generation_config = GenerationConfig(
            max_new_tokens=config.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        table = defaultdict(list)
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            for batch in self.eval_dataloader:
                query = batch["input_ids"]
                with torch.no_grad():
                    context_length = query.shape[1]
                    query_response, _ = batch_generation(
                        unwrapped_model.policy,
                        query,
                        query.shape[0],
                        processing_class.pad_token_id,
                        generation_config,
                    )
                    response = query_response[:, context_length:]
                    postprocessed_response = response
                    if config.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            config.stop_token_id, processing_class.pad_token_id, response
                        )
                    table["query"].extend(
                        gather_object(processing_class.batch_decode(query, skip_special_tokens=True))
                    )
                    table["model response"].extend(
                        gather_object(processing_class.batch_decode(postprocessed_response))
                    )

                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    reward_model_inputs, reward_model_pad_token = retokenize(
                        postprocessed_query_response,
                        self.accelerator.device,
                        processing_class,
                        reward_model_processing_class,
                    )
                    score = get_just_reward(
                        self.reward_model,
                        reward_model_inputs,
                        reward_model_pad_token,
                        context_length,
                    )
                    table["score"].extend(self.accelerator.gather(score).float().cpu().numpy())

                if sampling:
                    break

        df = pd.DataFrame(table)

        if self.accelerator.is_main_process:
            print_rich_table(df.iloc[0 : 0 + 5])
            if "wandb" in config.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, List[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str`, *optional*, defaults to `None`):
                The name of the model.
            dataset_name (`str`, *optional*, defaults to `None`):
                The name of the dataset used for training.
            tags (`str`, `List[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent("""\
        @article{mziegler2019fine-tuning,
            title        = {{Fine-Tuning Language Models from Human Preferences}},
            author       = {Daniel M. Ziegler and Nisan Stiennon and Jeffrey Wu and Tom B. Brown and Alec Radford and Dario Amodei and Paul F. Christiano and Geoffrey Irving},
            year         = 2019,
            eprint       = {arXiv:1909.08593}
        }""")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            trainer_name="PPO",
            trainer_citation=citation,
            paper_title="Fine-Tuning Language Models from Human Preferences",
            paper_id="1909.08593",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
