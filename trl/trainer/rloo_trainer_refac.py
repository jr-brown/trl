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
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import (
    CallbackHandler,
    ExportableState,
    PrinterCallback,
)
from trl.core import masked_whiten
from trl.trainer.on_policy_trainer import (
    OnPolicyStats,
    OnPolicyTrainer,
    ProcessingClass,
)
from trl.trainer.on_policy_utils import rollouts_to_loss_variables

from ..models.utils import unwrap_model_for_generation
from ..trainer.utils import (
    OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    get_reward,
    prepare_deepspeed,
    print_rich_table,
    truncate_response,
)
from .rloo_config import RLOOConfig
from .utils import generate_model_card


if is_wandb_available():
    import wandb

INVALID_LOGPROB = 1.0


class RLOOStats(OnPolicyStats):
    def __init__(self, stats_shape: Tuple[int, int, int], device: torch.device):
        super().__init__(stats_shape, device)
        self.approxkl_stats = torch.zeros(stats_shape, device=device)
        self.pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        self.pg_loss_stats = torch.zeros(stats_shape, device=device)
        self.vf_loss_stats = torch.zeros(stats_shape, device=device)
        self.vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        self.entropy_stats = torch.zeros(stats_shape, device=device)
        self.ratio_stats = torch.zeros(stats_shape, device=device)

    def update(
        self,
        update_location: Tuple[int, int, int],
        approxkl,
        pg_clipfrac,
        pg_loss,
        vf_loss,
        vf_clipfrac,
        entropy,
        ratio,
    ):
        self.approxkl_stats[update_location] = approxkl
        self.pg_clipfrac_stats[update_location] = pg_clipfrac
        self.pg_loss_stats[update_location] = pg_loss
        self.vf_loss_stats[update_location] = vf_loss
        self.vf_clipfrac_stats[update_location] = vf_clipfrac
        self.entropy_stats[update_location] = entropy
        self.ratio_stats[update_location] = ratio


def rloo_batch_update(
    config: RLOOConfig,
    generation_config: GenerationConfig,
    scoring_function: Callable,
    processing_class: ProcessingClass,
    # optimisation / performance
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    device: torch.device,
    # stateful models and stats to be updated
    model: nn.Module,
    ref_policy: nn.Module,
    rloo_stats: RLOOStats,
    # data for this batch!
    data: Dict[str, torch.Tensor],
    # Scheduled parameters
    lam: float,
):
    start_time = time.perf_counter()

    with torch.no_grad():
        queries = data["input_ids"].to(device)
        maybe_answer_ids = data.get("answer_ids")

        queries = queries.repeat(config.rloo_k, 1)
        maybe_answer_ids = maybe_answer_ids.repeat(config.rloo_k, 1)

        if maybe_answer_ids is not None:
            maybe_answer_ids = maybe_answer_ids.to(device)

        context_length = queries.shape[1]

        with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
            # query_responses and logitss are both torch Tensors.
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
        accelerator.print(
            f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}"
        )

        # be very careful with `padding_mask_plus_one`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
        response_idxs = torch.arange(
            responses.shape[1], device=responses.device
        ).repeat(responses.shape[0], 1)
        padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
        logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
        ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)
        sequence_lengths_plus_one = sequence_lengths + 1
        padding_mask_plus_one = response_idxs > (sequence_lengths_plus_one.unsqueeze(1))
        ## EDIT
        # state_values = torch.masked_fill(state_values, padding_mask_plus_one, 0)
        # action_values = torch.masked_fill(action_values, padding_mask_plus_one, 0)

        # 4. compute rewards
        prev_ref_log_ratio = logprobs - ref_logprobs  # This is just for logging!
        prev_ref_log_ratio = torch.masked_fill(
            prev_ref_log_ratio, padding_mask, 0
        )  # Set the log-ratio to 0 for padding tokens.

        # EDIT: simplified reeward calculation for RLOO
        # we have different shapes, highlighted in comments
        # previously 'rlhf'reward was only used for logging; here we use it for training
        # correspondingly, we end up with different variable names to the PPO/KLQ case
        non_score_reward = (-config.kl_coef * prev_ref_log_ratio).sum(1)  # [batch]
        rlhf_reward = scores + non_score_reward

        # vectorized RLOO advantages implementation
        rlhf_reward = rlhf_reward.reshape(config.rloo_k, -1)
        baseline = (rlhf_reward.sum(0) - rlhf_reward) / (config.rloo_k - 1)
        advantages = rlhf_reward - baseline
        advantages = advantages.flatten()

        # Normalize advantages
        if config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        torch.cuda.empty_cache()

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
    processing_stop_time = time.perf_counter()
    processing_time = processing_stop_time - generation_stop_time

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
                stats=rloo_stats,
            )
            torch.cuda.empty_cache()

    training_stop_time = time.perf_counter()
    training_time = training_stop_time - processing_stop_time

    # At the end of training, log a bunch of statistics in the metrics dictionary.
    with torch.no_grad():
        # The sum the non-score reward over the response length, and then take the mean over the batch.
        mean_non_score_reward = non_score_reward.sum(1).mean()
        # Compute the RLHF reward by adding the mean non-score reward to the mean score.
        mean_rlhf_reward = mean_non_score_reward + scores.mean()

        metrics = {}
        s = rloo_stats
        metrics_gathered_and_meaned = {
            "objective/traj/non_score_reward": mean_non_score_reward,
            "objective/traj/rlhf_reward": mean_rlhf_reward,
            "objective/traj/scores": scores.mean(),
            #
            "policy/token/entropy": s.entropy_stats,
            "policy/traj/prev_ref_log_ratio": s.prev_ref_log_ratio_stats,
            "policy/traj/prev_new_log_ratio": s.prev_new_log_ratio_stats,
            # "policy/kl_new_prev": s.kl_new_prev_stats,
            # "policy/kl_prev_new": s.kl_prev_new_stats,
            # "policy/kl_new_ref": s.kl_new_ref_stats,
            #
            "loss/token/action_value_loss": s.loss_function_stats,
            #
            "value_function/token/state_value": s.state_value_stats,
            "value_function/token/state_value_error": s.state_value_error_stats,
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


class RLOOTrainer(OnPolicyTrainer):
    _tag_names = ["trl", "klq"]

    def __init__(
        self,
        # config and tokenizers
        config: RLOOConfig,
        processing_class: ProcessingClass,
        reward_model_processing_class: ProcessingClass,
        # models
        policy: nn.Module,
        ref_policy: nn.Module,
        reward_model: nn.Module,
        ## note no value model for RLOO
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
            uses_value_model=False,
            value_model=None,
            train_dataset=train_dataset,
            data_collator=data_collator,
            eval_dataset=eval_dataset,
            optimizers=optimizers,
            callbacks=callbacks,
        )

    def _initialise_stats(self) -> RLOOStats:
        stats_shape = (
            self.args.num_epochs_per_batch_update,
            self.args.local_batch_size,
            self.args.local_mini_batch_size,
        )
        return RLOOStats(stats_shape, self.accelerator.device)

    def _batch_update(
        self,
        data: Dict[str, torch.Tensor],
        scoring_function: Callable,
    ) -> Dict[str, float]:
        return rloo_batch_update(
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
            klq_stats=self.stats,
            # data for this batch!
            data=data,
            # Scheduled parameters
            lam=self.lambda_scheduler.get(),
        )
