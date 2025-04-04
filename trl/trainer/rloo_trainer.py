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
from torch import Tensor
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
from trl.core import masked_mean, masked_whiten
from trl.trainer.on_policy_trainer import (
    ModelCardInfo,
    OnPolicyStats,
    OnPolicyTrainer,
    ProcessingClass,
)
from trl.trainer.on_policy_utils import rloo_rollouts_to_loss_variables

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
        self.pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        self.pg_loss_stats = torch.zeros(stats_shape, device=device)
        self.entropy_stats = torch.zeros(stats_shape, device=device)
        self.prev_ref_log_ratio_stats = torch.zeros(stats_shape, device=device)
        self.prev_new_log_ratio_stats = torch.zeros(stats_shape, device=device)

    def update(
        self,
        update_location: Tuple[int, int, int],
        pg_clipfrac,
        pg_loss,
        entropy,
        prev_ref_log_ratio,
        prev_new_log_ratio,
    ):
        self.pg_clipfrac_stats[update_location] = pg_clipfrac
        self.pg_loss_stats[update_location] = pg_loss
        self.entropy_stats[update_location] = entropy
        self.prev_ref_log_ratio_stats[update_location] = prev_ref_log_ratio
        self.prev_new_log_ratio_stats[update_location] = prev_new_log_ratio


def micro_batch_updates(
    config: RLOOConfig,
    # integers
    epoch_idx: int,
    minibatch_idx: int,
    mini_batch_start: int,
    context_length: int,
    pad_token_id: int,
    # tensors
    batch_inds: Tensor,
    responses: Tensor,
    query_responses: Tensor,
    logprobs: Tensor,
    ref_logprobs: Tensor,  # only used for logging
    advantages: Tensor,
    padding_mask: Tensor,
    ## no need for padding_mask_plus_one; it is only used for value heads in PPO and KLQ
    # Stateful parameters that get updated
    model: nn.Module,
    accelerator: Accelerator,
    optimizer: torch.optim.Optimizer,
    stats: RLOOStats,
):
    batch_inds = batch_inds
    mini_batch_end = mini_batch_start + config.local_mini_batch_size
    mini_batch_inds = batch_inds[mini_batch_start:mini_batch_end]
    gradient_accumulation_idx = 0
    for micro_batch_start in range(
        0, config.local_mini_batch_size, config.per_device_train_batch_size
    ):
        with accelerator.accumulate(model):
            micro_batch_end = micro_batch_start + config.per_device_train_batch_size
            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]
            # Retrieve the relevant variables for this microbatch
            micro_batch_advantage = advantages[micro_batch_inds]
            micro_batch_responses = responses[micro_batch_inds]
            micro_batch_query_responses = query_responses[micro_batch_inds]
            micro_batch_prev_logprobs = logprobs[micro_batch_inds]
            micro_batch_ref_logprobs = ref_logprobs[micro_batch_inds]
            # recall that prev is policy that generated the rollouts
            # new is the policy with the current set of parameters (which we update each step)

            output = forward(model, micro_batch_query_responses, pad_token_id)
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
            new_prev_log_ratio = new_logprobs - micro_batch_prev_logprobs
            # note the INVALID_LOGPROB values will cancel out to give 0 at padding locations
            # quantities above are all token-level of shape [batch, response_length]

            # quantities below are all sequence-level of shape [batch]
            new_prev_cumulative_log_ratio = new_prev_log_ratio.sum(1)
            ratio = torch.exp(
                new_prev_cumulative_log_ratio
            )  # one could call this new_prev_sequence_prob_ratio

            # new_cumulative_logprobs = new_logprobs.sum(1)
            # micro_batch_prev_cumulative_logprobs = micro_batch_prev_logprobs.sum(1)
            # cumulative_logprobs_diff = (
            #     new_cumulative_logprobs - micro_batch_prev_cumulative_logprobs
            # )
            # ratio = torch.exp(cumulative_logprobs_diff)
            pg_losses = -micro_batch_advantage * ratio
            pg_losses2 = -micro_batch_advantage * torch.clamp(
                ratio, 1.0 - config.cliprange, 1.0 + config.cliprange
            )
            pg_loss_max = torch.max(pg_losses, pg_losses2)
            pg_loss = pg_loss_max.mean()

            # Perform the update step.
            accelerator.backward(pg_loss)
            optimizer.step()
            optimizer.zero_grad()

            # This is all just for logging.
            with torch.no_grad():
                pg_clipfrac = (pg_losses2 > pg_losses).float().mean()

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
                )

                update_location = (
                    epoch_idx,
                    minibatch_idx,
                    gradient_accumulation_idx,
                )

                stats.update(
                    update_location,
                    pg_clipfrac=pg_clipfrac,
                    pg_loss=pg_loss,
                    entropy=avg_entropy,
                    prev_ref_log_ratio=prev_ref_log_ratio.sum(1).mean(),
                    prev_new_log_ratio=prev_new_log_ratio.sum(1).mean(),
                )
        gradient_accumulation_idx += 1
    minibatch_idx += 1


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

        if maybe_answer_ids is not None:
            maybe_answer_ids = maybe_answer_ids.to(device)
            maybe_answer_ids = maybe_answer_ids.repeat(config.rloo_k, 1)

        context_length = queries.shape[1]

        with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
            # query_responses and logitss are both torch Tensors.
            # query_responses has shape [batch, query_length]
            # logitss has shape [batch, response_length, vocabulary size]

            query_responses, logitss = batch_generation(
                unwrapped_model,  # .policy, in RLOO the model is just the policy
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
        ) = rloo_rollouts_to_loss_variables(
            queries=queries,
            query_responses=query_responses,
            maybe_answer_ids=maybe_answer_ids,
            logitss=logitss,
            ref_policy=ref_policy,
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

        ## EDIT: no values needed! so have been able to comment out the next few lines.
        # sequence_lengths_plus_one = sequence_lengths + 1
        # padding_mask_plus_one = response_idxs > (sequence_lengths_plus_one.unsqueeze(1))#
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
        raw_advantage_std = advantages.std()

        # Normalize advantages
        if config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # previously had a del statement in KLQ/PPO versions of code, but not in RLOO, so think all fine
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
                advantages=advantages,
                padding_mask=padding_mask,
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
        # For RLOO, we had already summed reward over the sequence dimension, so just need a mean over batch dimension
        mean_non_score_reward = non_score_reward.mean()
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
            "policy/clipfrac_avg": s.pg_clipfrac_stats,
            #
            "loss/policy_avg": s.pg_loss_stats,
        }
        for m_name, m_tensor in metrics_gathered_and_meaned.items():
            metrics[m_name] = accelerator.gather(m_tensor).mean().item()

        # Metrics that do not need gathering
        metrics["objective/traj/num_eos_tokens"] = (
            (responses == processing_class.eos_token_id).sum().item()
        )
        # Lennie: at this point KLQ tracked some statistics of the regression targets
        # RLOO doesn't have regression targets, and I was wondering what alternative extra metric would be good to plot for RLOO.
        # I felt this standard deviation of the raw advantages might be interesting.
        metrics["loss/advantages/raw_advantage_std"] = raw_advantage_std.item()
        # Worth reviewing to see if we can think of any other interesting metrics to track.

    metrics["time/iteration/generation"] = generation_time
    metrics["time/iteration/processing"] = processing_time
    metrics["time/iteration/training"] = training_time

    return metrics


class RLOOTrainer(OnPolicyTrainer):
    _tag_names = ["trl", "rloo"]

    @property
    def model_card_info(self) -> ModelCardInfo:
        """Citation information for model card"""
        return ModelCardInfo(
            trainer_name="RLOO",
            trainer_citation="""\
        @inproceedings{ahmadian2024back,
            title        = {{Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs}},
            author       = {Arash Ahmadian and Chris Cremer and Matthias Gall{\'{e}} and Marzieh Fadaee and Julia Kreutzer and Olivier Pietquin and Ahmet {\"{U}}st{\"{u}}n and Sara Hooker},
            year         = 2024,
            booktitle    = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), {ACL} 2024, Bangkok, Thailand, August 11-16, 2024},
            publisher    = {Association for Computational Linguistics},
            pages        = {12248--12267},
            editor       = {Lun{-}Wei Ku and Andre Martins and Vivek Srikumar},
        }""",
            paper_title="Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs",
            paper_id="2402.14740",
        )

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
            self.args.num_mini_batches,
            self.args.gradient_accumulation_steps,
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
            rloo_stats=self.stats,
            # data for this batch!
            data=data,
            # Scheduled parameters
            lam=self.lambda_scheduler.get(),
        )
