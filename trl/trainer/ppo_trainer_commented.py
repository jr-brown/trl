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
from typing import Dict, List, Optional, Tuple, Union

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

from ..core import masked_mean, masked_whiten
from ..models.utils import unwrap_model_for_generation
from ..trainer.utils import (
    OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    retokenize,
    get_just_value,
    get_just_reward,
    prepare_deepspeed,
    print_rich_table,
    truncate_response,
)
from .ppo_config import PPOConfig
from .utils import generate_model_card

if is_wandb_available():
    import wandb


# Old
# ProcessingClass = Union[
#    PreTrainedTokenizerBase,
#    BaseImageProcessor,
#    FeatureExtractionMixin,
#    ProcessorMixin
# ]

# To actually make type checking helpful and not throw errors everywhere
ProcessingClass = PreTrainedTokenizerBase


INVALID_LOGPROB = 1.0


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


class PPOTrainer(Trainer):
    _tag_names = ["trl", "ppo"]

    def __init__(
        self,
        config: PPOConfig,
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
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        callbacks: Optional[List[TrainerCallback]] = None,
    ) -> None:

        if ref_policy is policy:
            raise ValueError(
                "`policy` and `ref_policy` cannot be the same object. If you want `ref_policy` to be the "
                "same as `policy`, you must mass a copy of it, or `None` if you use peft."
            )

        assert processing_class is not None

        self.args = config
        args = config
        self.processing_class = processing_class
        self.policy = policy

        self.policy.generation_config.eos_token_id = (
            None  # disable `pad_token_id` and `eos_token_id` because we just want to
        )
        self.policy.generation_config.pad_token_id = (
            None  # generate tokens without truncation / padding
        )

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
        if (
            args.total_episodes is None
        ):  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
        self.accelerator = accelerator

        # The number of processes we're using
        args.world_size = accelerator.num_processes
        # per_device_train_batch_size
        # gradient_accumulation_steps
        # num_mini_batches
        args.local_batch_size = (
            args.per_device_train_batch_size
            * args.gradient_accumulation_steps
            * args.num_mini_batches
        )
        # Total batch size across all processes
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size,
            args.num_mini_batches,
            "`batch_size` must be a multiple of `num_mini_batches`",
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size,
            args.num_mini_batches,
            "`local_batch_size` must be a multiple of `num_mini_batches`",
        )
        if args.whiten_rewards:
            assert (
                args.local_mini_batch_size >= 8
            ), f"Per-rank minibatch size {args.local_mini_batch_size} is insufficient for whitening"
        # `per_rank_rollout_batch_size` is our `args.local_batch_size`
        # `per_rank_minibatch_size` is our `args.local_mini_batch_size`
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(
            time_tensor, 0
        ).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(
                1, args.num_total_batches // args.num_sample_generations
            )
        self.local_dataloader_batch_size = args.local_batch_size

        #########base_model_uses_position_ids
        # setup model, optimizer, and others
        #########
        for module in [policy, ref_policy, value_model, reward_model]:
            disable_dropout_in_model(module)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = processing_class.eos_token_id
        self.model = PolicyAndValueWrapper(policy, value_model)
        self.model.config = policy.config  # needed for pushing to hub
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        ### trainer specifics
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(
            self.args.report_to
        )
        self.callbacks = (
            default_callbacks if callbacks is None else default_callbacks + callbacks
        )
        self.callback_handler = CallbackHandler(
            self.callbacks,
            self.model,
            self.processing_class,
            self.optimizer,
            self.lr_scheduler,
        )
        self.add_callback(
            PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK
        )
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb
                for cb in self.callback_handler.callbacks + [self.control]
                if isinstance(cb, ExportableState)
            ],
        )
        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = (
            getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        )
        self.is_fsdp_enabled = (
            getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        )
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)

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
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(
            self.model, self.optimizer, self.dataloader
        )
        torch.manual_seed(self.local_seed)  # reset the local seed again

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=DataCollatorWithPadding(self.processing_class),
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if self.is_deepspeed_enabled:
            self.reward_model = prepare_deepspeed(
                self.reward_model,
                args.per_device_train_batch_size,
                args.fp16,
                args.bf16,
            )
            self.ref_policy = prepare_deepspeed(
                self.ref_policy, args.per_device_train_batch_size, args.fp16, args.bf16
            )
        else:
            self.ref_policy = self.ref_policy.to(self.accelerator.device)
            self.reward_model = self.reward_model.to(self.accelerator.device)

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    def save_model(
        self, output_dir: Optional[str] = None, _internal_call: bool = False
    ):
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
        args = self.args
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
            max_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
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
            args.num_ppo_epochs,
            args.num_mini_batches,
            args.gradient_accumulation_steps,
        )
        # Define a collection of tensors which track statistics over training
        approximate_kl_stats = torch.zeros(stats_shape, device=device)
        policy_gradient_clipfrac_stats = torch.zeros(stats_shape, device=device)
        policy_gradient_loss_stats = torch.zeros(stats_shape, device=device)
        value_function_loss_stats = torch.zeros(stats_shape, device=device)
        value_function_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches * args.num_mini_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(
                    self.state.max_steps * args.logging_steps
                )
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(
                    self.state.max_steps * args.eval_steps
                )
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(
                    self.state.max_steps * args.save_steps
                )
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control
        )

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        # The actual training loop
        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            with torch.no_grad():
                queries = data["input_ids"].to(device)
                context_length = queries.shape[1]
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                sequence_lengths = []
                values = []
                with unwrap_model_for_generation(
                    model, self.accelerator
                ) as unwrapped_model:
                    # query_respones and logitss are both torch Tensors.
                    # query_responses has shape [batch, query_length]
                    # logitss has shape [batch, response_length, vocabulary size]
                    query_responses, logitss = batch_generation(
                        unwrapped_model.policy,
                        queries,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        generation_config,
                    )

                # Note: local_rollout_forward_batch_size gives how many token generation steps we chunk trajectories into.
                # Iterate through chunks of the queries, moving along by the local_rollout_forward_batch_size each time.
                for i in range(
                    0, queries.shape[0], args.local_rollout_forward_batch_size
                ):
                    # Extract the query
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    # Extract the corresponding response
                    query_response = query_responses[
                        i : i + args.local_rollout_forward_batch_size
                    ]
                    # Discard the first context_length tokens.
                    response = query_response[:, context_length:]
                    # Get the logits for all tokens at each step of the generation.
                    logits = logitss[i : i + args.local_rollout_forward_batch_size]
                    # Get the log probabilities of every token at each step of the generation
                    all_logprob = F.log_softmax(logits, dim=-1)
                    # Get only those log-probabilities for the tokens that were actually generated.
                    logprob = torch.gather(
                        all_logprob, 2, response.unsqueeze(-1)
                    ).squeeze(-1)
                    # Memory management - delete the logits and all the log probabilities
                    del logits, all_logprob
                    torch.cuda.empty_cache()

                    # This computes the log-probabilities for the base model (reference model)
                    ref_output = forward(
                        ref_policy, query_response, processing_class.pad_token_id
                    )
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.temperature + 1e-7
                    ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                    ref_logprob = torch.gather(
                        ref_all_logprob, 2, response.unsqueeze(-1)
                    ).squeeze(-1)
                    # Memory management - delete the logits and all the log probabilities
                    del ref_output, ref_logits, ref_all_logprob
                    torch.cuda.empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    if (
                        args.stop_token_id is not None
                    ):  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, processing_class.pad_token_id, response
                        )

                    # Response Processing 2. run reward model on the truncated responses

                    # FLAG - This feels inefficient for when the action-value function is a head on the model doing the generation
                    # Maybe there's a way to get around this. For now, we might just want separate models.
                    postprocessed_query_response = torch.cat(
                        (query, postprocessed_response), 1
                    )
                    sequence_length = (
                        first_true_indices(
                            postprocessed_response == processing_class.pad_token_id
                        )
                        - 1
                    )
                    unwrapped_value_model = accelerator.unwrap_model(model).value_model
                    # It looks like TRL treates rewards and values the same way.
                    # We might need our own class for action-value functions.
                    # full_value has shape [batch, query_length, 1]

                    full_value = get_just_value(
                        unwrapped_value_model,
                        query_response,
                        processing_class.pad_token_id,
                    )
                    # Extract only the value estimates for the completion
                    value = full_value[:, context_length - 1 : -1].squeeze(-1)
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
                    values.append(value)

                # Now we stack all these sub-chunks together so we can pass them through as one batch.
                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                values = torch.cat(values, 0)
                # Memory management stuff
                del (logprob, ref_logprob, full_value, value, score, unwrapped_model)
                torch.cuda.empty_cache()
                gc.collect()

                # Response Processing 3. Filter completion. Ensure that the sample contains stop_token_id
                # Completions not passing that filter will receive a lower score.
                contain_eos_token = torch.any(
                    postprocessed_responses == self.processing_class.eos_token_id,
                    dim=-1,
                )
                if self.args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty
                # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(
                    responses.shape[1], device=responses.device
                ).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(
                    ref_logprobs, padding_mask, INVALID_LOGPROB
                )
                sequence_lengths_p1 = sequence_lengths + 1
                padding_mask_p1 = response_idxs > (sequence_lengths_p1.unsqueeze(1))
                values = torch.masked_fill(values, padding_mask_p1, 0)

                # 4. compute rewards
                kl = logprobs - ref_logprobs
                non_score_reward = -args.kl_coef * kl
                # rewards has shape [batch, response_length]
                rewards = non_score_reward.clone()
                actual_start = torch.arange(rewards.size(0), device=rewards.device)
                actual_end = torch.where(
                    sequence_lengths_p1 < rewards.size(1),
                    sequence_lengths_p1,
                    sequence_lengths,
                )
                rewards[[actual_start, actual_end]] += scores

                # 5. whiten rewards
                if args.whiten_rewards:
                    rewards = masked_whiten(
                        rewards, mask=~padding_mask_p1, shift_mean=False
                    )
                    rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

                # 6. compute advantages and returns
                # Initialise the GAE at 0 for the last time step.
                lastgaelam = 0
                advantages_reversed = []
                gen_length = responses.shape[1]
                for t in reversed(range(gen_length)):
                    nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
                    # Compute the TD-error
                    delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
                    # Use the GAE backwards recursion relationship
                    lastgaelam = delta + args.gamma * args.lam * lastgaelam
                    advantages_reversed.append(lastgaelam)
                # Create the advantage estimates by reversing the GAE backward recursion
                advantages = torch.stack(advantages_reversed[::-1], axis=1)
                # Set the return estimates to be the advantage estimates
                returns = advantages + values
                # Whiten the advantages. Note that this is *non-optional* and *done at the entire batch level*
                advantages = masked_whiten(advantages, ~padding_mask)
                advantages = torch.masked_fill(advantages, padding_mask, 0)
                torch.cuda.empty_cache()

            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            # num_ppo_epochs specifies how many times to loop over the PPO dataset.
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                # Draw a random permutation
                batch_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(
                    0, args.local_batch_size, args.local_mini_batch_size
                ):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = batch_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(
                        0, args.local_mini_batch_size, args.per_device_train_batch_size
                    ):
                        # I think that micro-batches are minibatches divided between machines.
                        with accelerator.accumulate(model):
                            micro_batch_end = (
                                micro_batch_start + args.per_device_train_batch_size
                            )
                            micro_batch_inds = mini_batch_inds[
                                micro_batch_start:micro_batch_end
                            ]
                            # Retrieve the relevant variables for this microbatch
                            micro_batch_advantage = advantages[micro_batch_inds]
                            micro_batch_responses = responses[micro_batch_inds]
                            micro_batch_query_responses = query_responses[
                                micro_batch_inds
                            ]
                            micro_batch_logprobs = logprobs[micro_batch_inds]
                            micro_batch_return = returns[micro_batch_inds]
                            micro_batch_values = values[micro_batch_inds]

                            output, value_prediction_temp = forward(
                                model,
                                micro_batch_query_responses,
                                processing_class.pad_token_id,
                            )
                            logits = output.logits[:, context_length - 1 : -1]
                            logits /= args.temperature + 1e-7
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
                            value_prediction = value_prediction_temp[
                                :, context_length - 1 : -1
                            ].squeeze(-1)
                            value_prediction = torch.masked_fill(
                                value_prediction, padding_mask_p1[micro_batch_inds], 0
                            )
                            value_prediction_clipped = torch.clamp(
                                value_prediction,
                                micro_batch_values - args.cliprange_value,
                                micro_batch_values + args.cliprange_value,
                            )
                            value_function_losses_unclipped = torch.square(
                                value_prediction - micro_batch_return
                            )
                            value_function_losses_clipped = torch.square(
                                value_prediction_clipped - micro_batch_return
                            )
                            value_function_loss_max = torch.max(
                                value_function_losses_unclipped,
                                value_function_losses_clipped,
                            )
                            value_function_loss = 0.5 * masked_mean(
                                value_function_loss_max,
                                ~padding_mask_p1[micro_batch_inds],
                            )
                            value_function_clipfrac = masked_mean(
                                (
                                    value_function_losses_clipped
                                    > value_function_losses_unclipped
                                ).float(),
                                ~padding_mask_p1[micro_batch_inds],
                            )

                            # Compute the policy gradient loss term.
                            logprobs_diff = new_logprobs - micro_batch_logprobs
                            ratio = torch.exp(logprobs_diff)
                            policy_gradient_losses_unclipped = (
                                -micro_batch_advantage * ratio
                            )
                            policy_gradient_losses_clipped = (
                                -micro_batch_advantage
                                * torch.clamp(
                                    ratio, 1.0 - args.cliprange, 1.0 + args.cliprange
                                )
                            )
                            policy_gradient_losses_max = torch.max(
                                policy_gradient_losses_unclipped,
                                policy_gradient_losses_clipped,
                            )
                            policy_gradient_loss = masked_mean(
                                policy_gradient_losses_max,
                                ~padding_mask[micro_batch_inds],
                            )
                            loss = (
                                policy_gradient_loss
                                + args.vf_coef * value_function_loss
                            )

                            # Perform the update step.
                            accelerator.backward(loss)
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
                                approximate_kl = 0.5 * (logprobs_diff**2).mean()
                                approximate_kl_stats[
                                    ppo_epoch_idx,
                                    minibatch_idx,
                                    gradient_accumulation_idx,
                                ] = approximate_kl
                                policy_gradient_clipfrac_stats[
                                    ppo_epoch_idx,
                                    minibatch_idx,
                                    gradient_accumulation_idx,
                                ] = policy_gradient_clipfrac
                                policy_gradient_loss_stats[
                                    ppo_epoch_idx,
                                    minibatch_idx,
                                    gradient_accumulation_idx,
                                ] = policy_gradient_loss
                                value_function_loss_stats[
                                    ppo_epoch_idx,
                                    minibatch_idx,
                                    gradient_accumulation_idx,
                                ] = value_function_loss
                                value_function_clipfrac_stats[
                                    ppo_epoch_idx,
                                    minibatch_idx,
                                    gradient_accumulation_idx,
                                ] = value_function_clipfrac
                                entropy_stats[
                                    ppo_epoch_idx,
                                    minibatch_idx,
                                    gradient_accumulation_idx,
                                ] = entropy.mean()
                                ratio_stats[
                                    ppo_epoch_idx,
                                    minibatch_idx,
                                    gradient_accumulation_idx,
                                ] = ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, value_prediction_temp, logits, new_all_logprobs, new_logprobs, value_prediction, value_prediction_clipped,
                        value_function_losses_unclipped, value_function_losses_clipped, value_function_loss, value_function_clipfrac, logprobs_diff, ratio, policy_gradient_losses_clipped, policy_gradient_losses_unclipped, policy_gradient_losses_max,
                        policy_gradient_loss, loss, policy_gradient_clipfrac, prob_dist, entropy, approximate_kl, micro_batch_return,
                        micro_batch_advantage, micro_batch_values, micro_batch_responses, micro_batch_query_responses, micro_batch_logprobs,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()

            # At the end of training, log a bunch of statistics in the metrics dictionary.
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.sum(1).mean()
                rlhf_reward = mean_non_score_reward + scores.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather(mean_kl).mean().item()
                metrics["objective/entropy"] = (
                    self.accelerator.gather(mean_entropy).mean().item()
                )
                metrics["objective/non_score_reward"] = (
                    self.accelerator.gather(mean_non_score_reward).mean().item()
                )
                metrics["objective/rlhf_reward"] = (
                    self.accelerator.gather(rlhf_reward).mean().item()
                )
                metrics["objective/scores"] = (
                    self.accelerator.gather(scores.mean()).mean().item()
                )
                metrics["policy/approximate_kl_avg"] = (
                    self.accelerator.gather(approximate_kl_stats).mean().item()
                )
                metrics["policy/clipfrac_avg"] = (
                    self.accelerator.gather(policy_gradient_clipfrac_stats)
                    .mean()
                    .item()
                )
                metrics["loss/policy_avg"] = (
                    self.accelerator.gather(policy_gradient_loss_stats).mean().item()
                )
                metrics["loss/value_avg"] = (
                    self.accelerator.gather(value_function_loss_stats).mean().item()
                )
                metrics["val/clipfrac_avg"] = (
                    self.accelerator.gather(value_function_clipfrac_stats).mean().item()
                )
                metrics["policy/entropy_avg"] = (
                    self.accelerator.gather(entropy_stats).mean().item()
                )
                metrics["val/ratio"] = (
                    self.accelerator.gather(ratio_stats).mean().item()
                )
                metrics["val/ratio_var"] = (
                    self.accelerator.gather(ratio_stats).var().item()
                )
                metrics["val/num_eos_tokens"] = (
                    (responses == processing_class.eos_token_id).sum().item()
                )
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = (
                    self.state.episode / self.train_dataset_len
                )  # used by self.log
                self.state.global_step += 1
                self.log(metrics)

            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(
                args, self.state, self.control
            )
            if self.control.should_save:
                self._save_checkpoint(model, trial=None, metrics=metrics)
                self.control = self.callback_handler.on_save(
                    self.args, self.state, self.control
                )
            del (
                kl,
                mean_kl,
                mean_entropy,
                mean_non_score_reward,
                scores,
                metrics,
                non_score_reward,
            )
            torch.cuda.empty_cache()
            gc.collect()

            if (
                args.num_sample_generations > 0
                and (update - 1) % self.sample_generations_freq == 0
            ):
                self.generate_completions(sampling=True)
                torch.cuda.empty_cache()
            del (
                query_responses,
                responses,
                postprocessed_responses,
                logprobs,
                ref_logprobs,
                values,
                sequence_lengths,
                contain_eos_token,
                sequence_lengths_p1,
                response_idxs,
                padding_mask,
                padding_mask_p1,
                rewards,
                actual_start,
                actual_end,
                advantages,
                returns,
            )
            torch.cuda.empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control
        )
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

    def generate_completions(self, sampling: bool = False):
        args = self.args
        processing_class = self.processing_class
        reward_model_processing_class = self.reward_model_processing_class
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        table = defaultdict(list)
        with unwrap_model_for_generation(
            self.model, self.accelerator
        ) as unwrapped_model:
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
                    if (
                        args.stop_token_id is not None
                    ):  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, processing_class.pad_token_id, response
                        )
                    table["query"].extend(
                        gather_object(
                            processing_class.batch_decode(
                                query, skip_special_tokens=True
                            )
                        )
                    )
                    table["model response"].extend(
                        gather_object(
                            processing_class.batch_decode(postprocessed_response)
                        )
                    )

                    postprocessed_query_response = torch.cat(
                        (query, postprocessed_response), 1
                    )

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
                    table["score"].extend(
                        self.accelerator.gather(score).float().cpu().numpy()
                    )

                if sampling:
                    break
        df = pd.DataFrame(table)

        if self.accelerator.is_main_process:
            print_rich_table(df.iloc[0 : 0 + 5])
            if "wandb" in args.report_to:
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

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(
            self.model.config._name_or_path
        ):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
        @article{mziegler2019fine-tuning,
            title        = {{Fine-Tuning Language Models from Human Preferences}},
            author       = {Daniel M. Ziegler and Nisan Stiennon and Jeffrey Wu and Tom B. Brown and Alec Radford and Dario Amodei and Paul F. Christiano and Geoffrey Irving},
            year         = 2019,
            eprint       = {arXiv:1909.08593}
        }"""
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=(
                wandb.run.get_url()
                if is_wandb_available() and wandb.run is not None
                else None
            ),
            trainer_name="PPO",
            trainer_citation=citation,
            paper_title="Fine-Tuning Language Models from Human Preferences",
            paper_id="1909.08593",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
