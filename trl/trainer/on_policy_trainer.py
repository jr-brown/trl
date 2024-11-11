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
from abc import ABC, abstractmethod

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
    forward,
    retokenize,
    prepare_deepspeed,
    print_rich_table,
    truncate_response,
)
from .ppo_config import PPOConfig
from .utils import generate_model_card
from .on_policy_utils import get_just_reward, forward_rollout
from ..trainer.utils import OnPolicyConfig

if is_wandb_available():
    import wandb


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

class OnPolicyStats(ABC):
    def __init__(self, stats_shape: Tuple[int, int, int], device: torch.device):
        self.stats_shape = stats_shape
        self.device = device

    @abstractmethod
    def update(self, **kwargs):
        pass


class OnPolicyTrainer(ABC, Trainer):
    _tag_names = ["trl", "ppo"]

    def __init__(
        self,
        config: OnPolicyConfig,
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
            config.total_episodes is None
        ):  # allow the users to define episodes in terms of epochs.
            config.total_episodes = int(config.num_train_epochs * self.train_dataset_len)
        accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps
        )
        self.accelerator = accelerator

        # The number of processes we're using
        config.world_size = accelerator.num_processes
        # per_device_train_batch_size
        # gradient_accumulation_steps
        # num_mini_batches
        config.local_batch_size = (
            config.per_device_train_batch_size
            * config.gradient_accumulation_steps
            * config.num_mini_batches
        )
        # Total batch size across all processes
        config.micro_batch_size = int(config.per_device_train_batch_size * config.world_size)
        config.batch_size = int(config.local_batch_size * config.world_size)
        config.mini_batch_size = exact_div(
            config.batch_size,
            config.num_mini_batches,
            "`batch_size` must be a multiple of `num_mini_batches`",
        )
        config.local_mini_batch_size = exact_div(
            config.local_batch_size,
            config.num_mini_batches,
            "`local_batch_size` must be a multiple of `num_mini_batches`",
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
        time_int = broadcast(
            time_tensor, 0
        ).item()  # avoid different timestamps across processes
        config.run_name = f"{config.exp_name}__{config.seed}__{time_int}"
        self.local_seed = config.seed + accelerator.process_index * 100003  # Prime
        if config.num_sample_generations > 0:
            self.sample_generations_freq = max(
                1, config.num_total_batches // config.num_sample_generations
            )
        self.local_dataloader_batch_size = config.local_batch_size

        #########base_model_uses_position_ids
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
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(
            config.report_to
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
            PrinterCallback if config.disable_tqdm else DEFAULT_PROGRESS_CALLBACK
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
        self.model, self.optimizer, self.dataloader = accelerator.prepare(
            self.model, self.optimizer, self.dataloader
        )
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
                self.reward_model,
                config.per_device_train_batch_size,
                config.fp16,
                config.bf16,
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

    @abstractmethod
    def _initialise_stats(self) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def batch_update(
        self,
        config: OnPolicyConfig,
        generation_config: GenerationConfig,
        processing_class: ProcessingClass,
        reward_model_processing_class: ProcessingClass,
        device: torch.device,
        accelerator: Accelerator,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        ref_policy: nn.Module,
        reward_model: nn.Module,
        stats: OnPolicyStats,
        data: Dict[str, torch.Tensor],
    ) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """Returns the updated model and the metrics for the batch."""
        pass

    def train(self):
        config = self.args
        accelerator = self.accelerator
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
        # num_ppo_epochs is the number of epochs for which we train on each PPO dataset for each increment of PPO
        # num_mini_batches 
        # gradient_accumulation_steps
        stats_shape = (
            config.num_ppo_epochs,
            config.num_mini_batches,
            config.gradient_accumulation_steps,
        )

        # Lightweight wrapper for a collection of tensors which track statistics over training
        stats = self._initialise_stats(stats_shape, device)

        # Set model to training mode (relevant for e.g. batch norm and dropout)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = config.num_total_batches * config.num_mini_batches
        self.state.num_train_epochs = config.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if config.logging_steps is not None:
            if config.logging_steps < 1:
                self.state.logging_steps = math.ceil(
                    self.state.max_steps * config.logging_steps
                )
            else:
                self.state.logging_steps = config.logging_steps
        if config.eval_steps is not None:
            if config.eval_steps < 1:
                self.state.eval_steps = math.ceil(
                    self.state.max_steps * config.eval_steps
                )
            else:
                self.state.eval_steps = config.eval_steps
        if config.save_steps is not None:
            if config.save_steps < 1:
                self.state.save_steps = math.ceil(
                    self.state.max_steps * config.save_steps
                )
            else:
                self.state.save_steps = config.save_steps
        self.control = self.callback_handler.on_train_begin(
            config, self.state, self.control
        )

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model
            self.model_wrapped = self.model

        # The actual training loop
        for update in range(1, config.num_total_batches + 1):
            self.state.episode += 1 * config.batch_size
            data = next(iter_dataloader)
            model, metrics = self.batch_update(
                model=model,
                ref_policy=ref_policy,
                reward_model=reward_model,
                data=data,
                accelerator=accelerator,
                processing_class=processing_class,
                reward_model_processing_class=reward_model_processing_class,
                config=config,
                generation_config=generation_config,
                device=device,
                optimizer=self.optimizer,
                stats=stats,
            )
            eps = int(self.state.episode / (time.time() - start_time))
            metrics["eps"] = eps
            metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
            metrics["episode"] = self.state.episode
            self.state.epoch = (
                self.state.episode / self.train_dataset_len
            )  # used by self.log
            self.state.global_step += 1
            self.log(metrics)
            self.lr_scheduler.step()
            self.control = self.callback_handler.on_step_end(
                config, self.state, self.control
            )
            if self.control.should_save:
                self._save_checkpoint(model, trial=None, metrics=metrics)
                self.control = self.callback_handler.on_save(
                    config, self.state, self.control
                )
            torch.cuda.empty_cache()
            gc.collect()

            if (
                config.num_sample_generations > 0
                and (update - 1) % self.sample_generations_freq == 0
            ):
                self.generate_completions(sampling=True)
                torch.cuda.empty_cache()

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(
            config, self.state, self.control
        )
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(
                config, self.state, self.control
            )

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
                        config.stop_token_id is not None
                    ):  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            config.stop_token_id, processing_class.pad_token_id, response
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
