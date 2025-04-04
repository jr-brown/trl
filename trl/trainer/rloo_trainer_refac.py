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
from trl.trainer.on_policy_trainer import OnPolicyTrainer, ProcessingClass

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
        scoring_function: Callable,
    ) -> Dict[str, float]:
        return klq_batch_update(
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
