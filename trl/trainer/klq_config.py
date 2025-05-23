import os
from dataclasses import dataclass

from typing import Optional

from ..trainer.on_policy_utils import OnPolicyConfig


@dataclass
class KLQConfig(OnPolicyConfig):
    r"""
    Configuration class for the [`KLQTrainer`].

    Using [`~transformers.HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        exp_name (`str`, *optional*, defaults to `os.path.basename(__file__)[:-3]`):
            Name of this experiment.
        reward_model_path (`str`, *optional*, defaults to `"EleutherAI/pythia-160m"`):
            Path to the reward model.
        num_epochs_per_batch_update (`int`, *optional*, defaults to `4`):
            Number of epochs to train.
        whiten_rewards (`bool`, *optional*, defaults to `False`):
            Whether to whiten the rewards.
        kl_coef (`float`, *optional*, defaults to `0.05`):
            KL coefficient.
        gamma (`float`, *optional*, defaults to `1.0`):
            Discount factor.
        lam (`float`, *optional*, defaults to `0.95`):
            Lambda value for GAE.
        loss_function (`str`, *optional*, defaults to `"l2_loss"`):
            Loss function to use.
        loss_kwargs (`dict`, *optional*, defaults to `None`):
            Additional arguments to pass to the loss function.
        normalize_Delta_errors (`bool`, *optional*, defaults to `True`):
            Whether to normalize the Delta errors.
            Note: in the trl PPO code these correspond to 'advantages' and the code includes a comment "this is *non-optional* and *done at the entire batch level*"
            We had this normalisation commented out until 2025-04-04.
            After a discussion on 2025-04-04, we decided to put the default at True.
            This should improve stability and avoid a failure mode of small gradients late in training.
            This should probably be set to False, but default is left as True to ensure consistent behaviour between new and historical runs.
        rescale_targets (`bool`, *optional*, defaults to `False`):
            Whether to rescale the targets by dividing by their standard deviation.
            This is (probably) preferred to z-normalizing the delta errors.
    """

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    num_epochs_per_batch_update: int = 4
    whiten_rewards: bool = False
    kl_coef: float = 0.05
    gamma: float = 1.0
    alpha: float = 1.0
    loss_function: str = "l2_loss"
    loss_kwargs: Optional[dict] = None
    normalize_Delta_errors: bool = True
    rescale_targets: bool = False

