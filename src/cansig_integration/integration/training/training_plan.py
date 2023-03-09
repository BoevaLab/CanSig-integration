from typing import Literal, Optional  # pytype: disable=not-supported-yet

import torch  # pytype: disable=import-error
from scvi.module.base import LossOutput  # pytype: disable=import-error
from scvi.train import TrainingPlan  # pytype: disable=import-error
from scvi.train._metrics import ElboMetric  # pytype: disable=import-error
from scvi.train._trainingplans import TorchOptimizerCreator

from cansig_integration.integration.base.module import \
    CanSigBaseModule  # pytype: disable=import-error


def _linear_annealing(
        epoch: int,
        step: int,
        beta: float,
        n_epochs_kl_warmup: Optional[int],
        n_steps_kl_warmup: Optional[int],
        min_weight: Optional[float] = None,
) -> float:
    epoch_criterion = n_epochs_kl_warmup is not None
    step_criterion = n_steps_kl_warmup is not None
    if epoch_criterion:
        kl_weight = min(beta, epoch / n_epochs_kl_warmup)
    elif step_criterion:
        kl_weight = min(beta, step / n_steps_kl_warmup)
    else:
        kl_weight = 1.0
    if min_weight is not None:
        kl_weight = max(kl_weight, min_weight)
    return kl_weight


def _cycle_annealing(
        epochs: int,
        step: int,
        beta: float,
        n_epochs_kl_warmup: Optional[int],
        n_steps_kl_warmup: Optional[int],
        n_cycle=4,
        ratio=0.5,
):
    epoch_criterion = n_epochs_kl_warmup is not None
    step_criterion = n_steps_kl_warmup is not None

    if epoch_criterion:
        period = n_epochs_kl_warmup / n_cycle
        stage = int(period * ratio)
        period = int(period)
        if epochs >= n_epochs_kl_warmup or epochs % period > stage:
            return beta
        return beta / stage * (epochs % period)

    if step_criterion:
        period = n_steps_kl_warmup / n_cycle
        stage = int(period * ratio)
        period = int(period)
        if step >= n_steps_kl_warmup or step % period > stage:
            return beta
        return beta / stage * (step % period)

    return beta


class CanSigTrainingPlan(TrainingPlan):
    def __init__(
            self,
            module: CanSigBaseModule,
            *,
            optimizer: Literal["Adam", "AdamW", "Custom"] = "Adam",
            optimizer_creator: Optional[TorchOptimizerCreator] = None,
            beta: float = 1.0,
            annealing: Literal["linear", "cyclical"] = "linear",
            lr: float = 1e-3,
            weight_decay: float = 1e-6,
            eps: float = 0.01,
            n_steps_kl_warmup: Optional[int] = None,
            n_epochs_kl_warmup: Optional[int] = 400,
            reduce_lr_on_plateau: bool = False,
            lr_factor: float = 0.6,
            lr_patience: int = 30,
            lr_threshold: float = 0.0,
            lr_scheduler_metric: Literal[
                "elbo_validation", "reconstruction_loss_validation", "kl_local_validation"
            ] = "elbo_validation",
            lr_min: Optional[float] = None,
            **loss_kwargs,
    ):
        self.annealing = annealing
        self.beta = beta
        super().__init__(
            module,
            optimizer=optimizer,
            optimizer_creator=optimizer_creator,
            lr=lr,
            weight_decay=weight_decay,
            eps=eps,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            lr_scheduler_metric=lr_scheduler_metric,
            lr_threshold=lr_threshold,
            lr_min=lr_min,
            **loss_kwargs,
        )

    @property
    def kl_weight(self):
        """Scaling factor on KL divergence during training."""
        if self.annealing == "linear":
            return _linear_annealing(
                self.current_epoch,
                self.global_step,
                self.beta,
                self.n_epochs_kl_warmup,
                self.n_steps_kl_warmup,
            )
        elif self.annealing == "cyclical":
            return _cycle_annealing(
                self.current_epoch,
                self.global_step,
                self.beta,
                self.n_epochs_kl_warmup,
                self.n_steps_kl_warmup,
            )
        else:
            raise NotImplementedError(f"{self.annealing} is not implemented.")
