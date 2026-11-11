"""Patience based on cumulative relative loss improvement, independent of weights."""

import math
from dataclasses import dataclass


@dataclass
class LossPlateau:
    patience: int
    min_epochs: int = 0
    relative_min_delta: float = 0.0
    reference_loss: float = math.inf
    reference_epoch: int = 0

    def update(self, epoch, loss):
        if not math.isfinite(loss):
            raise ValueError("Non-finite convergence loss")
        if not math.isfinite(self.reference_loss) or (
            self.reference_loss - loss
            > self.relative_min_delta * max(abs(self.reference_loss), 1e-12)
        ):
            self.reference_loss, self.reference_epoch = loss, epoch
        return self.should_stop(epoch)

    def should_stop(self, epoch):
        return bool(
            self.patience
            and epoch >= self.min_epochs
            and epoch - self.reference_epoch >= self.patience
        )

    def snapshot(self):
        return {
            "reference_loss": self.reference_loss,
            "reference_epoch": self.reference_epoch,
        }


def validate_update_policy(policy, name="training.round_update"):
    if not isinstance(policy, dict) or set(policy) != {
        "max_epochs",
        "min_epochs",
        "patience",
        "relative_min_delta",
    }:
        raise ValueError(
            f"{name} needs max_epochs, min_epochs, patience and relative_min_delta"
        )
    if any(
        type(policy[k]) is not int or policy[k] < 1
        for k in ("max_epochs", "min_epochs", "patience")
    ):
        raise ValueError(f"{name} epoch counts must be positive integers")
    if policy["min_epochs"] > policy["max_epochs"]:
        raise ValueError(f"{name} min_epochs exceeds max_epochs")
    delta = policy["relative_min_delta"]
    if (
        type(delta) not in (int, float)
        or not math.isfinite(delta)
        or not 0 <= delta < 1
    ):
        raise ValueError(f"{name}.relative_min_delta must be in [0,1)")
    return policy
