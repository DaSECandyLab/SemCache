"""Validated value-learning hyperparameters and convergence policy."""

from dataclasses import dataclass

from .data import finite


@dataclass
class TrainingConfig:
    algorithm: str = "balsa_monte_carlo"
    epochs: int = 64
    batch_size: int = 64
    checkpoint_epochs: tuple[int, ...] = (4, 8, 16, 32, 48, 64)
    learning_rate: float = 0.0003
    accuracy_weight: float = 1.0
    censor_weight: float = 1.0
    # Estimated horizons are simulator evidence, not measured timeouts. Keep
    # their one-sided pressure separate so measured censoring remains strict.
    estimated_censor_weight: float = 0.1
    complete_state_weight: float = 0.5
    partial_state_weight: float = 0.5
    time_scale: float = 1.0
    seed: int = 0
    balance_datasets: bool = True
    patience: int = 3
    warmup_fraction: float = 0.1
    discount: float = 1.0
    estimated_timeout: float | None = 1800.0
    convergence: dict | None = None

    def __post_init__(self):
        if self.algorithm not in ("balsa_monte_carlo", "supervised"):
            raise ValueError("Unknown value-learning algorithm")
        if any(type(v) is not int or v < 1 for v in (self.epochs, self.batch_size)):
            raise ValueError("Epochs and batch_size must be positive integers")
        if (
            not isinstance(self.checkpoint_epochs, (list, tuple))
            or any(type(e) is not int or e < 1 for e in self.checkpoint_epochs)
            or list(self.checkpoint_epochs) != sorted(set(self.checkpoint_epochs))
        ):
            raise ValueError(
                "checkpoint_epochs must contain increasing positive epoch numbers"
            )
        self.checkpoint_epochs = tuple(self.checkpoint_epochs)
        if self.convergence is not None:
            from planning.learning.convergence import validate_update_policy

            if not isinstance(self.convergence, dict) or set(self.convergence) != {
                "min_epochs",
                "relative_min_delta",
            }:
                raise ValueError("Convergence needs min_epochs and relative_min_delta")
            validate_update_policy(
                {
                    "max_epochs": self.epochs,
                    "patience": self.patience,
                    **self.convergence,
                },
                "convergence",
            )
        if type(self.patience) is not int or self.patience < 0:
            raise ValueError("Patience must be nonnegative")
        if not 0 <= self.warmup_fraction < 1:
            raise ValueError("Warmup fraction must be in [0,1)")
        if not finite(self.discount, lower=0.0, upper=1.0):
            raise ValueError("Discount must be in [0,1]")
        if self.estimated_timeout is not None and not finite(
            self.estimated_timeout, lower=1e-12
        ):
            raise ValueError("Estimated execution horizon must be finite and positive")
        for name in (
            "learning_rate",
            "accuracy_weight",
            "censor_weight",
            "estimated_censor_weight",
            "complete_state_weight",
            "partial_state_weight",
            "time_scale",
        ):
            if not finite(getattr(self, name), lower=1e-12):
                raise ValueError("Invalid training hyperparameter: " + name)
        if self.complete_state_weight + self.partial_state_weight <= 0:
            raise ValueError("At least one state-group loss weight must be positive")
