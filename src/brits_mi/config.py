"""Configuration objects shared by training and simulation code."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class LossWeights:
    """Weights in the BRITS-MI training objective."""

    reconstruction: float = 1.0
    consistency: float = 0.10
    trajectory_summary: float = 0.35
    downstream: float = 0.50


@dataclass(frozen=True)
class TrainingConfig:
    """Training, validation, and stochastic-completion settings."""

    epochs: int = 80
    hidden_size: int = 48
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0
    validation_fraction: float = 0.20
    train_holdout_fraction: float = 0.15
    validation_holdout_fraction: float = 0.18
    residual_holdout_fraction: float = 0.20
    residual_minimum: float = 0.05
    variance_factor: float = 0.85
    seed: int = 1
    device: str = "auto"
    losses: LossWeights = field(default_factory=LossWeights)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

