"""Small operator-level cache predictor, independent of value-model weights."""

from pathlib import Path

import torch
from torch import nn

from planning.execution.cache_features import (
    FEATURE_SCHEMA,
    INPUT_SIZES,
    PREFIX_FEATURES,
    feature_values,
    reuse_features,
)
from planning.execution.cache_fit import fit_cache_predictor

__all__ = ["CacheTokenMLP", "fit_cache_predictor", "reuse_features"]
PROTOCOL = "cache-operator-mlp-v2"
FEATURES = PREFIX_FEATURES


class CacheTokenMLP(nn.Module):
    """Predict an operator's mean cached-prefix length, in token units.

    Operator mode uses the same frozen node numeric features as the GAT,
    its history representation and upstream-source summaries. Actual HTTP
    usage and post-execution measurements never enter the input vector.
    """

    def __init__(
        self,
        *,
        feature_mode="operator",
        hidden_sizes=(64, 32),
        target_transform="log1p",
    ):
        super().__init__()
        if feature_mode not in INPUT_SIZES or target_transform not in {
            "linear",
            "log1p",
        }:
            raise ValueError("Invalid cache predictor configuration")
        if not hidden_sizes or any(type(v) is not int or v <= 0 for v in hidden_sizes):
            raise ValueError("Cache predictor needs positive hidden layer sizes")
        self.feature_mode, self.target_transform = feature_mode, target_transform
        self.hidden_sizes = tuple(hidden_sizes)
        size = INPUT_SIZES[feature_mode]
        layers = []
        for width in hidden_sizes:
            layers.extend([nn.Linear(size, width), nn.ReLU()])
            size = width
        layers.append(nn.Linear(size, 1))
        self.network = nn.Sequential(*layers)
        size = INPUT_SIZES[feature_mode]
        self.register_buffer("input_mean", torch.zeros(size))
        self.register_buffer("input_scale", torch.ones(size))
        self.register_buffer("target_mean", torch.tensor(0.0))
        self.register_buffer("target_scale", torch.tensor(1.0))
        self.metadata = {}

    def forward(self, values):
        inputs = torch.log1p(values) if self.feature_mode == "prefix" else values
        return self.network((inputs - self.input_mean) / self.input_scale).squeeze(-1)

    def tokens(self, prediction):
        result = prediction * self.target_scale + self.target_mean
        return (
            torch.expm1(result.clamp(0, 30))
            if self.target_transform == "log1p"
            else result.clamp_min(0)
        )

    @torch.no_grad()
    def predict(self, features):
        if not features:
            return []
        self.eval()
        values = torch.tensor(
            [feature_values(row, self.feature_mode) for row in features],
            dtype=torch.float32,
            device=next(self.parameters()).device,
        )
        return self.tokens(self(values)).tolist()

    def save(self, path):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "protocol": PROTOCOL,
                "feature_schema": FEATURE_SCHEMA,
                "config": {
                    "feature_mode": self.feature_mode,
                    "hidden_sizes": self.hidden_sizes,
                    "target_transform": self.target_transform,
                },
                "weights": self.state_dict(),
                "metadata": self.metadata,
            },
            temporary,
        )
        temporary.replace(path)

    @classmethod
    def load(cls, path, device="cpu"):
        data = torch.load(path, map_location=device, weights_only=True)
        if (
            data.get("protocol") != PROTOCOL
            or data.get("feature_schema") != FEATURE_SCHEMA
        ):
            raise ValueError("Incompatible cache-operator MLP checkpoint")
        model = cls(**data["config"])
        model.load_state_dict(data["weights"])
        model.metadata = data["metadata"]
        return model.to(device).eval()
