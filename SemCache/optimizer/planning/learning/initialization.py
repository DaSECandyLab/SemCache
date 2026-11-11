"""Explicit training initialization for the remaining-work architecture change."""

import hashlib
from pathlib import Path

import torch

from planning.learning.features.schema import FEATURE_VERSION
from planning.learning.model import PlanValueModel


def initialize_remaining_attention(checkpoint):
    """Reuse a v9 backbone; require fitting before the new model can guide search."""
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        payload.get("version") != 9
        or payload.get("feature_version") != FEATURE_VERSION
        or payload.get("value_target") != "unconditioned_mean_completion_pair"
        or payload.get("time_output") != "log1p_scaled_seconds"
    ):
        raise ValueError("Attention initialization requires a feature18 v9 checkpoint")
    model = PlanValueModel(**payload["config"])
    source, target = payload["weights"], model.state_dict()
    copied, expanded = [], []
    for name, value in source.items():
        if name in ("encoder.work.weight", "encoder.work.bias"):
            renamed = name.replace("encoder.work.", "encoder.work.0.")
            if target[renamed].shape != value.shape:
                raise ValueError("Remaining operator projection shape differs")
            target[renamed].copy_(value)
            copied.append(renamed)
        elif name in ("time_head.0.weight", "accuracy_head.0.weight"):
            width = model.encoder.width
            if value.shape != (width, 3 * width + 1):
                raise ValueError("Previous completion head shape differs")
            # Initialize the attention slot from mean-work columns; leave sum columns randomly initialized.
            target[name][:, :3 * width].copy_(value[:, :3 * width])
            target[name][:, -1].copy_(value[:, -1])
            expanded.append(name)
        elif name in target and target[name].shape == value.shape:
            target[name].copy_(value)
            copied.append(name)
        else:
            raise ValueError(f"Cannot explicitly initialize weight: {name}")
    model.load_state_dict(target, strict=True)
    model.training_info = {
        "initialization": "explicit_v9_remaining_attention_backbone",
        "source_checkpoint": str(checkpoint.resolve()),
        "source_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "copied_parameters": copied,
        "expanded_parameters": expanded,
        "new_parameters": sorted(set(target) - set(copied) - set(expanded)),
        "requires_fitting": True,
        "optimizer_state_reused": False,
    }
    # Transferred weights do not supervise the attention and nonlinear projections.
    model.head_availability = {"time": False, "accuracy": False}
    model.value_available = False
    return model
