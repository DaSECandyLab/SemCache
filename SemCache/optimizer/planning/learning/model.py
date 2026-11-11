"""§4.2 GAT-PE encoder and §4.4 time / accuracy completion values."""

import math
import threading
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn

from planning.learning.features.graph import encode
from planning.learning.features.schema import FEATURE_GROUPS, FEATURE_VERSION

from .encoder import GraphEncoder, mlp
from .remaining import REMAINING_ENCODING

CHECKPOINT_VERSION = 10
_INFERENCE_SESSION_LOCK = threading.Lock()


class PlanValueModel(nn.Module):
    def __init__(
        self,
        width=32,
        heads=4,
        layers=2,
        dropout=0.0,
        feature_groups=FEATURE_GROUPS,
        use_cache_features=True,
        time_scale=1.0,
    ):
        super().__init__()
        if (
            width < 1
            or heads < 1
            or width % heads
            or layers < 1
            or not 0 <= dropout < 1
        ):
            raise ValueError("Invalid GAT-PE dimensions")
        if len(set(feature_groups)) != len(feature_groups) or set(feature_groups) - set(
            FEATURE_GROUPS
        ):
            raise ValueError("Unknown or duplicate feature groups")
        if type(use_cache_features) is not bool:
            raise ValueError("use_cache_features must be a boolean")
        if not use_cache_features:
            feature_groups = tuple(
                group for group in feature_groups if group != "cache"
            )
        self.config = {
            "width": width,
            "heads": heads,
            "layers": layers,
            "dropout": dropout,
            "feature_groups": tuple(feature_groups),
            "use_cache_features": use_cache_features,
        }
        self.encoder = GraphEncoder(
            width=width,
            heads=heads,
            layers=layers,
            dropout=dropout,
            feature_groups=feature_groups,
        )
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("Time scale must be finite and positive")
        self.time_scale = float(time_scale)
        self.config["time_scale"] = self.time_scale
        self.time_head = mlp(self.encoder.output_size, width, 1)
        self.accuracy_head = mlp(self.encoder.output_size, width, 1)
        self.head_availability = {"time": False, "accuracy": False}
        self.value_available = False
        self.training_info: dict[str, Any] | None = None
        # Operational inference memory bound, independent of checkpoint/model
        # structure. Callers may override it after loading a checkpoint.
        self.inference_batch_size = 64
        self._inference_session_owner = None
        self._inference_session_weights = None

    @contextmanager
    def inference_session(self):
        """Reuse frozen topology for one search; feature rows stay immutable.

        Sessions are exclusive to one thread, require eval mode, and discard
        every cache on exit, including when scoring fails. Learned activations
        stay local to each scoring call to preserve floating-point batch shapes.
        """
        with _INFERENCE_SESSION_LOCK:
            if self._inference_session_owner is not None:
                raise RuntimeError("Inference sessions must not overlap")
            if any(module.training for module in self.modules()):
                raise RuntimeError("Inference sessions require an eval model")
            weights = tuple(
                (id(parameter), parameter._version) for parameter in self.parameters()
            )
            self._inference_session_owner = threading.get_ident()
            self._inference_session_weights = weights
        self.encoder.memo.clear()
        try:
            with torch.no_grad(), self.encoder.inference_static_inputs():
                yield self
        finally:
            self.encoder.memo.clear()
            self._inference_session_weights = None
            self._inference_session_owner = None

    def _check_inference_session(self):
        if self._inference_session_owner is None:
            return
        if self._inference_session_owner != threading.get_ident():
            raise RuntimeError("Inference sessions belong to another thread")
        if torch.is_grad_enabled() or any(module.training for module in self.modules()):
            raise RuntimeError("Inference sessions require eval and no gradients")
        if self._inference_session_weights != tuple(
            (id(parameter), parameter._version) for parameter in self.parameters()
        ):
            raise RuntimeError("Model weights changed during an inference session")

    def train(self, mode=True):
        if self._inference_session_owner is not None:
            self._check_inference_session()
            if mode:
                raise RuntimeError("Cannot train during an inference session")
        return super().train(mode)

    def _apply(self, fn, recurse=True):
        if self._inference_session_owner is not None:
            raise RuntimeError("Cannot move model weights during inference")
        return super()._apply(fn, recurse=recurse)

    def load_state_dict(self, *args, **kwargs):
        if self._inference_session_owner is not None:
            raise RuntimeError("Cannot load model weights during inference")
        return super().load_state_dict(*args, **kwargs)

    def predictions(self, z, *, log_time=False):
        """Predict one mean completion pair; no accuracy target is a model input."""
        predicted = torch.nn.functional.softplus(self.time_head(z))
        if not log_time:
            maximum = torch.finfo(predicted.dtype).max
            limit = math.log(torch.finfo(torch.float64).max) - 1
            predicted = (
                (predicted.double().clamp(max=limit).expm1() * self.time_scale)
                .clamp(max=maximum)
                .to(z.dtype)
            )
        return torch.cat((predicted, torch.sigmoid(self.accuracy_head(z))), dim=-1)

    def forward(self, features, *, incremental=True, packed=None, log_time=False):
        self._check_inference_session()
        return self.predictions(
            self.encoder(features, incremental=incremental, packed=packed),
            log_time=log_time,
        )

    def forward_batch(
        self, features, *, incremental=False, packed=None, log_time=False
    ):
        self._check_inference_session()
        return self.predictions(
            self.encoder.forward_batch(
                features, incremental=incremental, packed=packed
            ),
            log_time=log_time,
        )

    def state_scores(self, problem, state, *, incremental=True):
        return self(encode(problem, state), incremental=incremental)

    def values(self, problem, states):
        return self.values_features([encode(problem, state) for state in states])

    def values_features(self, features):
        """Return one paired prediction per state from a single forward pass."""
        self._check_inference_session()
        if not self.value_available:
            raise ValueError(
                "Untrained value model has no fitted mean completion values"
            )
        if self.training:
            self.eval()
        result = []
        with (
            torch.no_grad(),
            self.encoder.inference_static_inputs(
                reuse=self._inference_session_owner is not None
            ),
        ):
            for start in range(0, len(features), self.inference_batch_size):
                batch = features[start : start + self.inference_batch_size]
                if any(feature["nodes"] for feature in batch):
                    values = self.forward_batch(batch)
                else:
                    values = torch.stack(
                        [self(feature, incremental=False) for feature in batch]
                    )
                result.extend(values.cpu().tolist())
        return result

    def completion_prediction(self, features):
        from .prediction import completion_prediction

        return completion_prediction(self, features)

    def forward_native_batch(self, batch, *, return_tensor=False):
        """One mean pair per state, with one encoder/head call per memory chunk."""
        self._check_inference_session()
        if self.training:
            raise RuntimeError("Native graph batches require an eval model")
        groups = self.encoder.feature_groups
        if groups not in (
            frozenset(FEATURE_GROUPS),
            frozenset(FEATURE_GROUPS) - {"cache"},
        ):
            raise ValueError(
                "Native graph batches support full or no-cache feature groups"
            )
        cache_enabled = "cache" in groups
        if getattr(batch, "cache_features_enabled", True) != cache_enabled:
            raise ValueError("Native graph batch cache features do not match the model")
        if batch.chunk_size != self.inference_batch_size:
            raise ValueError("Native graph chunks do not match inference_batch_size")
        result = []
        cpu_values = (
            torch.empty(
                (len(batch.node_indices), 2),
                dtype=self.encoder.words.weight.dtype,
                device="cpu",
            )
            if return_tensor
            else None
        )
        with (
            torch.no_grad(),
            self.encoder.inference_static_inputs(
                reuse=self._inference_session_owner is not None
            ),
        ):
            buffers = self.encoder.prepare_native_batch(batch)
            node_vectors = self.encoder.words.weight.new_zeros(
                (max(1, len(batch.templates)), self.encoder.width)
            )
            for start in range(0, len(batch.node_indices), self.inference_batch_size):
                end = min(start + self.inference_batch_size, len(batch.node_indices))
                z = self.encoder.forward_native_batch(
                    batch, buffers, start, end, node_vectors
                )
                values = self.predictions(z)
                if cpu_values is None:
                    result.extend(values.cpu().tolist())
                else:
                    cpu_values[start:end].copy_(values)
        return result if cpu_values is None else cpu_values

    def completion_native_prediction(self, batch):
        if not self.value_available:
            raise ValueError(
                "Untrained value model has no fitted mean completion values"
            )
        if self.training:
            self.eval()
        from .prediction import paired_prediction

        return paired_prediction(self.forward_native_batch(batch))

    def scores(self, problem, states):
        return self.completion_prediction([encode(problem, state) for state in states])[
            "scores"
        ]

    def save(self, path, *, training_state=None):
        payload = {
            "version": CHECKPOINT_VERSION,
            "remaining_encoding": REMAINING_ENCODING,
            "value_target": "unconditioned_mean_completion_pair",
            "time_output": "log1p_scaled_seconds",
            "feature_version": FEATURE_VERSION,
            "config": self.config,
            "weights": self.state_dict(),
            "value_available": self.value_available,
            "head_availability": self.head_availability,
            "training_info": self.training_info,
        }
        if training_state is not None:
            payload["training_state"] = training_state
        torch.save(payload, path)

    @classmethod
    def load(cls, path, *, resume_training=False, device=None):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("version") != CHECKPOINT_VERSION
            or payload.get("remaining_encoding") != REMAINING_ENCODING
            or payload.get("value_target") != "unconditioned_mean_completion_pair"
            or payload.get("time_output") != "log1p_scaled_seconds"
            or payload.get("feature_version") != FEATURE_VERSION
        ):
            raise ValueError(
                "Checkpoint encoder architecture or feature schema differs; "
                "refit with nonlinear plan-conditioned remaining-work encoding"
            )
        model = cls(**payload["config"])
        model.load_state_dict(payload["weights"])
        availability = payload.get("head_availability")
        if (
            not isinstance(availability, dict)
            or set(availability) != {"time", "accuracy"}
            or any(type(value) is not bool for value in availability.values())
            or payload.get("value_available") is not all(availability.values())
        ):
            raise ValueError("Checkpoint has inconsistent completion-head supervision")
        model.head_availability = dict(availability)
        model.value_available = all(availability.values())
        model.training_info = payload.get("training_info")
        if resume_training:
            model._training_state = payload.get("training_state")
        model.eval()
        if device is not None:
            model.to(device)
        return model
