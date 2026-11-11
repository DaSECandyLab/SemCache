"""Online priority Beam and training-only pure value/exploration budgets."""

import math
from dataclasses import dataclass


@dataclass
class SearchConfig:
    beam_width: int = 90
    max_expansions: int = 10000
    max_seconds: float = 60.0
    training: bool = False
    exploration_fraction: float = 0.0
    seed: int = 0
    plan_generation: str = "heuristic_requests"
    fixed_model_id: str | None = "gemma4-e2b"
    workers: int = 1
    operator_priority_mode: str | None = None
    preference: str = "fastest"
    layer_pruning: str = "pareto"
    max_structure_groups: int = 15
    max_implementations_per_group: int = 6

    @property
    def uses_join_filter_cost_pruning(self):
        return (
            not self.training
            and self.plan_generation == "heuristic_value"
            and self.layer_pruning == "pareto"
        )

    @property
    def request_guided(self):
        return self.plan_generation == "heuristic_requests"

    @property
    def priority_mode(self):
        return self.operator_priority_mode or self.plan_generation

    def layer_ranking(self, minimum=None):
        if self.training:
            return "pareto"
        if self.request_guided:
            return "heuristic_time"
        return self.layer_pruning

    @classmethod
    def configured(cls, settings, plan_generation=None, **overrides):
        return cls.inference(
            settings,
            **{
                "plan_generation": plan_generation
                or settings["inference"].get("plan_generation", "heuristic_requests"),
                **overrides,
            },
        )

    @classmethod
    def inference(cls, settings, **overrides):
        return cls(
            **{
                "beam_width": settings["search"]["beam_width"],
                "workers": settings["search"].get("workers", 1),
                "fixed_model_id": settings["search"].get(
                    "fixed_model_id", "gemma4-e2b"
                ),
                **settings["search"]["inference"],
                "training": False,
                "preference": settings["inference"].get("preference", "fastest"),
                **overrides,
            }
        )

    def __post_init__(self):
        if self.preference not in ("fastest", "accurate"):
            raise ValueError("preference must be fastest or accurate")
        if self.layer_pruning not in ("pareto", "accuracy_time", "grouped"):
            raise ValueError("layer_pruning must be pareto, accuracy_time or grouped")
        if self.layer_pruning == "grouped" and (
            self.training or self.plan_generation != "heuristic_value"
        ):
            raise ValueError("Grouped pruning requires value-guided inference")
        if self.plan_generation not in (
            "heuristic_requests",
            "heuristic_value",
            "value",
        ):
            raise ValueError("Unknown plan generation policy")
        if self.plan_generation == "value" and not self.training:
            raise ValueError("Pure value guidance is a training policy")
        if self.operator_priority_mode is not None and (
            self.plan_generation != "heuristic_value"
            or self.operator_priority_mode
            not in ("heuristic_requests", "heuristic_value")
        ):
            raise ValueError(
                "Operator-priority overrides require heuristic_value guidance"
            )
        if (
            not math.isfinite(self.exploration_fraction)
            or not 0 <= self.exploration_fraction < 1
        ):
            raise ValueError("exploration_fraction must be in [0,1)")
        if not self.training and self.exploration_fraction:
            raise ValueError("Random exploration is reserved for training")
        if self.fixed_model_id is not None and (
            not isinstance(self.fixed_model_id, str) or not self.fixed_model_id
        ):
            raise ValueError("fixed_model_id must be a nonempty model identifier")
        if any(
            type(v) is not int or v < 1
            for v in (
                self.beam_width,
                self.max_expansions,
                self.workers,
                self.max_structure_groups,
                self.max_implementations_per_group,
            )
        ):
            raise ValueError(
                "Beam width, expansion budget and group quotas must be positive integers"
            )
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("Time budget must be finite and positive")
