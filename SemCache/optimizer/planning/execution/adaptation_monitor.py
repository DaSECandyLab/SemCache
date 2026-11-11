"""Cumulative runtime drift detection; missing API usage is never a zero hit."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptationSettings:
    enabled: bool = False
    beam_width: int = 12
    max_switches: int = 2
    selectivity_threshold: float = 0.3
    cache_threshold: float = 0.2
    cache_trigger_fraction: float = 1 / 3
    epsilon: float = 1e-8
    cache_predictor_checkpoint: str | None = None
    device: str = "cuda"

    def __post_init__(self):
        if type(self.beam_width) is not int or self.beam_width < 1:
            raise ValueError("Runtime beam width must be a positive integer")
        if self.device not in {"cuda", "cpu", "auto"}:
            raise ValueError("Runtime adaptation device must be cuda, cpu or auto")
        if (
            type(self.enabled) is not bool
            or type(self.max_switches) is not int
            or not 1 <= self.max_switches <= 128
        ):
            raise ValueError("Runtime switch limit must be in 1..128")
        for v in (
            self.selectivity_threshold,
            self.cache_threshold,
            self.epsilon,
        ):
            if not math.isfinite(v) or v <= 0:
                raise ValueError(
                    "Runtime tolerances/timeouts must be finite and positive"
                )
        if (
            type(self.cache_trigger_fraction) not in (int, float)
            or not math.isfinite(self.cache_trigger_fraction)
            or not 0 <= self.cache_trigger_fraction < 1
        ):
            raise ValueError("Runtime cache trigger fraction must be in [0,1)")


def drift_checks(
    statistics,
    estimated_selectivity,
    predicted_cache,
    kind,
    settings,
    *,
    generation=None,
):
    """Use all completed chunks of the current implementation generation."""
    result = {"selectivity_drift": None, "cache_drift": None, "triggers": []}
    if not statistics or not statistics.get("completed"):
        result["status"] = "no_observations"
        return result
    if generation is not None and statistics["config_generation"] != generation:
        result["status"] = "stale_generation"
        return result
    incoming, outgoing = statistics["completed"], statistics["emitted"]
    result.update(
        chunks=statistics["chunks"],
        input_rows=incoming,
        statistics_basis="cumulative actual completed inputs including the active chunk",
    )
    if estimated_selectivity is not None and kind in {
        "SEMANTIC_FILTER",
        "SEMANTIC_JOIN",
    }:
        actual = outgoing / incoming
        error = abs(actual - estimated_selectivity) / (
            estimated_selectivity + settings.epsilon
        )
        result.update(
            observed_selectivity=actual,
            estimated_selectivity=estimated_selectivity,
            selectivity_drift=error,
        )
        if error > settings.selectivity_threshold:
            result["triggers"].append("selectivity_drift")
    mean = statistics.get("mean_matched_tokens")
    result["cache_metric_basis"] = "API cache reads plus full prompt-cache hits"
    if (
        predicted_cache is not None
        and type(mean) in (int, float)
        and math.isfinite(mean)
        and mean >= 0
    ):
        error = abs(mean - predicted_cache) / (mean + settings.epsilon)
        result.update(
            cache_drift=error,
            predicted_mean_cached_tokens=predicted_cache,
            observed_mean_cached_tokens=mean,
        )
        if error > settings.cache_threshold:
            result["triggers"].append("cache_reuse_drift")
    else:
        result["cache_status"] = "missing_checkpoint_or_public_usage"
    result["status"] = "checked"
    return result


def plan_triggers(
    operators,
    predicted_accuracy,
    minimum,
    settings,
):
    """Combine operator drift with the plan-level final-result F1 estimate.

    All semantic operators in the current plan form the cache denominator.
    Unknown usage and absent/stale observations cannot count as violations.
    """
    triggers = []
    if minimum is not None and predicted_accuracy < minimum:
        triggers.append("accuracy_deficit")
    total = len(operators)
    selectivity_drifted = [
        oid for oid, o in operators.items() if "selectivity_drift" in o["triggers"]
    ]
    selectivity_fraction = len(selectivity_drifted) / total if total else 0.0
    selectivity_triggered = (
        total > 0 and selectivity_fraction > settings.cache_trigger_fraction
    )
    if selectivity_triggered:
        triggers.append("selectivity_drift")
    measured = sum(o["cache_drift"] is not None for o in operators.values())
    drifted = [
        oid for oid, o in operators.items() if "cache_reuse_drift" in o["triggers"]
    ]
    fraction = len(drifted) / total if total else 0.0
    triggered = total > 0 and fraction > settings.cache_trigger_fraction
    if triggered:
        triggers.append("cache_reuse_drift")
    return {
        "triggers": sorted(triggers),
        "cache_drift_summary": {
            "total_operators": total,
            "measured_operators": measured,
            "drifted_operators": drifted,
            "drifted_count": len(drifted),
            "drifted_fraction": fraction,
            "trigger_fraction": settings.cache_trigger_fraction,
            "comparison": ">",
            "triggered": triggered,
        },
        "selectivity_drift_summary": {
            "drifted_operators": selectivity_drifted,
            "drifted_count": len(selectivity_drifted),
            "drifted_fraction": selectivity_fraction,
            "trigger_fraction": settings.cache_trigger_fraction,
            "comparison": ">",
            "triggered": selectivity_triggered,
        },
    }
