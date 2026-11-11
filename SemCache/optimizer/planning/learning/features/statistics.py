"""Numeric value, reliability and validity triples from sampled/calibrated evidence."""

import math

from .schema import STATISTICS, kind, known, log_value


def timing_kind(op):
    """Calibration uses SCAN for bound DuckDB scan implementations."""
    return "SCAN" if kind(op) == "scan" else op.get("kind", op.get("type", "")).upper()


def input_time_calibration(op, profile, model_id=None):
    """Observed input cost, independent of selectivity and cache estimates."""
    if kind(op).startswith("semantic_"):
        bank = profile.get("request_profiles", {}).get(model_id, {}).get(op["kind"], {})
        value = bank.get("seconds_per_request")
    else:
        label = timing_kind(op)
        bank = (
            profile.get("operator_time_calibration", {})
            .get("relational", {})
            .get(label, {})
        )
        value = bank.get("seconds_per_input")
        if not known(value) or value < 0:
            bank = profile.get("relational_profiles", {}).get(label, {})
            value = bank.get("seconds_per_input")
    return {
        "seconds_per_input": value,
        "samples": bank.get("samples", bank.get("repeats")),
        "latency_cv": bank.get("latency_cv"),
    }


def add_input_time(stat, op, profile, model_id=None):
    """Populate timing and its own reliability; missing costs stay unknown."""
    calibration = input_time_calibration(op, profile, model_id)
    stat["seconds_per_input"] = calibration["seconds_per_input"]
    count, cv = calibration["samples"], calibration["latency_cv"]
    confidence = count / (count + 1) if known(count) and count > 0 else 0.0
    if known(cv) and cv > 0:
        confidence /= 1 + cv
    stat["reliability"] = {
        **stat.get("reliability", {}),
        "seconds_per_input": confidence,
    }
    return stat


def feature_reliability(values, key, value):
    """Reliability of an available estimate, independent of its zero value."""
    confidence = values.get("reliability", {}).get(key)
    if not known(confidence):
        samples = values.get("effective_samples")
        confidence = samples / (samples + 1) if known(samples) and samples > 0 else 0.0
        bounds = values.get("intervals", {}).get(key)
        if bounds and all(known(v) for v in bounds):
            confidence /= 1 + max(0, bounds[1] - bounds[0]) / max(1, abs(value))
    # A per-metric weight can come from independent evidence, such as the
    # timing bank or a completed population sample. Generic cardinality flags
    # must not erase that explicit evidence for every feature of the node.
    if not known(values.get("reliability", {}).get(key)) and (
        values.get("uncertain") or values.get("low_survival_uncertain")
    ):
        confidence = 0.0
    return min(1.0, max(0.0, confidence))


def estimated_feature(values, key, *, ratio=False, probability=True, applicable=True):
    value = values.get(key)
    if key == "selectivity" and values.get("shared_selectivity"):
        value = values.get("estimated_selectivity", value)
    valid = (
        applicable
        and known(value)
        and value >= 0
        and values.get("valid", {}).get(key, True)
    )
    if ratio and probability and valid and value > 1:
        # Weighted estimates can exceed one by floating-point roundoff.
        if not math.isclose(value, 1.0, rel_tol=1e-12):
            raise ValueError("Probability feature outside [0,1]: " + key)
        value = 1.0
    return [
        float(value) if ratio and valid else log_value(value) if valid else 0.0,
        feature_reliability(values, key, value) if valid else 0.0,
        float(valid),
    ]


def statistics(values, semantic):
    triples = [
        estimated_feature(
            values,
            key,
            ratio=key in ("selectivity", "accuracy"),
            # Relational output/input is a cardinality multiplier. A scalar
            # aggregate still emits one row when estimated input is below one.
            probability=semantic or key != "selectivity",
            applicable=semantic or key != "accuracy",
        )
        for key in STATISTICS
    ]
    return [v[i] for i in range(3) for v in triples]
