"""Features for unfinished operators, independent of any future chosen implementation."""

from .schema import KINDS, MODEL_SLOTS, field_id, kind, known
from .statistics import add_input_time, estimated_feature


def remaining_statistics(op, node, profile):
    """Only prepared operator evidence, never future chosen-plan statistics."""
    nid = str(node["node_id"])
    evidence = profile.get("remaining_selectivity", {}).get(nid)
    if evidence is None:
        semantic = kind(op or node).startswith("semantic_")
        evidence = (
            profile.get("operator_statistics", {}).get(str(op["operator_id"]), {})
            if semantic and op
            else profile.get("node_statistics", {}).get(nid, {})
        )
    names = (
        "selectivity",
        "estimated_selectivity",
        "shared_selectivity",
        "effective_samples",
        "valid",
        "reliability",
        "intervals",
        "uncertain",
        "low_survival_uncertain",
    )
    result = {k: evidence[k] for k in names if k in evidence}
    if not known(result.get("selectivity")):
        if "projection" in kind(op or node):
            result["selectivity"] = 1.0
        elif known(evidence.get("duckdb_selectivity")):
            result["selectivity"] = evidence["duckdb_selectivity"]
    return add_input_time(result, op or node, profile)


def remaining_models(op, profile):
    """Keep model identity for every legal unfinished semantic implementation."""
    if op is None or not kind(op).startswith("semantic_"):
        return {}
    allowed = profile.get("constraints", {}).get("allowed_models", op["models"])
    result, slots = {}, {}
    for model in sorted(set(op["models"]) & set(allowed)):
        meta = profile.get("models", {}).get(model, {})
        if "supported_kinds" in meta and op["kind"] not in meta["supported_kinds"]:
            continue
        slot = field_id(("model", model)) % MODEL_SLOTS
        if slot in slots:
            raise ValueError(
                "Unfinished model timing identity collision: "
                + model
                + ", "
                + slots[slot]
            )
        result[model], slots[slot] = slot, model
    return result


def remaining_model_times(op, profile):
    """Each model keeps its observed mean, reliability and validity; never average models."""
    values = [0.0] * (MODEL_SLOTS * 3)
    for model, slot in remaining_models(op, profile).items():
        stat = add_input_time({}, op, profile, model)
        values[3 * slot : 3 * slot + 3] = estimated_feature(stat, "seconds_per_input")
    return values


def remaining_numeric(op, node, profile):
    label = kind(op or node)
    stat = remaining_statistics(op, node, profile)
    return (
        [float(label == name) for name in KINDS]
        + estimated_feature(
            stat, "selectivity", ratio=True, probability=label.startswith("semantic_")
        )
        + estimated_feature(stat, "seconds_per_input")
        + remaining_model_times(op, profile)
    )
