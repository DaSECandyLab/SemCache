"""Population-weighted Judge agreement updates."""

from typing import Any


def update_quality(estimate: Any, match, weight, teacher):
    if not isinstance(estimate, dict):
        estimate = {"correct": 0, "total": 0, "value": None}
    if "total" not in estimate:
        estimate.update(correct=0, total=0, value=None)
    estimate["correct"] += int(match)
    estimate["total"] += 1
    estimate["weighted_correct"] = estimate.get(
        "weighted_correct", estimate["correct"] - int(match)
    ) + weight * int(match)
    old_weight = estimate.get("weighted_total", estimate["total"] - 1)
    squares = estimate.get(
        "weighted_sum_squares",
        old_weight if old_weight == estimate["total"] - 1 else None,
    )
    estimate["weighted_total"] = old_weight + weight
    estimate["weighted_sum_squares"] = (
        squares + weight**2 if squares is not None else None
    )
    estimate["effective_samples"] = (
        estimate["weighted_total"] ** 2 / estimate["weighted_sum_squares"]
        if estimate["weighted_sum_squares"]
        else None
    )
    estimate["value"] = estimate["weighted_correct"] / estimate["weighted_total"]
    estimate["basis"] = "stratified weighted LLM judge agreement"
    estimate["judge_model"] = teacher
    return estimate


def update_layout_quality(estimate, layout, match, weight, teacher):
    from planning.observations.accuracy import layout_key

    estimate = update_quality(estimate, match, weight, teacher)
    layouts = estimate.setdefault("layouts", {})
    key = layout_key(layout)
    layouts[key] = update_quality(layouts.get(key), match, weight, teacher)
    return estimate


def record_quality(
    profile,
    op,
    model,
    layout,
    actual,
    reference,
    weight,
    teacher,
    actual_key,
    reference_key,
):
    """Merge new agreement evidence once, retaining separate block orders."""
    from planning.state import digest, task_key

    identity = digest([actual_key, reference_key, *actual, *reference])
    seen = profile.setdefault("accuracy_observations", {})
    if identity in seen:
        return
    seen[identity] = True
    for bank, key in (("accuracy", op["kind"]), ("task_accuracy", task_key(op))):
        entries = profile.setdefault(bank, {}).setdefault(model, {})
        entries[key] = update_layout_quality(
            entries.get(key), layout, actual == reference, weight, teacher
        )
