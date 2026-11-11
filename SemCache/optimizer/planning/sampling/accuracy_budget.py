"""Per-query operator rounds share fresh parameter groups across all models."""

import random

from ..state import digest


def select_accuracy_groups(coordinator, op, observed):
    """Admit one nonempty round, independent of candidate model or layout."""
    oid = op["operator_id"]
    if (
        coordinator.accuracy_batches_used[oid]
        >= coordinator.max_accuracy_batches_per_operator
    ):
        return []
    sampled = coordinator.accuracy_sampled_inputs[oid]
    available = [group for group in observed if group.key not in sampled]
    random.Random(int(digest([op["kind"], op["task"]])[:16], 16)).shuffle(available)
    selected = available[: coordinator.accuracy_size]
    if selected:
        sampled.update(group.key for group in selected)
        coordinator.accuracy_batches_used[oid] += 1
    return selected


def legacy_input_key(op, unit, canonical_model):
    """Group legacy backend rows by task arguments, preserving duplicate mass."""
    from .semantic import input_keys

    keys = input_keys(op)
    right = set(op.get("right_input_fields", []))
    descriptors = {
        field["name"]: field.get("value_key", field["name"])
        for field in op.get("input_field_schema", [])
    }
    right = {descriptors.get(name, name) for name in right}
    values = (
        [unit[int(len(unit) == 2 and key in right)][0].values[key] for key in keys]
        if keys
        else [side[0].values for side in unit]
    )
    return digest(
        [
            op["kind"],
            op["task"],
            op.get("output_schema"),
            canonical_model,
            op["default_layout"],
            values,
        ]
    )
