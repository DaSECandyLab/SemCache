"""Multiply independent operator selectivities at matching character depths."""

from .prefix_rates import local_prefix_selectivity, prefix_rate


def estimate(coordinator, problem, node, op, observations, info):
    return propagate_depth_statistics(
        info, [observations[child].statistics for child in node.children]
    )


def propagate_depth_statistics(info, children):
    """Refresh plan-local products from immutable independent operator facts.

    Shared by sampled and frozen-profile Python planning; no sampler or native
    execution is needed. The return value is the current input's depth rates.
    """
    incoming = {
        field: dict(depths)
        for child in children
        for field, depths in child.get("estimated_output_survival", {}).items()
    }
    local = local_prefix_selectivity(info)
    # A generated value begins a new population at its projection. It can also
    # first appear as a prepared expression on an otherwise unchanged input.
    for field, depths in local.items():
        if field.startswith("generated:"):
            incoming[field] = info.get("generated_input_survival", {}).get(
                field, incoming.get(field, dict.fromkeys(depths, 1.0))
            )
    info["depth_survival_statistics"] = {
        field: {
            depth: {
                "rate": value,
                "reliable": False,
                "effective_samples": 0,
                "interval": [0.0, 1.0],
                "interval_basis": "product of independent preceding depth selectivities",
            }
            for depth, value in depths.items()
        }
        for field, depths in incoming.items()
    }
    outgoing = {}
    for field, depths in incoming.items():
        outgoing[field] = {}
        for depth, value in depths.items():
            factor = prefix_rate(info, field, depth)
            outgoing[field][depth] = (
                0.0
                if value == 0 or factor == 0
                else value * factor
                if value is not None and factor is not None
                else None
            )
    info["estimated_output_survival"] = outgoing
    info["input_survival_precision"] = False
    info["survival_estimation_basis"] = (
        "product of independent preceding depth selectivities"
    )
    return incoming


def restore_cache_fields(problem, action, info, incoming):
    """Bind a frozen independent field population to this layout and plan path."""
    op = problem.ops[action["operator_id"]]
    evidence = info.get("input_field_summaries", {})
    layout = action["layout"]
    first = layout.get("field_order", layout.get("left_field_order", []))
    if not first or not evidence:
        return
    descriptors = {field["name"]: field for field in op.get("input_field_schema", [])}

    def read(name):
        descriptor = descriptors.get(name, {"name": name})
        key = descriptor.get("value_key", descriptor["name"])
        item = evidence.get(key)
        if item is None:
            return None
        rates = incoming.get(item["survival_field"], {"0": 1.0})
        alpha = {
            "total": rates.get("0"),
            "depths": {depth: value for depth, value in rates.items() if depth != "0"},
        }
        return item, alpha

    left = read(first[0])
    if left is None:
        return
    item, info["alpha"] = left
    info["summary"] = item["summary"]
    if "origin" in item:
        info["cache_field_origin"] = item["origin"]
    else:
        info.pop("cache_field_origin", None)
    if "right_field_order" in layout:
        right = read(layout["right_field_order"][0])
        if right is None:
            return
        item, info["right_alpha"] = right
        info["right_summary"] = item["summary"]
        if info.get("left_data_characters") is not None:
            info["left_total_length"] = (
                info["left_data_characters"] * info["summary"]["count"]
            )
    else:
        info.pop("right_summary", None)
        info.pop("right_alpha", None)
    from ..observations.statistics import prefix, prefix_length

    info["pre"] = prefix(
        op,
        action,
        problem.profile.get("system_prefix", "semantic-global-v1"),
        problem.profile.get("operator_specifications", {}).get(op["kind"], op["kind"]),
    )
    info["pre_length"] = prefix_length(info["pre"])
    info["basis"] = (
        "independent base-field summary with preceding depth selectivity product"
    )
