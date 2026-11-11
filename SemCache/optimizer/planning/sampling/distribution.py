"""Bounded, weighted summaries of the input samples, in serialized characters.

Length histograms describe text/request skew, not predicate satisfiability.
Distinct ratios describe the observed sample, never a population NDV estimate.
"""

import math

from planning.observations.statistics import serialize_value

LENGTH_BOUNDS = (8, 32, 128, 512, 2048, 8192, 32768)
LENGTH_BINS = len(LENGTH_BOUNDS) + 1


def summarize_rows(rows, weight):
    """One observation owns scalar metrics and weighted boundary statistics.

    Only plain immutable scalars share metrics, within this call. Compound
    values are serialized from their current contents; weights are never cached.
    """
    if not rows:
        return None, {}
    from .rows import CartesianRows

    if isinstance(rows, CartesianRows):
        left_width, left = summarize_rows(rows.left, weight)
        right_width, right = summarize_rows(rows.right, weight)
        effective = rows.effective(weight)
        fields = {}
        for summaries, repeats in ((left, len(rows.right)), (right, len(rows.left))):
            for name, summary in summaries.items():
                summary = dict(summary)
                ratio = summary.get("sample_distinct_fraction")
                if ratio is not None:
                    summary["sample_distinct_fraction"] = ratio / repeats
                summary["sample_effective_samples"] = effective
                fields[name] = summary
        width = (
            left_width + right_width
            if left_width is not None and right_width is not None
            else None
        )
        return width, fields
    weights = [weight(row) for row in rows]
    total = math.fsum(weights)
    if total <= 0 or not all(math.isfinite(w) and w > 0 for w in weights):
        raise ValueError("Field summaries require finite positive sampling weights")
    from ..kernels.observations import SummaryWeights, summarize_values

    prepared = SummaryWeights(weights, total)
    return summarize_values(rows, prepared, serialize_value)


def request_characters(op, inputs, summaries):
    """Actual prompt attributes rather than the entire materialized row width."""
    names = op.get("input_fields") or op.get("left_input_fields", []) + op.get(
        "right_input_fields", []
    )
    descriptors = {field["name"]: field for field in op.get("input_field_schema", [])}
    lengths = []
    for name in names:
        key = descriptors.get(name, {}).get("value_key", name)
        value = summaries.get(key, {}).get("average_length")
        if value is None:
            return None, None, None
        lengths.append(value)
    if op.get("right_input_fields"):
        boundary = len(op.get("left_input_fields", []))
        return sum(lengths), sum(lengths[:boundary]), sum(lengths[boundary:])
    return sum(lengths), None, None
