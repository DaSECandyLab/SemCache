"""Propagate DuckDB relational priors and sampled semantic boundaries.

Binary input cardinality is the candidate pair population, not rows read by a
hash join. Widths consistently measure serialized UTF-8 bytes; prompt lengths
remain characters. A DuckDB cardinality prior is never an execution label.
"""

import math

from planning.cost.objectives import number
from planning.state import field_key


def row_width(fields, summaries):
    widths = [
        summaries.get(
            field_key(f), summaries.get(f.get("name") if isinstance(f, dict) else f, {})
        ).get("average_bytes")
        for f in fields
    ]
    return sum(widths) if all(number(w) for w in widths) else None


def relational_prior(node, logical):
    """Read bound logical-node statistics, without matching physical profiles."""
    result = dict(node.get("statistics", {}))
    output = node.get("estimated_rows")
    children = [logical[i].get("estimated_rows") for i in node.get("children", [])]
    if number(output):
        result.setdefault("output_rows", output)
    if children and all(number(n) for n in children):
        incoming = math.prod(children)
        result.setdefault("input_rows", incoming)
        result.setdefault("input_side_rows", children)
        if incoming and number(output):
            result.setdefault("selectivity", output / incoming)
            result["duckdb_selectivity"] = min(1.0, output / incoming)
    result.setdefault("cardinality_basis", "DuckDB bound logical cardinality estimate")
    return result


def propagate(op, prior, sizes):
    """Rebase a local selectivity on this state's inputs; respect scalar aggregate."""
    result = dict(prior)
    if not sizes or not all(number(n) for n in sizes):
        for name in (
            "input_rows",
            "output_rows",
            "input_side_rows",
            "relational_cost",
            "relational_seconds",
        ):
            result.pop(name, None)
        return result
    incoming = math.prod(sizes)
    kind = op.get("kind", op.get("type", "")).upper()
    left_join = "JOIN" in kind and op.get("join_type") == "LEFT" and len(sizes) == 2
    group_prior = result.pop("group_count_estimated", False)
    rate = None if group_prior else result.get("selectivity")
    if (
        rate is None
        and not kind.startswith("SEMANTIC")
        and prior.get("sampled_selectivity_complete")
        and number(prior.get("sampled_selectivity"))
    ):
        rate = prior["sampled_selectivity"]
        result["cardinality_basis"] = (
            "weighted sample of this relational stage; fallback for unknown cardinality"
        )
    old_input = result.get("input_rows")
    uncertain_cardinality = bool(
        prior.get("uncertain") or prior.get("survival_estimated") or group_prior
    )
    if (
        rate is None
        and not uncertain_cardinality
        and number(old_input)
        and old_input
        and number(result.get("output_rows"))
    ):
        rate = result["output_rows"] / old_input
    if "PROJECTION" in kind or kind in ("CROSS_PRODUCT", "ORDER_BY", "CTE_SCAN"):
        rate = 1.0
    if number(result.get("estimated_selectivity")) and (
        result.get("selectivity_fusion")
        or (
            result.get("survival_estimated") and result.get("survival_estimation_basis")
        )
    ):
        rate = result["estimated_selectivity"]
    if (
        rate is None
        and number(result.get("duckdb_selectivity"))
        and ("FILTER" in kind or "JOIN" in kind)
    ):
        point = result["duckdb_selectivity"]
        rate = point if point > 0 or left_join else 0.25
        result.update(
            estimated_selectivity=rate,
            survival_estimated=True,
            selectivity_fusion={"weights": {"duckdb": 1.0}, "repair": True},
            survival_estimation_basis="weak DuckDB selectivity prior; not observed",
        )
    result.update(input_rows=incoming, input_side_rows=sizes)
    if old_input != incoming:
        for name in ("output_rows", "relational_cost", "relational_seconds"):
            result.pop(name, None)
    if op.get("join_type") == "MARK" and len(sizes) == 2:
        result.update(
            output_rows=sizes[0],
            selectivity=sizes[0] / incoming if incoming else None,
            cardinality_basis="MARK join preserves each left input row",
        )
    elif "sampling_passthrough_input" in op:
        output = sizes[op["sampling_passthrough_input"]]
        result.update(
            output_rows=output,
            selectivity=output / incoming if incoming else None,
            cardinality_basis="materialized CTE consumer output",
        )
    elif op.get("scalar_aggregate"):
        result["output_rows"] = 1.0
        result["selectivity"] = 1.0 / incoming if incoming else None
    elif "LIMIT" in kind and number(op.get("limit")):
        output = min(max(0, incoming - op.get("offset", 0)), op["limit"])
        result.update(
            output_rows=output, selectivity=output / incoming if incoming else None
        )
    elif (
        "AGGREGATE" in kind
        and rate is None
        and number(op.get("estimated_rows"))
        and op["estimated_rows"] > 0
    ):
        # A logical group-count prior is not a conditional selectivity. Rebase
        # it on the current upstream population; do not reuse a stale ratio.
        output = min(incoming, op["estimated_rows"])
        result.update(
            output_rows=output,
            selectivity=output / incoming if incoming else None,
            group_count_estimated=True,
            cardinality_basis="current upstream rows capped by DuckDB logical group-count prior",
        )
    elif uncertain_cardinality and rate is None:
        # An survival estimate with no observed selectivity is evidence of an
        # unmeasured boundary, not an identity filter. Do not infer rate from
        # the stale output/input pair; downstream cardinalities must remain
        # unknown and force the caller to use a valid fallback.
        result.pop("output_rows", None)
        result.pop("relational_cost", None)
        result.pop("relational_seconds", None)
        result["selectivity"] = None
    elif incoming == 0:
        result["output_rows"] = 0.0
    elif rate is not None and number(rate):
        if not result.get("selectivity_fusion"):
            result["selectivity"] = rate
        result["output_rows"] = incoming * rate
    if left_join:
        # An empty right side still emits each left tuple with NULL right
        # fields. A nonempty right side can multiply, but never remove, them.
        output = result.get("output_rows")
        if not sizes[0] or not sizes[1]:
            output = sizes[0]
        elif number(output):
            output = max(sizes[0], output)
        if number(output):
            result.update(
                output_rows=output,
                selectivity=output / incoming if incoming else None,
            )
    result["cardinality_basis"] = result.get(
        "cardinality_basis", "upstream estimates × local selectivity"
    )
    return result
