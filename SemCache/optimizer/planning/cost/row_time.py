"""Shared plan time: independent cardinalities times measured per-input costs."""

import math

from planning.cost.objectives import MissingEstimate, number
from planning.cost.physical_work import physical_work
from planning.cost.requests import request_predictions
from planning.state import action_key


def input_units(kind, inputs, semantic=False):
    """Semantic joins/cross products process pairs; relational joins read both sides."""
    if not all(number(value) for value in inputs):
        return None
    kind = kind.upper()
    if len(inputs) > 1 and not semantic and kind != "CROSS_PRODUCT":
        return sum(inputs)
    return math.prod(inputs)


def plan_time(
    *,
    nodes,
    actions,
    operators,
    semantic,
    logical,
    profile,
    statistics,
    audit=None,
    physical_context=None,
):
    calibration = profile.get("operator_time_calibration", {})
    nodes = tuple(nodes)
    work = (
        physical_context.work(nodes)
        if physical_context is not None
        else physical_work(nodes, operators, logical)
    )
    chosen = {action["operator_id"]: action for action in actions}
    sizes, total = {}, 0.0
    if audit is not None:
        audit.update(estimator="calibrated_input_rows_v1", operators={})
    for node in nodes:
        overlay = work.get(node.node_id)
        oid = node.operator_id
        if operators.get(oid, {}).get("sampling_cardinality_only") or str(
            node.node_id
        ) in profile.get("cardinality_only_sources", {}):
            raise MissingEstimate(
                "Full plan time is unknown: correlated terminal stages were not materialized"
            )
        prior = (
            {}
            if oid in semantic
            else profile.get("node_statistics", {}).get(str(node.node_id), {})
        )
        info = {**prior, **statistics(node)}
        if oid is None:
            kind, model = "SCAN", None
            rows = info.get("output_rows", logical[node.node_id].get("estimated_rows"))
            units = rows
        else:
            op = operators[oid]
            kind = op.get("kind", op.get("type", "")).upper()
            inputs = [sizes[child] for child in node.children]
            units = (
                inputs[op["sampling_passthrough_input"]]
                if "sampling_passthrough_input" in op
                else input_units(kind, inputs, oid in semantic)
            )
            incoming = (
                math.prod(inputs)
                if all(number(n) for n in inputs)
                else info.get("input_rows")
            )
            if units is None and len(inputs) == 1:
                units = info.get("input_rows")
            model = chosen[oid].get("model_id") if oid in semantic else None
            rows = info.get("output_rows")
            if op.get("scalar_aggregate"):
                rows = 1.0
            elif (
                "JOIN" in kind
                and op.get("join_type") in ("LEFT", "MARK")
                and len(inputs) == 2
                and number(inputs[0])
                and (inputs[0] == 0 or inputs[1] == 0)
            ):
                rows = inputs[0]
            elif incoming == 0:
                rows = 0.0
            elif rows is None and number(incoming) and number(info.get("selectivity")):
                rows = incoming * info["selectivity"]
            elif rows is None and (
                "PROJECTION" in kind
                or kind in ("CROSS_PRODUCT", "ORDER_BY", "CTE_SCAN")
            ):
                rows = incoming
        cost_kind = overlay["cost_kind"] if overlay else kind
        if overlay:
            units = (
                input_units(
                    cost_kind, [sizes[child] for child in overlay["input_nodes"]]
                )
                if overlay["input_nodes"]
                else 0.0
            )
        if not number(units):
            raise MissingEstimate(f"Missing input cardinality for {kind} time estimate")
        bank = (
            calibration.get("semantic", {}).get(model, {})
            if model is not None
            else calibration.get("relational", {})
        )
        rate = bank.get(cost_kind, {}).get("seconds_per_input")
        join_rate_fallback = False
        if overlay and cost_kind == "COMPARISON_JOIN" and not number(rate):
            rate = bank.get("FILTER", {}).get("seconds_per_input")
            join_rate_fallback = number(rate)
        if units == 0:
            seconds = 0.0
        elif not number(rate):
            raise MissingEstimate(
                f"Missing per-input time calibration for {model + '/' if model else ''}{cost_kind}"
            )
        else:
            seconds = units * rate
        sizes[node.node_id] = rows
        total += seconds
        if audit is not None:
            audit["operators"][str(node.node_id)] = {
                "kind": kind,
                "logical_kind": kind,
                "cost_kind": cost_kind,
                "fusion_owner": overlay["fusion_owner"] if overlay else None,
                "input_units_basis": overlay["input_units_basis"]
                if overlay
                else "logical operator inputs",
                "join_rate_fallback": join_rate_fallback,
                "model_id": model,
                "input_units": units,
                "seconds_per_input": rate,
                "seconds": seconds,
            }
    if not math.isfinite(total):
        raise ValueError("Nonfinite calibrated plan time")
    return total


def estimated_plan_time(problem, state, *, audit=None):
    actions = {action["operator_id"]: action for action in state.actions}
    return plan_time(
        nodes=state.partial_plans.nodes,
        actions=state.actions,
        operators=problem.all_ops,
        semantic=problem.ops,
        logical=problem.logical,
        profile=problem.profile,
        statistics=lambda node: state.statistics.get(
            action_key(actions[node.operator_id])
            if node.operator_id is not None
            else f"scan:{node.node_id}",
            {},
        ),
        audit=audit,
    )


def time_predictions(states, estimate, *, request_fallback=None):
    """A scalar seconds objective; no accuracy probe is required."""
    states = tuple(states)
    prediction = request_predictions(states, estimate)
    prediction["ranking_basis"] = "calibrated_input_time"
    if request_fallback is not None and any(
        value is None for value in prediction["times"]
    ):
        fallback = request_predictions(states, request_fallback)
        if all(value is not None for value in fallback["times"]):
            prediction["scores"] = fallback["scores"]
            prediction["ranking_basis"] = "semantic_request_count_fallback"
            prediction["time_reasons"] = [
                reason + "; ranking by estimated semantic request count"
                if reason is not None
                else None
                for reason in prediction["time_reasons"]
            ]
    return prediction
