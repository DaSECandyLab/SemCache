"""Selection-rate cardinality propagation for request-only Beam ranking."""

import math
from typing import cast

from planning.cost.objectives import MissingEstimate, estimated_predictions, number
from planning.state import action_key


def request_count(*, nodes, operators, semantic, logical, profile, statistics):
    """Count semantic input tuples/pairs; caches and model speed do not alter work."""
    nodes = tuple(nodes)
    by_id = {node.node_id: node for node in nodes}
    needed = set()
    pending = [node.node_id for node in nodes if node.operator_id in semantic]
    while pending:
        node_id = pending.pop()
        if node_id not in needed:
            needed.add(node_id)
            pending.extend(by_id[node_id].children)
    cardinalities = {}
    total = 0.0
    for node in nodes:
        if node.node_id not in needed:
            continue
        prior = profile.get("node_statistics", {}).get(str(node.node_id), {})
        if node.operator_id is None:
            rows = prior.get("output_rows", logical[node.node_id].get("estimated_rows"))
        else:
            op = operators[node.operator_id]
            info = {**prior, **statistics(node)}
            inputs = [cardinalities[child] for child in node.children]
            incoming = math.prod(inputs) if all(number(n) for n in inputs) else None
            if incoming is None:
                incoming = info.get("input_rows")
            if not number(incoming):
                raise MissingEstimate("Missing cardinality for request-count ranking")
            incoming = cast(float, incoming)
            if node.operator_id in semantic:
                total += incoming
            if op.get("scalar_aggregate"):
                rows = 1.0
            elif (
                "JOIN" in op.get("kind", op.get("type", "")).upper()
                and op.get("join_type") in ("LEFT", "MARK")
                and len(inputs) == 2
                and number(inputs[0])
                and (inputs[0] == 0 or inputs[1] == 0)
            ):
                rows = inputs[0]
            elif incoming == 0:
                rows = 0.0
            else:
                rate = info.get("selectivity")
                if number(rate):
                    rows = incoming * cast(float, rate)
                else:
                    rows = info.get("output_rows")
                    kind = op.get("kind", op.get("type", "")).lower()
                    if rows is None and (
                        "projection" in kind or kind == "cross_product"
                    ):
                        rows = incoming
        if rows is not None and not number(rows):
            raise ValueError("Invalid cardinality in request-count ranking")
        cardinalities[node.node_id] = rows
    if not math.isfinite(total):
        raise ValueError("Nonfinite semantic request count")
    return total


def request_predictions(states, estimate):
    prediction = estimated_predictions(
        states, lambda state: (estimate(state), None), lambda state: None
    )
    # One scalar objective: unknown accuracy is never a second ranking signal.
    prediction["scores"] = [(row[0], 0.0) for row in prediction["scores"]]
    return prediction


def estimated_request_count(problem, state):
    actions = {action["operator_id"]: action for action in state.actions}
    return request_count(
        nodes=state.partial_plans.nodes,
        operators=problem.all_ops,
        semantic=problem.ops,
        logical=problem.logical,
        profile=problem.profile,
        statistics=lambda node: state.statistics.get(
            action_key(actions[node.operator_id]), {}
        ),
    )
