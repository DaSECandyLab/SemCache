"""Shared calibrated time and accuracy formulas for pretraining and planning.

Adapters only resolve topology and statistic keys; all objective arithmetic lives
here, including cardinality propagation, cache-aware latency and global quality.
"""

import math

from planning.cost.cache_cost import REUSE_BASIS, request_seconds
from planning.observations.accuracy import value as accuracy_value
from planning.state import action_key


class MissingEstimate(ValueError):
    """An objective is unavailable because a required estimate is unknown."""


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def estimated_predictions(states, estimate, accuracy):
    """One objective evaluation per state, preserving unknowns beside rank scores."""
    times, accuracies, reasons = [], [], []
    for state in states:
        try:
            seconds, quality = estimate(state)
            reason = None
        except MissingEstimate as error:
            seconds, quality, reason = None, accuracy(state), str(error)
        times.append(seconds)
        accuracies.append(quality)
        reasons.append(reason)
    known = [seconds for seconds in times if seconds is not None]
    mixed = bool(known) and len(known) != len(times)
    # A shared positive scale preserves Pareto ordering while unknown latency
    # receives a finite floor below every known time. Neither is a prediction.
    scale = max(1.0, max(known)) if mixed else 1.0
    floor = -2.0 if mixed else 0.0
    return {
        "scores": [
            (-seconds / scale if seconds is not None else floor, quality or 0.0)
            for seconds, quality in zip(times, accuracies, strict=True)
        ],
        "times": times,
        "accuracies": accuracies,
        "time_reasons": reasons,
    }


def action_accuracy(actions, semantic, profile):
    values = []
    for action in actions:
        oid = action["operator_id"]
        if oid not in semantic:
            continue
        estimate = accuracy_value(
            profile, action["model_id"], semantic[oid]["kind"], action["layout"]
        )
        if estimate is None:
            return None
        if not number(estimate) or estimate > 1:
            raise ValueError("Global operator accuracy must be in [0,1]")
        values.append(estimate)
    return math.prod(values)


def plan_accuracy(problem, state):
    return action_accuracy(state.actions, problem.ops, problem.profile)


def estimated_objectives(problem, state, *, audit=None):
    return objectives(
        nodes=state.partial_plans.nodes,
        actions=state.actions,
        operators=problem.all_ops,
        semantic=problem.ops,
        logical=problem.logical,
        profile=problem.profile,
        statistics=lambda node, action: state.statistics.get(action_key(action), {}),
        audit=audit,
    )


def objectives(
    *, nodes, actions, operators, semantic, logical, profile, statistics, audit=None
):
    """Seconds from explicit DB cost conversion or sampled stage calibration.

    DuckDB does not expose a calibrated latency cost; unknown coefficients cause
    rejection, rather than silently treating cardinalities as milliseconds.
    Prefix lengths use serialized characters, never tokenizer token counts.
    """
    if "operator_time_calibration" in profile:
        from planning.cost.row_time import plan_time

        by_operator = {action["operator_id"]: action for action in actions}
        seconds = plan_time(
            nodes=nodes,
            actions=actions,
            operators=operators,
            semantic=semantic,
            logical=logical,
            profile=profile,
            statistics=lambda node: statistics(
                node, by_operator.get(node.operator_id, {})
            ),
            audit=audit,
        )
        quality = action_accuracy(actions, semantic, profile)
        if quality is None:
            raise MissingEstimate(
                "Missing global model quality for estimated training target"
            )
        return seconds, quality
    if audit is not None:
        audit.update(estimator=REUSE_BASIS, semantic_nodes={})
    cardinalities = {}
    total = 0.0
    actions = {a["operator_id"]: a for a in actions}
    for node in nodes:
        if node.operator_id is None:
            info = profile.get("node_statistics", {}).get(str(node.node_id), {})
            rows = info.get("output_rows", logical[node.node_id].get("estimated_rows"))
            cardinalities[node.node_id] = rows
            continue
        action, op = actions[node.operator_id], operators[node.operator_id]
        info = statistics(node, action)
        inputs = [cardinalities[c] for c in node.children]
        rows = info.get("input_rows")
        if rows is None and all(v is not None for v in inputs):
            rows = math.prod(inputs)
        if rows is None:
            raise MissingEstimate(
                "Missing input cardinality for estimated training target"
            )
        if not number(rows):
            raise ValueError("Invalid input cardinality for estimated training target")
        cardinalities[node.node_id] = info.get("output_rows")
        if rows == 0:
            # Mixed stages preserve input identities. The per-input work prior
            # is zero here; fixed materialization overhead is measured later.
            cardinalities[node.node_id] = 1.0 if op.get("scalar_aggregate") else 0.0
            if audit is not None and node.operator_id in semantic:
                audit["semantic_nodes"][str(node.operator_id)] = {
                    "requests": 0,
                    "seconds": 0.0,
                    "basis": "no requests",
                }
            continue
        if cardinalities[node.node_id] is None:
            selectivity = info.get("selectivity")
            if selectivity is not None:
                cardinalities[node.node_id] = rows * selectivity
        if node.operator_id in semantic:
            calibration = (
                profile.get("request_profiles", {})
                .get(action["model_id"], {})
                .get(op["kind"], {})
            )
            node_audit = None
            if audit is not None:
                node_audit = audit["semantic_nodes"][str(node.operator_id)] = {
                    "requests": rows,
                    "model_id": action["model_id"],
                }
            base = request_seconds(calibration, info, audit=node_audit)
            if base is None:
                raise MissingEstimate(
                    "Missing sampled model request time for estimated training target"
                )
            seconds = rows * base
            total += seconds
            if node_audit is not None:
                node_audit["seconds"] = seconds
        else:
            seconds = info.get("relational_seconds")
            if seconds is None and info.get("relational_cost") is not None:
                factor = profile.get("cost_seconds_per_unit")
                seconds = (
                    info["relational_cost"] * factor if factor is not None else None
                )
            if seconds is None:
                profiles = profile.get("relational_profiles", {})
                kind = op.get("type", op.get("kind", "")).upper()
                rate = None
                for key in (str(node.operator_id), kind, "default"):
                    candidate = profiles.get(key, {}).get("seconds_per_input")
                    if number(candidate):
                        rate = candidate
                        break
                seconds = rows * rate if rate is not None else None
            if seconds is None:
                raise MissingEstimate(
                    "Missing calibrated relational cost for estimated training target"
                )
            if not number(seconds):
                raise ValueError(
                    "Invalid calibrated relational cost for estimated training target"
                )
            total += seconds
    quality = action_accuracy(actions.values(), semantic, profile)
    if quality is None:
        raise MissingEstimate(
            "Missing global model quality for estimated training target"
        )
    return total, quality
