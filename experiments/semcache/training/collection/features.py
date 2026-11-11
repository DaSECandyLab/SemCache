"""Audit inference-time estimates; never populate inputs from full plan labels."""

import copy
import math

from planning.learning.features.graph import encode
from planning.learning.training.trajectory import refresh_statistics, state_key
from planning.observations.accuracy import value as accuracy_value
from planning.state import SearchProblem, action_key


def audit_trajectory(problem, trace, encoded_trajectory):
    counts = {
        "semantic_nodes": 0,
        "existing_edges": 0,
        "previous_edges": 0,
        "unknown_estimates": 0,
    }
    for state, record in zip(trace, encoded_trajectory, strict=True):
        features = record["features"]
        for row in features["nodes"].values():
            if any(not math.isfinite(v) for v in row["numeric"]):
                raise ValueError("Nonfinite encoded node feature")
        for action in problem.semantic_actions(state):
            op = problem.ops[action["operator_id"]]
            info = state.statistics[action_key(action)]
            for name in ("input_rows", "input_width", "selectivity", "l_self"):
                value = info.get(name)
                if value is None:
                    counts["unknown_estimates"] += 1
                elif (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError("Invalid estimated feature " + name)
            for group in ("l_existing", "l_previous"):
                if group not in info:
                    counts["unknown_estimates"] += 1
                    continue
                counts[group.replace("l_", "") + "_edges"] += len(info[group])
                if any(
                    v is not None
                    and (
                        not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                    )
                    for v in info[group].values()
                ):
                    raise ValueError("Invalid cache estimate in " + group)
            value = accuracy_value(
                problem.profile, action["model_id"], op["kind"], action["layout"]
            )
            latency = (
                problem.profile.get("request_profiles", {})
                .get(action["model_id"], {})
                .get(op["kind"], {})
                .get("seconds_per_request")
            )
            if value is None or latency is None:
                raise ValueError(
                    "Missing quality or request-time calibration for "
                    f"model={action['model_id']!r}, kind={op['kind']}, "
                    f"operator_id={action['operator_id']} "
                    f"(accuracy={value!r}, seconds_per_request={latency!r})"
                )
            counts["semantic_nodes"] += 1
    return {
        "status": "passed",
        "statistics_basis": "same estimates and masks as online inference",
        **counts,
    }


def normalize_explain(query):
    """Remove EXPLAIN's extra root and restore execution-time logical node IDs."""
    query = copy.deepcopy(query)
    nodes = [n for n in query["operators"] if n["type"] != "EXPLAIN"]
    ids = {
        n["node_id"]: i for i, n in enumerate(sorted(nodes, key=lambda n: n["node_id"]))
    }
    for node in nodes:
        node["node_id"] = ids[node["node_id"]]
        node["children"] = [ids[c] for c in node["children"]]
        if "pair_targets" in node:
            node["pair_targets"] = [ids[n] for n in node["pair_targets"]]
    query["operators"] = nodes
    for node in query["relations"] + query["semantic_operators"]:
        node["node_id"] = ids[node["node_id"]]
    for region in query.get("tree_regions", []) + query.get("predicate_regions", []):
        for key in ("region_id", "base_node_id"):
            if key in region:
                region[key] = ids[region[key]]
        for key in ("nodes", "base_nodes"):
            if key in region:
                region[key] = [ids[n] for n in region[key]]
    return query


def frozen_features(query, profile, spec, history):
    # SearchProblem owns the isolated copy; only override the outer context here.
    problem = SearchProblem(query, {**profile, "history": history})
    return replay_features(problem, spec)


def replay_features(problem, spec):
    """Replay candidates in one frozen context, with branch-local statistics."""
    trace = problem.replay(
        spec, refresh=lambda state: refresh_statistics(problem, state)
    )
    state = trace[-1]
    features = [
        {"key": state_key(s), "depth": len(s.history), "features": encode(problem, s)}
        for s in trace
    ]
    audit = audit_trajectory(problem, trace, features)
    return problem, state, features, audit
