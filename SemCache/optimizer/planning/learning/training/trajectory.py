"""Freeze and backfill each executed plan's complete state trajectory."""

import copy

from planning.learning.features.graph import encode
from planning.state import digest

from .data import dataset, validate_dataset


def state_key(state):
    return digest(
        {
            "actions": sorted(state.actions, key=lambda a: a["operator_id"]),
            "roots": state.partial_plans.roots,
            "remaining": state.partial_plans.remaining,
        }
    )


def refresh_statistics(problem, state):
    """Refresh a frozen prefix before its statistics reach the next transition."""
    if not problem.sampler:
        for chosen in problem.semantic_actions(state):
            values = state.statistics[digest(chosen)]
            values.pop("l_existing_features", None)
            values.update(problem.estimate(state, chosen, state.partial_plans))


def trajectory(problem, state):
    resident = getattr(state, "_resident_trajectory", None)
    if resident is not None:
        # Native exports are already recontextualized against the final common
        # calibration/history and retain each prefix's own sampling cohort.
        problem.native_problem.complete(state.native_handle)
        return list(resident)
    current = problem.initial
    states = [current]
    for action in state.actions:
        current = problem.transition(current, action)
        refresh_statistics(problem, current)
        states.append(current)
    return states


def experience(
    problem,
    state,
    *,
    source,
    times=None,
    accuracy=None,
    context="cold",
    metadata=None,
    time_lower_bound=None,
    trajectory_features=None,
):
    """Features precede outcomes; censored execution is not a completed latency."""
    if source not in ("estimated", "measured") or not problem.complete(state):
        raise ValueError("Experience requires a complete plan and a known label source")
    record = {
        "query_key": problem.key,
        "query_id": problem.query.get("query_id", problem.key),
        "split": problem.query.get("training_split", "train"),
        "split_group": problem.query.get("training_group", problem.key),
        "dataset": problem.query.get("training_dataset", "default"),
        "context": context,
        "source": source,
        "plan_key": state_key(state),
        "plan": problem.spec(state),
        "trajectory": trajectory_features
        if trajectory_features is not None
        else [
            {
                "key": state_key(s),
                "depth": len(s.history),
                "features": encode(problem, s),
            }
            for s in trajectory(problem, state)
        ],
        "times": list(times or []),
        "accuracy": accuracy,
        "time_lower_bound": time_lower_bound,
        "status": "timeout"
        if time_lower_bound is not None
        else "estimated_unknown"
        if source == "estimated" and not times
        else "completed",
        "metadata": copy.deepcopy(metadata or {}),
    }
    validate_dataset(dataset([record]))
    return record
