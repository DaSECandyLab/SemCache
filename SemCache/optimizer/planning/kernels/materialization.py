"""Restore native Beam structures and statistics without carrying sample rows."""

import json

from planning.observations.evidence import freeze_evidence

from ..state import SearchState, action_key


def materialize_state(problem, raw):
    """Create Python views after the native loop has released its search frame."""
    handle = raw["state"]
    problem.native_problem.complete(handle)   
    owner = problem._native_owner
    history = tuple(handle.history)
    actions = tuple(freeze_evidence(json.loads(value)) for value in history)
    keys = {
        problem.all_ops[action["operator_id"]]["node_id"]: action_key(action)
        for action in actions
    }
    statistics = {
        keys.get(int(nid), f"scan:{nid}"): info
        for nid, info in raw["statistics"].items()
    }
    return SearchState(
        history,
        statistics,
        None,
        owner.partial_plans(handle),
        {},
        handle,
        actions,
    )


def materialize_result(problem, raw):
    """Retain every complete plan and its frozen native prefix observations."""
    states = {}

    def materialize(row):
        identity = row["state"].handle_id
        if identity not in states:
            states[identity] = materialize_state(problem, row)
        return states[identity]

    complete = [materialize(row) for row in raw["completed"]]
    paths = raw.get("trajectories", ())
    if len(paths) != len(complete):
        raise ValueError("Native result omitted frozen completion trajectories")
    trajectories = []
    for state, path in zip(complete, paths):
        if (
            not path
            or path[0]["state"].depth != 0
            or path[-1]["state"].handle_id != state.native_handle.handle_id
        ):
            raise ValueError("Native result has an incomplete frozen trajectory")
        prefix = tuple(materialize(row) for row in path[:-1]) + (state,)
        if [len(row.history) for row in prefix] != list(range(len(prefix))):
            raise ValueError("Native result has a discontinuous frozen trajectory")
        object.__setattr__(state, "_resident_trajectory", prefix)
        trajectories.append(prefix)
    return complete, trajectories
