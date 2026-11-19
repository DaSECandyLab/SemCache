"""Frozen feedback states and serving-cache reset."""

import copy
import urllib.request

from planning.learning.training.trajectory import trajectory
from planning.state import (
    SearchProblem,
    SearchState,
    copy_profile,
    copy_statistics,
)


def freeze_problem(problem, candidates):
    """Freeze sampling/global quality before labels; subsequent replay makes no model calls."""
    profile = copy_profile(problem.profile)
    profile.pop("conditional_inputs", None)
    profile.pop("prefix_statistics", None)
    frozen = SearchProblem(problem.query, profile)
    if frozen.native_problem is not None:
        # Rebind structural ownership without rerunning transition observation
        # or copying native sample payloads. Cache estimates were refreshed in a
        # common final context before the native trajectories were exported.
        initial = frozen.initial
        native = frozen.native_problem
        owner = frozen._native_owner
        if native is None or owner is None:
            raise ValueError(
                "Resident trajectories require native structural ownership"
            )
        imported = {}
        states = []
        for state in candidates:
            path = []
            source_path = getattr(state, "_resident_trajectory", None)
            if source_path is None:
                # Structural exploration can append an ordinary Python-built
                # completion to resident Beam states. Obtain its path from its
                # original owner before importing every handle into `frozen`.
                source_path = trajectory(problem, state)
            for original in source_path:
                if original.history not in imported:
                    handle = initial.native_handle
                    if original.history:
                        parent = imported[original.history[:-1]]
                        handle = native.transition(
                            parent.native_handle, original.actions[-1]
                        )
                    imported[original.history] = SearchState(
                        original.history,
                        copy_statistics(original.statistics),
                        copy.copy(original.sample),
                        owner.partial_plans(handle),
                        dict(original.sample_observations),
                        handle,
                        original.action_views,
                    )
                current = imported[original.history]
                path.append(current)
            object.__setattr__(path[-1], "_resident_trajectory", tuple(path))
            states.append(path[-1])
        return frozen, states
    return frozen, [trajectory(frozen, state)[-1] for state in candidates]


def reset_serving_cache(models):
    """Explicit experiment operation for an isolated SGLang service."""
    results = []
    for endpoint in sorted({m["base_url"].rstrip("/") for m in models.values()}):
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
            urllib.request.Request(endpoint + "/flush_cache?timeout=20", method="POST"),
            timeout=30,
        ) as reply:
            if reply.status != 200:
                raise ValueError("Serving cache reset failed")
            results.append(
                {
                    "endpoint": endpoint,
                    "status": "flushed",
                    "response": reply.read().decode(errors="replace"),
                }
            )
    return results
