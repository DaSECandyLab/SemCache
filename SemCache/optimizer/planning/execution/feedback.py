"""Sample calibration, frozen feedback states and serving-cache reset."""

import copy
import math
import urllib.request

from planning.learning.training.trajectory import trajectory
from planning.state import (
    SearchProblem,
    SearchState,
    copy_profile,
    copy_statistics,
)


class SampleCalibration:
    """Online moments for new sample observations, preserving supplied priors."""

    def __init__(self, profile, *, resume=True):
        self.profile = profile
        self.moments = {}
        self.resume = resume

    def _add(self, bank, owner, key, values, metric, basis):
        if not values:
            return
        entries = self.profile.setdefault(bank, {})
        if owner is not None:
            entries = entries.setdefault(owner, {})
        old = entries.get(key)
        counted = bank == "request_profiles"
        if old is not None and old.get("basis") != basis and not counted:
            return
        identity = bank, owner, key
        if identity not in self.moments:
            seed = (0, 0.0, 0.0)
            if counted and self.resume and old is not None:
                seed = old["samples"], old[metric], old["latency_m2"]
                if (
                    isinstance(seed[0], bool)
                    or not isinstance(seed[0], int)
                    or seed[0] < 1
                    or any(
                        isinstance(x, bool) or not math.isfinite(x) or x < 0
                        for x in seed[1:]
                    )
                ):
                    raise ValueError("Invalid counted latency prior")
            self.moments[identity] = seed
        n, mean, m2 = self.moments[identity]
        for value in values:
            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid sampled latency")
            n += 1
            delta = value - mean
            mean += delta / n
            m2 += delta * (value - mean)
        self.moments[(bank, owner, key)] = n, mean, m2
        result = {metric: mean, "basis": basis, "samples": n, "latency_m2": m2}
        if counted:
            result["basis"] = "counted initial and online atomic request latency"
        if metric == "seconds_per_request":
            result["latency_cv"] = (
                math.sqrt(max(0.0, m2) / (n - 1)) / mean if n > 1 and mean > 0 else None
            )
        entries[key] = result

    def observe(self, observation):
        model = observation["model_id"]
        if model:
            values = [
                e["latency_ms"] / 1000
                for e in observation["events"]
                if e.get("event") == "semantic_row"
                and e.get("parse_status") == "validated"
                and e.get("sampling_batch_size", 1) == 1
                and not (e.get("submit_ns") == 0 and e.get("response_ns") == 0)
                and e.get("latency_ms") is not None
            ]
            self._add(
                "request_profiles",
                model,
                observation["kind"],
                values,
                "seconds_per_request",
                "sampled warm request latency",
            )
            if observation.get("task_key"):
                self._add(
                    "task_request_profiles",
                    model,
                    observation["task_key"],
                    values,
                    "seconds_per_request",
                    "sampled task request latency",
                )
        else:
            count = math.prod(observation["inputs"])
            if count:
                self._add(
                    "relational_profiles",
                    None,
                    str(observation["operator_id"]),
                    [observation["seconds"] / count],
                    "seconds_per_input",
                    "sampled materialized SQL stage",
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
