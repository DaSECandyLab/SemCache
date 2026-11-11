"""Measure one representative per operator kind before a whole workload runs.

The coordinator constructs legal plans without executing them. Only explicitly
selected representatives and their necessary input producers reach the native
calibrator; it never runs Beam, selectivity preparation, or teacher probes.
"""

import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path


def _fingerprint(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _canonical_state(problem, mode, fixed_model_id):
    from planning.search.heuristics import action_priorities

    native = problem.native_problem
    state = native.initial()
    priorities = action_priorities(problem, mode)
    while not native.complete(state):
        actions = native.actions(
            state,
            fixed_layout=True,
            highest_priorities=priorities,
            fixed_model_id=fixed_model_id if mode == "heuristic_requests" else None,
        )
        if not actions:
            raise ValueError("Cannot construct a legal operator-calibration plan")
        state = native.transition(state, actions[0])
    return state


def _kind(problem, node):
    if node.operator_id is None:
        return "SCAN"
    op = problem.all_ops[node.operator_id]
    kind = op.get("kind", op.get("type", "")).upper()
    if not kind:
        raise ValueError(f"Calibration operator {node.operator_id} has no kind")
    return kind


def _audit_models(models):
    # Keep credentials out of the reusable evidence and its progress callbacks.
    public = (
        "base_url",
        "model",
        "structured_output",
        "enable_thinking",
        "max_tokens",
        "supported_kinds",
        "context_length",
    )
    return {
        name: {key: copy.deepcopy(config[key]) for key in public if key in config}
        for name, config in models.items()
    }


def _coefficient(record, *, kind, representative):
    result = copy.deepcopy(record)
    seconds = result.get("seconds_per_input")
    units = result.get("input_units")
    repeats = result.get("repeats")
    if (
        type(seconds) not in (int, float)
        or not math.isfinite(seconds)
        or seconds < 0
        or type(units) not in (int, float)
        or not math.isfinite(units)
        or units <= 0
        or type(repeats) is not int
        or repeats < 1
    ):
        raise ValueError(f"No measured workload calibration for {kind}")
    if result.get("node_id") != representative["node_id"]:
        raise ValueError(f"Native calibrator used the wrong representative for {kind}")
    result.update(
        query_id=representative["query_id"],
        source_database=representative["source_database"],
        query_sha256=representative["query_sha256"],
        representative=copy.deepcopy(representative),
    )
    return result


def calibrate_workload(
    items,
    *,
    build,
    models,
    catalog,
    settings,
    mode="heuristic_requests",
    fixed_model_id=None,
    progress=None,
):
    """Return frozen JSON coefficients shared by every query in ``items``.

    Each item supplies ``id``, normalized ``query`` and ``database``. Selection
    is uniform over all actual nodes of each kind across the whole workload,
    using ``calibration.operator_time.seed`` and stable query/node ordering.
    ``progress(**event)`` receives isolated JSON snapshots at selection,
    calibration-start, type-completion and workload-completion boundaries.

    Callers must establish their run's cache boundary before calling this
    function. This function never resets serving caches, writes artifacts, or
    mutates the caller's query/settings/history. Target HTTP and necessary
    producer HTTP remain separate in ``calibration_runs``.
    """
    from planning.backends.sampling import NativeSamplingFactory
    from planning.observations.cache import CacheHistory
    from planning.settings.modes import (
        configured_fixed_model,
        models_for_mode,
        profile_for_mode,
        sampling_for_mode,
        validate_mode,
    )
    from planning.state import SearchProblem

    validate_mode(mode)
    items = list(items)
    if not items:
        raise ValueError("Workload calibration requires at least one query")
    identifiers = [item.get("id") for item in items]
    if any(not isinstance(key, str) or not key for key in identifiers) or len(
        set(identifiers)
    ) != len(identifiers):
        raise ValueError("Workload calibration requires unique nonempty query IDs")
    fixed = fixed_model_id or configured_fixed_model(settings)
    models, catalog = models_for_mode(
        models, catalog, settings, mode, fixed_model_id=fixed
    )
    timing = copy.deepcopy(settings["calibration"]["operator_time"])
    seed = timing.get("seed", 42)
    if type(seed) is not int:
        raise ValueError("Workload calibration seed must be an integer")
    candidates, prepared = {}, {}
    workload = []
    for item in sorted(items, key=lambda entry: entry["id"]):
        key = item["id"]
        database = str(Path(item["database"]).resolve())
        query_sha256 = _fingerprint(item["query"])
        profile = profile_for_mode(
            {"models": catalog, "constraints": {"allowed_models": list(models)}},
            item["query"],
            mode,
            fixed_model_id=fixed,
        )
        problem = SearchProblem(item["query"], profile)
        state = _canonical_state(problem, mode, fixed)
        prepared[key] = (problem, state, database)
        workload.append(
            {"query_id": key, "source_database": database, "query_sha256": query_sha256}
        )
        for node in sorted(state.nodes(), key=lambda value: value.node_id):
            kind = _kind(problem, node)
            candidates.setdefault(kind, []).append(
                {
                    "query_id": key,
                    "source_database": database,
                    "query_sha256": query_sha256,
                    "node_id": node.node_id,
                    "operator_id": node.operator_id,
                    "kind": kind,
                    "semantic": node.operator_id in problem.ops,
                }
            )
    randomizer = random.Random(seed)
    representatives = {
        kind: {**randomizer.choice(choices), "candidate_count": len(choices)}
        for kind, choices in sorted(candidates.items())
    }
    if not representatives:
        raise ValueError("Workload contains no calibratable operator nodes")
    groups = {}
    for kind, representative in representatives.items():
        groups.setdefault(representative["query_id"], []).append(kind)
    evidence = {
        "protocol": 1,
        "scope": "workload",
        "frozen": False,
        "basis": "one actual workload operator per kind on independent base inputs; stage wall seconds / input units",
        "sample_rows": timing["sample_rows"],
        "seed": seed,
        "plan_generation": mode,
        "fixed_model_id": fixed if mode == "heuristic_requests" else None,
        "relational": {},
        "semantic": {},
        "representatives": copy.deepcopy(representatives),
        "workload": workload,
        "workload_sha256": _fingerprint(workload),
        "settings": {
            "operator_time": timing,
            "execution": copy.deepcopy(settings["execution"]),
            "sampling": copy.deepcopy(settings["sampling"]),
        },
        "models": _audit_models(models),
        "catalog": copy.deepcopy(catalog),
        "calibration_runs": [],
        "teacher_requests": 0,
        "accuracy_sampling_batches": 0,
    }
    completed = 0

    def notify(stage, **values):
        if progress is not None:
            progress(
                stage=stage,
                completed_types=completed,
                total_types=len(representatives),
                evidence=copy.deepcopy(evidence),
                **values,
            )

    notify("representatives_selected")
    for key, kinds in sorted(groups.items()):
        problem, state, database = prepared[key]
        selected_nodes = [representatives[kind]["node_id"] for kind in kinds]
        notify("calibrating", current_query=key, kinds=list(kinds))
        sampling = sampling_for_mode(settings["sampling"], mode, fixed_model_id=fixed)
        # Accuracy and depth sampling are never part of this independent phase,
        # including when collecting time priors for the value-model policy.
        sampling.update(sample_accuracy=False, sample_depth_selectivity=False)
        started = time.monotonic()
        with CacheHistory(":memory:") as history:
            factory = NativeSamplingFactory(
                build,
                models,
                database=database,
                setup="SET semantic_response_cache='{}'; SET semantic_request_reuse=false;",
                config=sampling,
                history=history,
                catalog=catalog,
                calibration_config={
                    **timing,
                    "parallelism": settings["execution"]["cpu_parallelism"],
                    "max_concurrent_requests_per_model": settings["execution"][
                        "max_concurrent_requests_per_model"
                    ],
                    "selected_nodes": selected_nodes,
                },
            )
            try:
                with factory(problem) as sampler:
                    returned = sampler.native_sampler.calibrate_time(state)
                    partial = (
                        returned
                        if isinstance(returned, dict) and "relational" in returned
                        else sampler.native_sampler.profile().get(
                            "operator_time_calibration", {}
                        )
                    )
                    calls = copy.deepcopy(sampler.calls)
                    if calls.get("accuracy", 0) or calls.get("judge", 0):
                        raise ValueError(
                            "Workload time calibration invoked accuracy/teacher sampling"
                        )
                    run = {
                        "query_id": key,
                        "source_database": database,
                        "selected_nodes": selected_nodes,
                        "kinds": list(kinds),
                        "calls": calls,
                        "timed_operator_http_requests": calls.get(
                            "calibration_model_requests", 0
                        ),
                        "input_preparation_http_requests": calls.get(
                            "selection_model_requests", 0
                        ),
                        "total_http_requests": calls.get("model_http_requests", 0),
                        "timings": copy.deepcopy(sampler.native_sampler.timings()),
                    }
            finally:
                factory.close()
        run["wall_seconds"] = time.monotonic() - started
        evidence["calibration_runs"].append(run)
        for kind in kinds:
            representative = representatives[kind]
            if representative["semantic"]:
                found = False
                for model, bank in partial.get("semantic", {}).items():
                    if kind in bank:
                        evidence["semantic"].setdefault(model, {})[kind] = _coefficient(
                            bank[kind], kind=kind, representative=representative
                        )
                        found = True
                if not found:
                    raise ValueError(
                        f"Missing semantic workload calibration for {kind}"
                    )
            else:
                record = partial.get("relational", {}).get(kind)
                if record is None:
                    raise ValueError(
                        f"Missing relational workload calibration for {kind}"
                    )
                evidence["relational"][kind] = _coefficient(
                    record, kind=kind, representative=representative
                )
            completed += 1
            notify("type_completed", current_query=key, current_kind=kind)
    evidence["frozen"] = True
    evidence["total_http_requests"] = sum(
        run["total_http_requests"] for run in evidence["calibration_runs"]
    )
    evidence["timed_operator_http_requests"] = sum(
        run["timed_operator_http_requests"] for run in evidence["calibration_runs"]
    )
    evidence["input_preparation_http_requests"] = sum(
        run["input_preparation_http_requests"] for run in evidence["calibration_runs"]
    )
    json.dumps(evidence, allow_nan=False)
    notify("completed")
    return evidence
