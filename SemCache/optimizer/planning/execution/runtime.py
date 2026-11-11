"""Fixed-tree runtime implementation search using the frozen completion model."""

import copy
import math

from planning.observations.statistics import execution_rows, record_history
from planning.state import SearchProblem, canonical


def update_history(events, query, history, adaptive=None):
    """Actual generation-specific input rows; never combine different layouts."""
    operators = {o["operator_id"]: o for o in query["semantic_operators"]}
    groups = {}
    for row in events:
        if (
            row.get("event") == "semantic_row"
            and row.get("parse_status") == "validated"
        ):
            groups.setdefault(
                (row["operator_id"], row.get("config_generation", 0)), []
            ).append(row)
    for (oid, _), rows in groups.items():
        first, op = rows[0], operators[oid]
        keys = op["default_layout"].keys()
        action = {
            "operator_id": oid,
            "model_id": first["model_id"],
            "layout": {k: first[k] for k in keys},
        }
        record_history(
            history,
            op,
            action,
            execution_rows(rows)[oid],
            started_at=first.get("observed_unix_seconds"),
            adaptive=adaptive,
        )


def implementation_key(action):
    return action["model_id"], canonical(action["layout"])


def runtime_problem(query, profile, request, history):
    """One frozen context for both the current plan and candidate implementations."""
    if request.get("version") != 1 or request["query_id"] != query["query_id"]:
        raise ValueError("Runtime request identity mismatch")
    problem = SearchProblem(query, copy.deepcopy(profile))
    for oid, statistics in request.get("runtime_input_statistics", {}).items():
        problem.profile.setdefault("operator_statistics", {}).setdefault(oid, {})[
            "input_field_statistics"
        ] = statistics["input_field_statistics"]
    entries = {e["operator_id"]: e for e in request["state"]}
    if set(entries) != set(problem.ops):
        raise ValueError("Runtime state has missing/extra operators")
    if request.get("events_complete") and not request.get("chunk_policy"):
        update_history(
            request["events"], query, history, profile.get("adaptive_sketch")
        )
    snapshot = {entry["id"]: entry for entry in profile.get("history", [])}
    if history is not None:
        records = history if isinstance(history, (list, tuple)) else history.snapshot()
        snapshot.update({entry["id"]: entry for entry in records})
    problem.cache_history = tuple(snapshot.values())
    observed, latencies = {}, {}
    for row in request["observations"]:
        if row["operator_id"] not in entries or row["completed"] < 1:
            raise ValueError("Invalid runtime observation")
        if (
            row["latency_avg_ms"] is not None
            and (not math.isfinite(row["latency_avg_ms"]) or row["latency_avg_ms"] < 0)
            or not 0 <= row["selectivity"] <= 1
        ):
            raise ValueError("Invalid runtime latency/selectivity")
        action = row["action"]
        if action["operator_id"] != row["operator_id"]:
            raise ValueError("Runtime observation action identity mismatch")
        if row["latency_avg_ms"] is not None:
            key = action["model_id"], problem.ops[action["operator_id"]]["kind"]
            count = row.get("llm_calls", row["completed"])
            seconds, samples = latencies.get(key, (0.0, 0))
            latencies[key] = (
                seconds + row["latency_avg_ms"] / 1000 * count,
                samples + count,
            )
        # Keep latency calibration model-specific. Chunk-mode selectivity is
        # shared when evaluating other implementations of the same operator.
        observed[(row["operator_id"], implementation_key(action))] = row
    for (model_id, kind), (seconds, samples) in latencies.items():
        if samples:
            problem.profile.setdefault("request_profiles", {}).setdefault(model_id, {})[
                kind
            ] = {
                "seconds_per_request": seconds / samples,
                "samples": samples,
                "basis": "runtime actual HTTP responses, weighted across configuration generations",
            }
    return problem, entries, observed


def current_spec(request, updates=()):
    """Retain tree choices, ON conditions and semantic order exactly as bound."""
    spec = copy.deepcopy(request["plan_spec"])
    choices = {entry["operator_id"]: entry["current"] for entry in request["state"]}
    for update in updates:
        choices[update["operator_id"]] = update
    for action in spec["actions"]:
        choice = choices[action["operator_id"]]
        action.update(
            {key: copy.deepcopy(choice[key]) for key in ("model_id", "layout")}
        )
    return spec


def runtime_snapshot(query, profile, request, history, *, observed=True, updates=()):
    """Materialize current runtime features without sampling or new LLM calls."""
    from planning.kernels.sampling import prepare_profile
    from planning.learning.features.graph import encode

    problem, entries, observations = runtime_problem(query, profile, request, history)
    path = problem.replay(current_spec(request, updates), structural_only=True)
    proxy = prepare_profile(problem)
    environment = proxy.native_sampler
    for state in path:
        environment.observe(state.native_handle)
        if observed and state.actions:
            action = state.actions[-1]
            oid = action["operator_id"]
            if oid in entries:
                observation = observations.get((oid, implementation_key(action)))
                if observation is None and request.get("chunk_policy"):
                    observation = next(
                        (
                            o
                            for o in request["observations"]
                            if o["operator_id"] == oid
                            and o["config_generation"]
                            == entries[oid]["current"]["config_generation"]
                        ),
                        None,
                    )
                environment.apply_runtime(
                    state.native_handle, oid, entries[oid], observation
                )
    state = proxy.state(problem, path[-1].native_handle)
    return problem, state, encode(problem, state)


def optimize_runtime(model, query, profile, request, history, config):
    if not model or not model.value_available:
        raise ValueError("Runtime value policy requires a trained checkpoint")
    problem, entries, observed = runtime_problem(query, profile, request, history)
    baseline = problem.replay(current_spec(request), structural_only=True)
    from planning.kernels import _search_core
    from planning.kernels.raw_batch import RawFeatureContext
    from planning.kernels.resident import predict_scores
    from planning.kernels.sampling import prepare_profile

    environment = prepare_profile(problem).native_sampler
    context = RawFeatureContext(problem)
    native = problem.native_problem
    if native is None:
        raise ValueError("Runtime optimization requires native structural ownership")
    return _search_core.search_runtime(
        native,
        environment,
        {
            "request": request,
            "baseline_actions": [step.actions[-1] for step in baseline[1:]],
            "domain": {str(oid): choices for oid, choices in problem.domain.items()},
            "beam_width": config.beam_width,
            "seed": config.seed,
            "max_expansions": config.max_expansions,
            "max_seconds": config.max_seconds,
            "a_min": problem.constraints.get("a_min"),
            "preference": config.preference,
        },
        lambda batch: predict_scores(problem, context, model, batch),
    )
