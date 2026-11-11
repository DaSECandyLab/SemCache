"""Coordinate native Beam prediction and materialize search results."""

import random
from contextlib import nullcontext

from planning.search.heuristics import action_priorities

from .materialization import (
    materialize_result,
    materialize_state,
)
from .raw_batch import RawFeatureContext
from .sampling import apply_context


def predict_scores(problem, context, model, batch):
    """Preserve value models and public score callbacks at the batch boundary."""
    if hasattr(model, "values_features"):
        return context.predict(model, batch)
    profile, history = context.context(batch)
    apply_context(problem, profile, history)
    states = [materialize_state(problem, row) for row in batch["states"]]
    return model.scores(problem, states)


def run_resident(search, problem):
    """Only raw-statistics prediction crosses Python while C++ drives search."""
    from . import _search_core

    proxy = problem.sampler
    # Every completion must be reached through this policy's legal priority groups.
    problem.sampler = None
    try:
        _, _, original, original_error = search._start(problem)
    finally:
        problem.sampler = proxy
    context = RawFeatureContext(problem)
    learned = search.uses_value_model

    def predict(batch):
        if learned:
            return predict_scores(problem, context, search.ranker, batch)
        return context.request_prediction(batch)

    engine = _search_core.NativeSearchEngine(
        problem.native_problem, proxy.native_sampler
    )
    search.native_session = engine
    search.backend = "cpp17-resident-full"
    session = getattr(search.ranker, "inference_session", None) if learned else None
    if callable(session):
        search.ranker.eval()
    with session() if callable(session) else nullcontext():
        native_config = {
            "beam_width": search.config.beam_width,
            "max_expansions": search.config.max_expansions,
            "max_seconds": search.config.max_seconds,
            "training": search.config.training,
            "preference": search.config.preference,
            "layer_pruning": search.config.layer_pruning,
            "max_structure_groups": search.config.max_structure_groups,
            "max_implementations_per_group": search.config.max_implementations_per_group,
            "mode": search.config.plan_generation,
            "compact_requests": search.config.request_guided,
            "compact_features": learned
            and hasattr(search.ranker, "values_features"),
            "feature_context": context.native_features(
                search.ranker, workers=search.config.workers
            )
            if learned
            else None,
            "planner_workers": search.config.workers,
            "fixed_model_id": search._fixed_model_id,
            "exploration_fraction": search.config.exploration_fraction,
            "a_min": None
            if search.config.request_guided
            else problem.constraints.get("a_min"),
            "action_priorities": action_priorities(
                problem, search.config.priority_mode
            ),
            "trace_timings": proxy._config.get("sampling", {}).get(
                "trace_timings", False
            ),
        }
        raw = engine.search(
            config=native_config,
            predict=predict,
            random_state=random.Random(search.config.seed).getstate()[1],
            original=[state.native_handle for state in original],
        )
    proxy.finish(problem, raw)
    complete, _ = materialize_result(problem, raw)
    problem._resident_completed = complete
    search.trace = raw["trace"]
    result = finish_resident(search, problem, raw, complete, original_error)
    result["planner_workers"] = search.config.workers
    if search.config.uses_join_filter_cost_pruning:
        result["join_filter_cost_pruning"] = {"trace": raw["join_filter_trace"]}
    if search.config.layer_pruning == "grouped":
        result["group_pruning"] = {
            "group_score": "arithmetic_mean_of_retained_predictions",
            "max_structure_groups": search.config.max_structure_groups,
            "max_implementations_per_group": search.config.max_implementations_per_group,
            "trace": raw["group_trace"],
        }
    return result


def finish_resident(search, problem, raw, complete, original_error):
    """Keep native selection decisions under the common public result contract."""
    from planning.search.results import finish_search

    return finish_search(
        search,
        problem,
        complete,
        scores=raw["scores"],
        times=raw["times"],
        accuracies=raw["accuracies"],
        time_reasons=raw["time_reasons"],
        decision=raw["selection"],
        exhausted=raw["budget_exhausted"],
        action_budget_exhausted=raw["action_budget_exhausted"],
        expansions=raw["expansions"],
        original_error=original_error,
        heuristic_guard={
            "status": raw["heuristic_guard_status"],
            "complete_index": raw["heuristic_guard_complete"],
        },
    )
