"""Beam candidates, frozen sampling observations and selection artifacts."""

import random
import time
from contextlib import nullcontext
from dataclasses import dataclass

from planning.backends.sampling import NativeSamplingFactory
from planning.execution.feedback import freeze_problem
from planning.kernels.sampling import prepare_profile
from planning.observations.request_timing import record_phase
from planning.state import SearchProblem, SearchState, digest

from training.artifacts import save
from training.collection.policy import (
    PLAN_GENERATION_POLICIES,
    logical_structure_signature,
    select_accuracy_rollout,
    select_rollout,
)
from training.collection.prepared_sampling import (
    merge_prepared_statistics,
    prepared_descriptor,
)


@dataclass
class CandidateBatch:
    frozen: SearchProblem
    states: list[SearchState]
    result: dict
    selected_spec: dict | None
    uses_value_model: bool


def add_unseen_structure(problem, search, result, states, excluded):
    """Recover a legal structural exploration candidate when the beam collapses."""
    from planning.search.heuristics import action_priorities

    excluded = set(excluded)
    if not excluded or any(
        logical_structure_signature(spec, problem.query) not in excluded
        for spec in result["complete"]
    ):
        return
    priorities = (
        None if search.config.priority_mode == "value"
        else action_priorities(problem, search.config.priority_mode)
    )
    rng = random.Random(search.config.seed + 15485863)

    def refresh_candidate(state):
        """Refresh optional accuracy evidence without rejecting a legal plan."""
        try:
            return problem.update_accuracy(state)
        except (KeyError, IndexError):
            # Diversity exploration may produce unobserved legal states; score their available partial features and replay before execution.
            return state
    for attempt in range(256):
        state = problem.initial
        for _ in range(len(problem.all_ops) + 1):
            if problem.complete(state):
                break
            options = problem.actions(state, highest_priorities=priorities)
            if not options:
                break
            groups = {}
            for action in options:
                groups.setdefault(action["operator_id"], []).append(action)
            state = problem.transition(
                state, rng.choice(groups[rng.choice(sorted(groups))]), observe=False
            )
        if not problem.complete(state):
            continue
        spec = problem.spec(state)
        if logical_structure_signature(spec, problem.query) in excluded:
            continue
        state = refresh_candidate(state)
        search.refresh_context(problem, [state])
        prediction = search.predictions(problem, [state])
        score = prediction["scores"][0]
        seconds, accuracy = prediction["times"][0], prediction["accuracies"][0]
        search._resident_scores[state.history] = tuple(score)
        states.append(state)
        result["complete"].append(spec)
        result["assessments"].append({
            "plan_key": digest(state.history), "accuracy": accuracy,
            "time_prediction": seconds, "basis": "learned completion value",
            "status": "feasible" if accuracy >= problem.constraints.get("a_min", 0)
            else "infeasible",
        })
        result["structural_diversity_exploration"] = {
            "attempts": attempt + 1, "policy": "highest_legal_priority_group",
            "additional_sampling": False,
        }
        return
    # If priority groups prevent topology diversity, widen only structural exploration; Beam ranking retains its priority policy.
    if priorities is not None:
        for attempt in range(256, 320):
            state = problem.initial
            for _ in range(len(problem.all_ops) + 1):
                if problem.complete(state):
                    break
                options = problem.actions(state, highest_priorities=None)
                if not options:
                    break
                groups = {}
                for action in options:
                    groups.setdefault(action["operator_id"], []).append(action)
                state = problem.transition(
                    state, rng.choice(groups[rng.choice(sorted(groups))])
                )
            if not problem.complete(state):
                continue
            spec = problem.spec(state)
            if logical_structure_signature(spec, problem.query) in excluded:
                continue
            state = refresh_candidate(state)
            search.refresh_context(problem, [state])
            prediction = search.predictions(problem, [state])
            score = prediction["scores"][0]
            seconds, accuracy = prediction["times"][0], prediction["accuracies"][0]
            search._resident_scores[state.history] = tuple(score)
            states.append(state)
            result["complete"].append(spec)
            result["assessments"].append({
                "plan_key": digest(state.history), "accuracy": accuracy,
                "time_prediction": seconds, "basis": "learned completion value",
                "status": "feasible" if accuracy >= problem.constraints.get("a_min", 0)
                else "infeasible",
            })
            result["structural_diversity_exploration"] = {
                "attempts": attempt + 1,
                "policy": "fallback_all_legal_actions",
                "additional_sampling": False,
            }
            return
    raise ValueError("Could not construct a third distinct legal plan structure")


def search_candidates(
    problem,
    search,
    *,
    build,
    models,
    database,
    history,
    sampling_config,
    teacher,
    catalog,
    output,
    workload_id,
    phase,
    timings,
    online_timing,
    descriptive_profile,
    plan_generation,
    selection_policy,
    progress,
    minimum_accuracy=None,
    excluded_plan_signatures=(),
    excluded_structure_signatures=(),
    prepared_statistics=None,
):
    """Run one sampled Beam and freeze its completions before making labels."""
    request_guided = search.config.request_guided
    measured_search_policy = PLAN_GENERATION_POLICIES[plan_generation]
    selected_spec = None
    ranker = search.ranker
    progress(f"{workload_id}: online beam sampling and search", flush=True)
    sampling_started = time.perf_counter()
    if prepared_statistics is not None:
        problem.profile = merge_prepared_statistics(
            problem.query, problem.profile, prepared_statistics
        )
        problem.constraints = problem.profile.get("constraints", {})
        problem.catalog = problem.profile.get("models", catalog)
        # ProfileEnvironment has no sampler-side history provider. Publish the
        # current round's ledger explicitly and never reuse the first search's.
        problem.cache_history = tuple(history.snapshot())
        sampling_context = nullcontext(prepare_profile(problem))
    else:
        sampling_context = NativeSamplingFactory(
            build,
            models,
            database=database,
            history=history,
            config=sampling_config,
            teacher=teacher,
            catalog=catalog,
        )(problem)
    with sampling_context as sampler:
        problem.sampler = sampler
        timings.checkpoint("sampler_initialization")
        result = search.run(problem)
        result["operator_priority_mode"] = search.config.priority_mode
        result["structure_policy"] = (
            "all_legal_operators_layouts_models_value"
            if plan_generation == "value"
            else "highest_legal_group_fixed_gotdf_requests"
            if request_guided
            else "highest_legal_group_all_layouts_models_value"
        )
        if search.uses_value_model:
            result["value_model_device"] = str(next(ranker.parameters()).device)
            result["value_model_gpu_batch_size"] = ranker.inference_batch_size
        states = getattr(problem, "_resident_completed", None)
        if states is None:
            states = [problem.replay(spec)[-1] for spec in result["complete"]]
        if phase == "measured":
            if selection_policy == "minimum_accuracy":
                add_unseen_structure(
                    problem, search, result, states, excluded_structure_signatures
                )
            eligible = (
                states if selection_policy == "minimum_accuracy" else result["states"]
            )
            if not eligible:
                raise ValueError("Beam produced no eligible complete plan")
            scores = [search._resident_scores[state.history] for state in eligible]
            if selection_policy == "minimum_accuracy":
                assessments = result["assessments"]
                if len(assessments) != len(eligible):
                    raise ValueError("Complete plan predictions are misaligned")
                chosen, selection = select_accuracy_rollout(
                    scores,
                    [row["accuracy"] for row in assessments],
                    [row["time_prediction"] for row in assessments],
                    result["complete"],
                    search.config.seed,
                    minimum_accuracy,
                    excluded_plan_signatures,
                    excluded_structure_signatures,
                    query=problem.query,
                )
                result.update(
                    constraint_satisfied=selection["constraint_satisfied"],
                    accuracy_fallback=selection["accuracy_fallback"],
                    selection_basis=selection["selection_basis"],
                )
                if selection["accuracy_fallback"]:
                    result["reason"] = (
                        "Selected a random unseen plan after detecting a duplicate; "
                        "the original minimum accuracy is not satisfied"
                        if selection["duplicate_replaced"]
                        else "Selected the highest predicted accuracy completion; "
                        "the original minimum accuracy is not satisfied"
                    )
                else:
                    result["reason"] = None
            else:
                chosen, selection = select_rollout(
                    scores, search.config.seed, selection_policy
                )
            selected_spec = problem.spec(eligible[chosen])
            result["selected"] = selected_spec
            result["rollout_selection"] = selection
        timings.checkpoint("beam_search")
        beam_timing = timings.values["beam_search"]
        start_ns, end_ns = beam_timing["intervals_ns"][-1]
        latency = (
            {"wall_seconds": 0.0}
            if prepared_statistics is not None
            else sampler.request_latency(start_ns=start_ns, end_ns=end_ns)
        )
        beam_timing["llm_wall_seconds"] = min(
            beam_timing["wall_seconds"], latency["wall_seconds"]
        )
        beam_timing["non_llm_wall_seconds"] = (
            beam_timing["wall_seconds"] - beam_timing["llm_wall_seconds"]
        )
        save(
            output / "search.json",
            {k: v for k, v in result.items() if k != "states"},
        )
        timings.checkpoint("search_artifact_save")
        progress(
            f"{workload_id}: search expanded {result['expansions']} states; "
            f"completed={len(result['complete'])}; budget_exhausted={result['budget_exhausted']}",
            flush=True,
        )
        if not states:
            raise ValueError("Beam search produced no legal complete plans")
        frozen, states = freeze_problem(problem, states)
        timings.checkpoint("freeze_and_replay")
        progress(
            f"{workload_id}: {len(states)} distinct complete plans; sampling={sampler.calls}",
            flush=True,
        )
        frozen.profile.pop("history", None)
        all_latency = (
            {
                "reported_requests": 0,
                "seconds": 0.0,
                "wall_seconds": 0.0,
                "source": "prepared ProfileEnvironment; no new sampling",
                "http_requests_by_model": {},
            }
            if prepared_statistics is not None
            else sampler.request_latency()
        )
        sampling_artifact = {
            "calls": {} if prepared_statistics is not None else sampler.calls,
            "backend_timings": sampler.timings.values,
            "request_latency": all_latency,
            "http_requests_by_model": all_latency["http_requests_by_model"],
            "adaptive_sketch": sampler.sketch_config,
            "base_sketch_plans": sampler.population_sketch_plans,
            "statistics_source": "DuckDB relation estimates + online semantic samples; unknowns retain masks",
        }
        if prepared_statistics is not None:
            sampling_artifact.update(prepared_descriptor(prepared_statistics))
        save(output / "sampling.json", sampling_artifact)
    # Descriptive response timing is separate from isolated cost priors.
    # Harvest each archive once; scheduler completions and result reuse
    # are not additional physical response samples.
    online_timing["sampling"] = record_phase(
        descriptive_profile,
        [] if prepared_statistics is not None else sampler.backend.events,
        source="sampling",
        scope=str(output.resolve()) + "/sampling",
        wall_seconds=time.perf_counter() - sampling_started,
        wall_basis="sampler context elapsed including initialization and cleanup",
    )
    if prepared_statistics is not None:
        online_timing["sampling"].update(
            requests=0, request_count_source="prepared_statistics_no_http"
        )
    save(output / "online-timing.json", online_timing)
    frozen.profile.update(
        {
            k: v
            for k, v in problem.profile.items()
            if k
            in (
                "operator_time_calibration",
                "relational_profiles",
                "request_profiles",
                "accuracy",
                "task_accuracy",
                "task_request_profiles",
            )
        }
    )
    if phase == "measured":
        save(
            output / "candidate-selection.json",
            {
                "policy": measured_search_policy,
                "plan_generation": plan_generation,
                "search_runs": 1,
                "value_model_required": not request_guided,
                "score_basis": "calibrated_input_rows"
                if request_guided
                else "value_model",
                "layout_policy": "fixed_gotdf" if request_guided else "search",
                "operator_policy": "all_legal"
                if plan_generation == "value"
                else "highest_legal_group",
                "operator_priority_mode": search.config.priority_mode,
                "rollout_selection": result["rollout_selection"],
                "selected": selected_spec,
                "heuristic_guard": result["heuristic_guard"],
                "exploration_fraction": search.config.exploration_fraction,
                "candidates": len(states),
            },
        )
    return CandidateBatch(
        frozen, states, result, selected_spec, search.uses_value_model
    )
