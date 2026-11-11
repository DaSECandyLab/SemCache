"""Orchestrate sampled Beam candidates and frozen native-query feedback."""

import copy
import gc
import time
from pathlib import Path

from planning.backends.execution import DuckDBEvaluator
from planning.execution.cache_reset import clear_prompt_answers
from planning.execution.feedback import reset_serving_cache
from planning.execution.timeouts import TimeoutPolicy
from planning.execution.timing import PhaseTimings
from planning.learning.devices import load_inference_model
from planning.learning.training.data import dataset, validate_dataset
from planning.observations.cache import CacheHistory
from planning.search.beam import ParetoBeamSearch, SearchConfig
from planning.settings.modes import (
    configured_fixed_model,
    models_for_mode,
    profile_for_mode,
    sampling_for_mode,
)
from planning.state import SearchProblem

from training.artifacts import save
from training.calibration.global_profile import shared_profile
from training.collection.candidates import search_candidates
from training.collection.execution import expected_requests as expected_requests
from training.collection.execution import measured_record
from training.collection.features import normalize_explain
from training.collection.labels import COLLECTION_PROTOCOL as COLLECTION_PROTOCOL
from training.collection.labels import estimated_records, record_metadata
from training.collection.policy import (
    ACCURACY_FEEDBACK_PROTOCOL,
    FEEDBACK_PROTOCOL,
    PLAN_GENERATION_POLICIES,
    validate_minimum_accuracy,
)
from training.collection.prepared_sampling import merge_prepared_statistics


def reset_context(models, history):
    # Query clients have closed before this boundary. Release any resident
    # trajectory cycles before removing the disk-backed answer cache.
    gc.collect()
    reset_serving_cache(models)
    history.clear()
    return {
        "kv_cache": "cleared",
        "history": "cleared",
        "prompt_answers": clear_prompt_answers(),
    }


def collect_native(
    workload,
    models,
    catalog,
    database,
    build,
    output,
    *,
    calibration_profile,
    checkpoint=None,
    phase="measured",
    repeats=2,
    candidates=8,
    seed=17,
    progress=print,
    history=None,
    timeout_policy=None,
    sampling_config=None,
    search_seconds=600,
    max_expansions=4096,
    teacher=None,
    exploration_fraction=0.1,
    inference_settings=None,
    max_concurrent_requests_per_model=1,
    enforce_model_operator_order=False,
    execution_cpu_parallelism=1,
    plan_generation=None,
    cache_segment=None,
    rollout_index=1,
    selection_policy="pareto_random",
    operator_priority_mode=None,
    minimum_accuracy=None,
    excluded_plan_signatures=(),
    excluded_structure_signatures=(),
    searches_per_query=2,
    prepared_statistics=None,
):
    """One independent sampled Beam -> frozen features -> one execution.

    Estimated rounds label every beam completion.  A measured round follows
    Balsa's online-feedback contract: search may produce many candidates, but
    one selected completion of that search is executed. The query orchestrator
    invokes this function for each rollout with separate search/execution artifacts. The
    candidate answer is persisted; F1 is attached immediately by the pipeline
    using the frozen teacher answer. Model updates remain at round boundaries.
    """
    if phase not in ("estimated", "measured"):
        raise ValueError("Invalid collection phase")
    if minimum_accuracy is not None:
        validate_minimum_accuracy(minimum_accuracy)
        if phase != "measured" or selection_policy != "minimum_accuracy":
            raise ValueError(
                "minimum_accuracy requires measured minimum_accuracy selection"
            )
    elif selection_policy == "minimum_accuracy":
        raise ValueError("minimum_accuracy selection requires an accuracy floor")
    if type(searches_per_query) is not int or searches_per_query < 1:
        raise ValueError("searches_per_query must be a positive integer")
    plan_generation = plan_generation or (
        "heuristic_requests" if phase == "estimated" else "heuristic_value"
    )
    if plan_generation not in PLAN_GENERATION_POLICIES:
        raise ValueError("Unknown plan generation policy")
    if phase == "estimated" and plan_generation != "heuristic_requests":
        raise ValueError("Estimated pretraining uses heuristic_requests")
    if minimum_accuracy is not None and plan_generation == "heuristic_requests":
        raise ValueError("minimum_accuracy requires value model/layout guidance")
    if teacher is None:
        raise ValueError("collect_native requires a resolved teacher")
    settings = inference_settings or {}
    fixed_model_id = configured_fixed_model(settings)
    models, catalog = models_for_mode(models, catalog, settings, plan_generation)
    sampling_config = sampling_for_mode(
        sampling_config, plan_generation, fixed_model_id=fixed_model_id
    )
    ranker = (
        load_inference_model(checkpoint, inference_settings)
        if plan_generation != "heuristic_requests"
        else None
    )
    if plan_generation != "heuristic_requests" and not (
        ranker and ranker.value_available
    ):
        raise ValueError("heuristic_value feedback requires a trained value model")
    search = ParetoBeamSearch(
        ranker,
        SearchConfig(
            beam_width=candidates,
            max_expansions=max_expansions,
            max_seconds=search_seconds,
            training=phase == "measured",
            seed=seed,
            exploration_fraction=exploration_fraction if phase == "measured" else 0.0,
            plan_generation=plan_generation,
            fixed_model_id=fixed_model_id,
            workers=settings["search"].get("workers", 1),
            operator_priority_mode=operator_priority_mode,
        ),
    )
    initial_calibration = shared_profile(calibration_profile)
    policy = timeout_policy or TimeoutPolicy()
    descriptive_profile = {
        "online_request_profiles": copy.deepcopy(
            initial_calibration.get("online_request_profiles", {})
        )
    }
    complete = False
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started_at = time.time()
    timings = PhaseTimings(
        lambda phases: save(
            output / "timing.json",
            {
                "complete": False,
                "phase": phase,
                "started_at": started_at,
                "phases": phases,
            },
        )
    )
    label_timings = PhaseTimings()
    online_timing = {}
    own_history = history is None
    history = history if history is not None else CacheHistory(":memory:")
    try:
        with DuckDBEvaluator(
            build,
            teacher.transport_models(models),
            workload["sql"],
            database=database,
            output=output / "sql",
            timeout=policy.maximum,
            parallelism=execution_cpu_parallelism,
            max_concurrent_requests_per_model=max_concurrent_requests_per_model,
            enforce_model_operator_order=enforce_model_operator_order,
        ) as evaluator:
            query = normalize_explain(evaluator.run(explain=True))
            query.update(
                training_split=workload["split"],
                training_group=workload["group"],
                training_dataset=workload["dataset"],
                training_input_id=workload["id"],
            )
            timings.checkpoint("setup_and_explain")
            profile = {
                **copy.deepcopy(
                    {
                        k: v
                        for k, v in initial_calibration.items()
                        if k != "online_request_profiles"
                    }
                ),
                "models": catalog,
                "constraints": {"allowed_models": list(models)},
                "judge_model": teacher.model_id,
                "teacher": teacher.configuration(),
            }
            if minimum_accuracy is not None:
                profile["constraints"]["a_min"] = minimum_accuracy
            profile = profile_for_mode(
                profile, query, plan_generation, fixed_model_id=fixed_model_id
            )
            if prepared_statistics is not None:
                profile = merge_prepared_statistics(query, profile, prepared_statistics)
                profile["history"] = history.snapshot()
            problem = SearchProblem(query, profile)
            timings.checkpoint("search_problem_setup")
            batch = search_candidates(
                problem,
                search,
                build=build,
                models=models,
                database=database,
                history=history,
                sampling_config=sampling_config,
                teacher=teacher,
                catalog=catalog,
                output=output,
                workload_id=workload["id"],
                phase=phase,
                timings=timings,
                online_timing=online_timing,
                descriptive_profile=descriptive_profile,
                plan_generation=plan_generation,
                selection_policy=selection_policy,
                progress=progress,
                minimum_accuracy=minimum_accuracy,
                excluded_plan_signatures=excluded_plan_signatures,
                excluded_structure_signatures=excluded_structure_signatures,
                prepared_statistics=prepared_statistics,
            )
            frozen, states, result = batch.frozen, batch.states, batch.result
            save(output / "query_ir.json", query)
            save(output / "profile.json", frozen.profile)
            save(
                output / "calibration-update.json",
                {
                    "before": initial_calibration,
                    "after": shared_profile({**frozen.profile, **descriptive_profile}),
                },
            )
            save(output / "candidate_plans.json", [frozen.spec(s) for s in states])
            timings.checkpoint("sample_and_plan_artifact_save")
            metadata = record_metadata(
                workload,
                phase=phase,
                plan_generation=plan_generation,
                uses_value_model=batch.uses_value_model,
                cache_segment=cache_segment,
            )
            if phase == "measured":
                metadata.update(
                    feedback_protocol=ACCURACY_FEEDBACK_PROTOCOL
                    if selection_policy == "minimum_accuracy"
                    else FEEDBACK_PROTOCOL,
                    rollout_index=rollout_index,
                    searches_per_query=searches_per_query,
                    executions_per_query=searches_per_query,
                    selection_policy=selection_policy,
                    quality_scope="execution",
                    operator_priority_mode=search.config.priority_mode,
                )
                if minimum_accuracy is not None:
                    metadata.update(
                        minimum_accuracy=minimum_accuracy,
                        predicted_constraint_satisfied=result["rollout_selection"][
                            "constraint_satisfied"
                        ],
                        plan_signature=result["rollout_selection"]["plan_signature"],
                        duplicate_replaced=result["rollout_selection"][
                            "duplicate_replaced"
                        ],
                        repeat_unavoidable=result["rollout_selection"][
                            "repeat_unavoidable"
                        ],
                    )
            if phase == "estimated":
                records = estimated_records(
                    query,
                    frozen,
                    states,
                    history=history,
                    output=output,
                    timings=label_timings,
                    metadata=metadata,
                )
            else:
                records = [
                    measured_record(
                        evaluator,
                        query,
                        batch,
                        workload=workload,
                        history=history,
                        output=output,
                        policy=policy,
                        plan_generation=plan_generation,
                        metadata=metadata,
                        online_timing=online_timing,
                        descriptive_profile=descriptive_profile,
                        initial_calibration=initial_calibration,
                        progress=progress,
                    )
                ]
            timings.checkpoint(
                "estimated_labels" if phase == "estimated" else "measured_feedback"
            )
            payload = dataset(records)
            validate_dataset(payload)
            save(output / "experience.json", payload)
            save(
                output / "collection.json",
                {
                    "status": phase + "_complete",
                    "workload": workload,
                    "search": {k: v for k, v in result.items() if k != "states"},
                    "records": len(records),
                },
            )
            timings.checkpoint("feedback_artifact_save")
            complete = True
            return payload
    finally:
        if own_history:
            history.close()
        timings.checkpoint("unfinished_or_cleanup")
        save(
            output / "timing.json",
            {
                "complete": complete,
                "started_at": started_at,
                "phase": phase,
                "phases": timings.values,
                "label_details": label_timings.values,
                "basis": "phases are sequential wall/process CPU intervals; label_details are nested in estimated_labels; backend_timings span beam_search and freeze_and_replay; timings are not training latency labels",
            },
        )
