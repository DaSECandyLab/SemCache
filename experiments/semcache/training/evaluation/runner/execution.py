"""Heldout selected text SQL evaluation after training; no labels enter replay."""

import csv
import time
from contextlib import ExitStack, nullcontext
from pathlib import Path

from planning.backends.execution import DuckDBEvaluator
from planning.backends.sampling import NativeSamplingFactory
from planning.execution.cache.reset import clear_prompt_answers
from planning.execution.feedback import (
    freeze_problem,
    reset_serving_cache,
)
from planning.observations.cache import CacheHistory
from planning.search.beam import ParetoBeamSearch, SearchConfig
from planning.settings.modes import (
    profile_for_mode,
)
from planning.state import SearchProblem, digest
from reference_answers.sql.aggregate import capture_plan
from training.artifacts import save
from training.collection.features import frozen_features, normalize_explain
from training.references.evaluation import complete_quality

from .setup import original_query


def run_evaluation(ctx):
    settings = ctx.settings
    search_seconds = ctx.search_seconds
    max_expansions = ctx.max_expansions
    beam_width = ctx.beam_width
    timeout = ctx.timeout
    execution_cpu_parallelism = ctx.execution_cpu_parallelism
    max_concurrent_requests_per_model = ctx.max_concurrent_requests_per_model
    enforce_model_operator_order = ctx.enforce_model_operator_order
    models = ctx.models
    databases = ctx.databases
    profile = ctx.profile
    manifest = ctx.manifest
    order = ctx.order
    results = ctx.results
    completed = ctx.completed
    terminal_statuses = ctx.terminal_statuses
    pending_execution = ctx.pending_execution
    work = ctx.work
    minimum_accuracy = ctx.minimum_accuracy
    planning_only = ctx.planning_only
    execution_only = ctx.execution_only
    cache_reset_scope = ctx.cache_reset_scope
    progress = ctx.progress
    update = ctx.update
    build = ctx.build
    checkpoint = ctx.checkpoint
    fixed_model_id = ctx.fixed_model_id
    teacher = ctx.teacher
    catalog = ctx.catalog
    output = ctx.output
    fixed_layout = ctx.fixed_layout
    layer_pruning = ctx.layer_pruning
    operator_priority_mode = ctx.operator_priority_mode
    plan_generation = ctx.plan_generation
    preference = ctx.preference
    quality_policy = ctx.quality_policy
    ranker = ctx.ranker
    reference_output = ctx.reference_output
    reference_reuse_only = ctx.reference_reuse_only
    reference_teacher_config = ctx.reference_teacher_config
    reference_teacher = ctx.reference_teacher
    request_guided = ctx.request_guided
    rewrite_relations = ctx.rewrite_relations
    sampling_config = ctx.sampling_config
    saved_references = ctx.saved_references
    transport_models = ctx.transport_models
    def save_results():
        save(
            output / "results.json",
            {
                "queries": results,
                "total_queries": len(manifest),
                "completed": sum(row["status"] in terminal_statuses for row in results),
                "teacher_agreement_available": sum(
                    row.get("accuracy") is not None for row in results
                ),
                "training_feedback_used": False,
                "input_scope": "original tables, all rows",
                **({"planning_only": True} if planning_only else {}),
            },
        )

    def publish_result(directory, entry):
        pending_execution.discard(entry["query"])
        if minimum_accuracy is not None:
            accuracy = entry.get("accuracy")
            entry["observed_constraint_satisfied"] = (
                accuracy >= minimum_accuracy if accuracy is not None else None
            )
        save(directory / "summary.json", entry)
        progress(
            f"Heldout {entry['query']}: {entry['status']}, full original input, seconds={entry.get('execution_seconds')}",
            flush=True,
        )
        results[:] = [row for row in results if row["query"] != entry["query"]]
        results.append(entry)
        save_results()
        if update:
            update(
                pending_execution_queries=[
                    key for key in order if key in pending_execution
                ],
                completed_test_queries=len(results),
                total_test_queries=len(manifest),
                last_result=entry,
            )

    by_id = {item["id"]: item for item in manifest}
    evaluation_cache_reset = None
    with ExitStack() as evaluation_context:
        shared_history = (
            evaluation_context.enter_context(CacheHistory(":memory:"))
            if cache_reset_scope == "evaluation"
            else None
        )
        for query_id, reference_only in work:
            item = by_id[query_id]
            key = item["id"]
            if reference_only and key not in completed:
                continue
            if any(
                row["query"] == key
                and row.get("reference_status") == "completed"
                and row.get("accuracy") is not None
                for row in results
            ):
                continue
            path = Path(item["path"])
            dataset = {"movies": "movies_nirvana"}.get(
                path.parent.name, path.parent.name
            )
            database = databases[dataset]
            sql = (
                original_query(path.read_text())
                if rewrite_relations
                else path.read_text()
            )
            if update:
                update(
                    current_query=key,
                    completed_test_queries=len(results),
                    total_test_queries=len(manifest),
                )

            def stage(name, **values):
                entry.update(stage=name, stage_started_at=time.time(), **values)
                save(directory / "progress.json", entry)
                if update:
                    update(current_stage=name)

            def reset_caches(phase):
                reset = {"phase": phase, "started_at": time.time()}
                started = time.monotonic()
                try:
                    reset["model_cache"] = reset_serving_cache(models)
                    reset["prompt_cache"] = clear_prompt_answers()
                    reset["status"] = "completed"
                except (OSError, ValueError) as error:
                    reset.update(status="failed", error=str(error))
                    raise
                finally:
                    reset["seconds"] = time.monotonic() - started
                    target = output if cache_reset_scope == "evaluation" else directory
                    save(target / f"{phase}-cache-reset.json", reset)
                return reset

            if key in completed:
                # Preserve successful candidate measurements. Repair only the
                # reference/F1 when an API failed or the teacher policy changed.
                entry = dict(completed[key])
                directory = Path(entry["result_artifact"]).parent.parent
                started = time.monotonic()
                original_seconds = entry.get("process_seconds", 0)

                complete_quality(
                    entry,
                    directory,
                    output=reference_output,
                    key=key,
                    sql=sql,
                    database=database,
                    teacher=reference_teacher,
                    teacher_config=reference_teacher_config,
                    reuse_only=reference_reuse_only,
                    saved_references=saved_references,
                    build=build,
                    settings=settings,
                    timeout=None,
                    stage=stage,
                )
                entry["process_seconds"] = original_seconds + time.monotonic() - started
                entry["candidate_measurement_reused"] = True
                publish_result(directory, entry)
                continue
            attempt = 1
            while (output / key / f"attempt-{attempt}").exists():
                attempt += 1
            directory = output / key / f"attempt-{attempt}"
            directory.mkdir(parents=True)
            entry = {
                "query": key,
                "sql_sha256": item["sql_sha256"],
                "checkpoint": str(checkpoint) if checkpoint is not None else None,
                "source_database": str(database),
                "input_scope": "original tables, all rows",
                "models": list(models),
                "quality_basis": (
                    "Beam completion prediction; no measured query result"
                    if planning_only
                    else "same-input fixed-GOTDF heuristic-teacher agreement; not human ground truth"
                ),
                "preference": preference,
                "execution_timeout_seconds": timeout,
                "plan_generation": plan_generation,
                **({"fixed_layout": fixed_layout} if fixed_layout is not None else {}),
                **(
                    {
                        "planning_only": True,
                        "execution_status": "not_run",
                        "execution_seconds": None,
                        "accuracy": None,
                        "reference_status": "skipped_planning_only",
                        "predicted_execution_seconds": None,
                        "predicted_f1": None,
                        "plan_artifact": None,
                    }
                    if planning_only
                    else {}
                ),
                **(
                    {
                        "minimum_accuracy": minimum_accuracy,
                        "constraint_satisfied": None,
                        "search_reason": None,
                        "infeasible_reason": None,
                    }
                    if minimum_accuracy is not None
                    else {}
                ),
            }
            start = time.monotonic()
            entry["started_at"] = time.time()

            try:
                stage("准备中")
                (directory / "query.sql").write_text(sql)
                # Reset once for the requested scope. The evaluation scope keeps
                # prompt answers and serving KV warm across every query/phase.
                if cache_reset_scope == "query" or evaluation_cache_reset is None:
                    reset = reset_caches(
                        "before-evaluation"
                        if cache_reset_scope == "evaluation"
                        else "before-search"
                    )
                    entry["prompt_cache_reset"] = reset["prompt_cache"]
                    if cache_reset_scope == "evaluation":
                        evaluation_cache_reset = reset
                    else:
                        entry["before_search_cache_reset"] = reset
                if cache_reset_scope == "evaluation":
                    entry["evaluation_cache_reset"] = evaluation_cache_reset
                with (
                    (
                        nullcontext(shared_history)
                        if shared_history is not None
                        else CacheHistory(":memory:")
                    ) as history,
                    DuckDBEvaluator(
                        build,
                        transport_models,
                        sql,
                        database=database,
                        output=directory / "sql",
                        history_sink=history,
                        timeout=timeout,
                        parallelism=execution_cpu_parallelism,
                        max_concurrent_requests_per_model=max_concurrent_requests_per_model,
                        enforce_model_operator_order=enforce_model_operator_order,
                    ) as evaluator,
                ):
                    query = normalize_explain(evaluator.run(explain=True))
                    save(directory / "query_ir.json", query)
                    entry["query_ir_artifact"] = str(
                        (directory / "query_ir.json").resolve()
                    )
                    query_profile = profile_for_mode(
                        profile, query, plan_generation, fixed_model_id=fixed_model_id
                    )
                    problem = SearchProblem(query, query_profile)
                    factory = NativeSamplingFactory(
                        build,
                        models,
                        database=database,
                        history=history,
                        config=sampling_config,
                        teacher=teacher,
                        catalog=catalog,
                    )
                    try:
                        with factory(problem) as sampler:
                            problem.sampler = sampler
                            config = SearchConfig.configured(
                                settings,
                                plan_generation,
                                beam_width=beam_width,
                                max_seconds=search_seconds,
                                max_expansions=max_expansions,
                                preference=preference,
                                operator_priority_mode=operator_priority_mode,
                                **(
                                    {"layer_pruning": layer_pruning}
                                    if layer_pruning is not None
                                    else {}
                                ),
                            )
                            if layer_pruning is None:
                                config = config.for_problem(problem)
                            search = ParetoBeamSearch(
                                ranker,
                                config,
                            )
                            stage("搜索中")
                            search_start = time.monotonic()
                            search_cpu = time.process_time()
                            search_start_ns = time.monotonic_ns()
                            result = search.run(problem)
                            search_end_ns = time.monotonic_ns()
                            entry.update(
                                planning_seconds=time.monotonic() - search_start,
                                beam_cpu_seconds=time.process_time() - search_cpu,
                            )
                            save(
                                directory / "search.json",
                                {k: v for k, v in result.items() if k != "states"},
                            )
                            save(directory / "profile.json", problem.profile)
                            entry.update(
                                sampling_calls=sampler.calls,
                                candidates=len(result["complete"]),
                                expansions=result["expansions"],
                            )
                            latency = sampler.request_latency(
                                start_ns=search_start_ns, end_ns=search_end_ns
                            )
                            entry["beam_llm_wall_seconds"] = min(
                                entry["planning_seconds"], latency["wall_seconds"]
                            )
                            entry["beam_non_llm_wall_seconds"] = (
                                entry["planning_seconds"]
                                - entry["beam_llm_wall_seconds"]
                            )
                            entry["sampling_http_requests_by_model"] = latency[
                                "http_requests_by_model"
                            ]
                            save(
                                directory / "sampling.json",
                                {
                                    "calls": sampler.calls,
                                    "backend_timings": sampler.native_sampler.timings(),
                                    "request_latency": latency,
                                    "http_requests_by_model": latency[
                                        "http_requests_by_model"
                                    ],
                                },
                            )
                            entry.update(
                                search_artifact=str(
                                    (directory / "search.json").resolve()
                                ),
                                sampling_artifact=str(
                                    (directory / "sampling.json").resolve()
                                ),
                            )
                            if planning_only or minimum_accuracy is not None:
                                entry.update(
                                    constraint_satisfied=result.get(
                                        "constraint_satisfied"
                                    ),
                                    search_reason=result.get("reason"),
                                    accuracy_fallback=result.get(
                                        "accuracy_fallback", False
                                    ),
                                )
                            feasible = bool(result["states"]) and (
                                request_guided
                                or minimum_accuracy is None
                                or result.get("constraint_satisfied") is True
                                or result.get("accuracy_fallback") is True
                            )
                            if not feasible:
                                reason = result.get("reason") or (
                                    "Search did not produce a candidate satisfying minimum accuracy"
                                )
                                entry["infeasible_reason"] = reason
                                if not planning_only:
                                    raise RuntimeError(reason)
                                entry.update(status="infeasible", reason=reason)
                                stage("本次搜索未找到达标计划")
                            else:
                                selected = search.choose(problem, result, preference)
                                chosen = next(
                                    state
                                    for state in result["states"]
                                    if problem.spec(state) == selected
                                )
                                assessment = next(
                                    (
                                        row
                                        for row in result.get("assessments", [])
                                        if row.get("plan_key") == digest(chosen.history)
                                    ),
                                    {},
                                )
                                entry.update(
                                    estimated_requests=assessment.get(
                                        "estimated_requests"
                                    ),
                                    objective_units=result.get("objective_units"),
                                    predicted_execution_seconds=assessment.get(
                                        "time_prediction"
                                    ),
                                    predicted_f1=assessment.get("accuracy"),
                                )
                                frozen, states = freeze_problem(problem, [chosen])
                                spec = frozen.spec(states[0])
                                stage("保存计划")
                    finally:
                        factory.close()
                    if entry.get("status") != "infeasible":
                        save(directory / "plan_spec.json", spec)
                        save(directory / "profile.json", frozen.profile)
                        before = history.snapshot()
                        _, _, features, feature_audit = frozen_features(
                            query, frozen.profile, spec, before
                        )
                        save(
                            directory / "features.json",
                            {"audit": feature_audit, "trajectory": features},
                        )
                        entry["plan_artifact"] = str(
                            (directory / "plan_spec.json").resolve()
                        )
                        if planning_only:
                            entry["status"] = "planned"
                            stage("计划已保存")
                        else:
                            stage("执行中")
                            actual = evaluator.run(
                                spec,
                                source_query=query,
                                allow_timeout=True,
                                capture_aggregate_input=capture_plan(sql) is not None,
                                include_rows=False,
                            )
                            result_rows = None
                            if actual["status"] == "completed":
                                with (Path(actual["artifact"]) / "result.csv").open(
                                    newline=""
                                ) as stream:
                                    result_rows = max(
                                        0, sum(1 for _ in csv.reader(stream)) - 1
                                    )
                            entry.update(
                                status=actual["status"],
                                execution_seconds=actual["seconds"],
                                requests=sum(
                                    e["event"] == "submit" for e in actual["events"]
                                ),
                                result_rows=result_rows,
                                time_lower_bound=actual.get("time_lower_bound"),
                                result_artifact=actual["artifact"],
                            )
                            # Full traces and CSV already live in the query artifact.
                            # Do not retain their payload while running the reference.
                            actual.pop("csv", None)
                            actual.pop("events", None)
                            if actual["status"] == "completed":
                                entry.update(
                                    reference_status="skipped_execution_only"
                                    if execution_only
                                    else "pending",
                                    accuracy=None,
                                    reference_policy=quality_policy,
                                )
                                stage("执行完成" if execution_only else "待参考")
                if entry.get("status") == "completed" and not (
                    planning_only or execution_only
                ):
                    complete_quality(
                        entry,
                        directory,
                        output=reference_output,
                        key=key,
                        sql=sql,
                        database=database,
                        teacher=reference_teacher,
                        teacher_config=reference_teacher_config,
                        reuse_only=reference_reuse_only,
                        saved_references=saved_references,
                        build=build,
                        settings=settings,
                        timeout=None,
                        stage=stage,
                    )
            except KeyboardInterrupt:
                # An interrupted teacher/reference must not invalidate a measured
                # candidate that already completed successfully.
                if entry.get("status") == "completed":
                    entry.update(
                        reference_status="interrupted",
                        reference_error="Reference generation or scoring interrupted",
                    )
                else:
                    entry["status"] = "interrupted"
                entry.update(stage="已中断", process_seconds=time.monotonic() - start)
                save(directory / "summary.json", entry)
                raise
            except (
                OSError,
                ValueError,
                KeyError,
                RuntimeError,
                TimeoutError,
            ) as error:
                if entry.get("status") == "completed":
                    entry.update(
                        reference_status="failed",
                        reference_error=str(error),
                        accuracy=None,
                    )
                else:
                    entry.update(status="failed", reason=str(error))
            entry["process_seconds"] = time.monotonic() - start
            if entry.get("status") in terminal_statuses:
                completed[key] = entry
            publish_result(directory, entry)
    save_results()
    return results
