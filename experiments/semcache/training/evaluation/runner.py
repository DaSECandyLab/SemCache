"""Heldout selected text SQL evaluation after training; no labels enter replay."""

import csv
import hashlib
import json
import math
import time
from contextlib import ExitStack, nullcontext
from pathlib import Path

import sqlglot
from planning.backends.execution import DuckDBEvaluator
from planning.backends.sampling import NativeSamplingFactory
from planning.execution.cache_reset import clear_prompt_answers
from planning.execution.feedback import (
    freeze_problem,
    reset_serving_cache,
)
from planning.learning.devices import load_inference_model
from planning.observations.cache import CacheHistory
from planning.search.beam import ParetoBeamSearch, SearchConfig
from planning.settings.configuration import (
    activate_native,
    load_config,
    runtime_path,
    teacher_path,
)
from planning.settings.modes import (
    MODES,
    configured_fixed_model,
    models_for_mode,
    profile_for_mode,
    sampling_for_mode,
)
from planning.settings.teacher import TeacherModel, default_teacher_model, load_teacher
from planning.state import SearchProblem, digest
from reference_answers.sql.aggregate import capture_plan
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

from training.artifacts import save
from training.calibration.epoch import calibration_paths, validate_measured_epoch
from training.calibration.global_profile import GlobalCalibration
from training.collection.features import frozen_features, normalize_explain
from training.query_manifest import heldout_manifest, ordered_heldout_manifest
from training.references.evaluation import complete_quality
from training.references.evaluation import policy as reference_policy
from training.references.saved_evaluation import SavedEvaluationReferences


def original_query(sql):
    """Bind benchmark relation names to existing raw tables, leaving CTEs alone."""
    tree = sqlglot.parse_one(sql, read="duckdb")
    for scope in traverse_scope(tree):
        for alias, (_, source) in scope.selected_sources.items():
            if isinstance(source, exp.Table) and not source.name.startswith("raw_"):
                if not source.alias:
                    source.set("alias", exp.TableAlias(this=exp.to_identifier(alias)))
                source.set("this", exp.to_identifier("raw_" + source.name, quoted=True))
    return tree.sql(dialect="duckdb")


def freeze_evaluation_contract(output, checkpoint, settings):
    """Reuse evidence only for identical trained weights and evaluation inputs."""
    output = Path(output)
    contract = {
        "version": 1,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
        if checkpoint is not None
        else None,
        **settings,
    }
    path = output / "contract.json"
    if path.exists():
        if json.loads(path.read_text()) != contract:
            raise ValueError("Evaluation contract changed; use a new output directory")
    else:
        if output.exists() and any(output.glob("**/attempt-*/summary.json")):
            raise ValueError("Previous evaluation has no frozen reuse contract")
        output.mkdir(parents=True, exist_ok=True)
        save(path, contract)


def evaluate(
    run_dir,
    checkpoint,
    *,
    timeout=600,
    beam_width=8,
    search_seconds=600,
    max_expansions=4096,
    sampling_config=None,
    progress=print,
    update=None,
    teacher=None,
    reference_teacher=None,
    reference_teacher_config=None,
    reference_output=None,
    reference_evaluation_run=None,
    reference_aggregate_manifest=None,
    max_concurrent_requests_per_model=1,
    enforce_model_operator_order=None,
    execution_cpu_parallelism=1,
    settings=None,
    databases=None,
    rewrite_relations=True,
    output=None,
    preference=None,
    query_order=None,
    plan_generation="heuristic_requests",
    reference_reuse_only=False,
    candidate_models=None,
    fixed_layout=None,
    query_ids=None,
    execution_only=False,
    planning_only=False,
    cache_reset_scope="query",
    minimum_accuracy=None,
    layer_pruning="pareto",
    operator_priority_mode=None,
):
    if fixed_layout not in (None, "GOTDF") or (
        fixed_layout is not None and plan_generation != "heuristic_requests"
    ):
        raise ValueError(
            "Fixed GOTDF is the heuristic_requests contract; heuristic_value searches layouts"
        )
    if not isinstance(planning_only, bool):
        raise ValueError("planning_only must be a boolean")
    if planning_only and execution_only:
        raise ValueError("planning_only and execution_only are mutually exclusive")
    if layer_pruning not in ("pareto", "accuracy_time", "grouped"):
        raise ValueError("layer_pruning must be pareto, accuracy_time or grouped")
    if minimum_accuracy is not None and (
        isinstance(minimum_accuracy, bool)
        or not isinstance(minimum_accuracy, (int, float))
        or not 0 <= minimum_accuracy <= 1
        or not math.isfinite(minimum_accuracy)
    ):
        raise ValueError("Minimum accuracy must be a finite number in [0,1]")
    if plan_generation == "heuristic_requests" and minimum_accuracy is not None:
        raise ValueError(
            "minimum_accuracy requires heuristic_value; heuristic_requests does not predict accuracy"
        )
    run_dir = Path(run_dir)
    if databases is None:
        databases = json.loads((run_dir / "source-data.json").read_text())["databases"]
    databases = {name: Path(path) for name, path in databases.items()}
    settings = settings if settings is not None else load_config()
    activate_native(settings)
    if cache_reset_scope not in ("query", "evaluation"):
        raise ValueError("Cache reset scope must be query or evaluation")
    preference = preference or settings["inference"]["preference"]
    if preference not in ("fastest", "accurate"):
        raise ValueError("Evaluation preference must be fastest or accurate")
    if plan_generation not in MODES:
        raise ValueError("Unknown evaluation plan generation policy")
    if operator_priority_mode is not None and (
        plan_generation != "heuristic_value"
        or operator_priority_mode not in ("heuristic_requests", "heuristic_value")
    ):
        raise ValueError("Operator-priority overrides require heuristic_value guidance")
    request_guided = plan_generation == "heuristic_requests"
    if request_guided:
        preference, checkpoint = "fastest", None
        fixed_layout = "GOTDF"
    fixed_model_id = configured_fixed_model(settings)
    sampling_config = sampling_for_mode(
        sampling_config, plan_generation, fixed_model_id=fixed_model_id
    )
    if enforce_model_operator_order is None:
        enforce_model_operator_order = settings["execution"][
            "enforce_model_operator_order"
        ]
    build = runtime_path(settings, "build")
    models = json.loads((run_dir / "models.json").read_text())
    catalog = json.loads((run_dir / "model-catalog.json").read_text())
    manifest = heldout_manifest()
    audit = json.loads((run_dir / "split-audit.json").read_text())
    if audit.get("heldout_manifest") != manifest:
        raise ValueError("Heldout SQL changed after workload generation")
    if query_ids is not None:
        if (
            not isinstance(query_ids, (list, tuple))
            or not query_ids
            or any(not isinstance(key, str) for key in query_ids)
            or len(set(query_ids)) != len(query_ids)
            or not set(query_ids) <= {item["id"] for item in manifest}
        ):
            raise ValueError(
                "Query IDs must be a nonempty unique subset of the frozen workload"
            )
        selected = set(query_ids)
        manifest = [item for item in manifest if item["id"] in selected]
    order = query_order or [item["id"] for item in ordered_heldout_manifest(manifest)]
    if len(order) != len(manifest) or set(order) != {item["id"] for item in manifest}:
        raise ValueError("Evaluation query order must cover the exact frozen query set")
    reference_evaluation_run = reference_evaluation_run or settings["inference"].get(
        "reference_evaluation_run"
    )
    reference_aggregate_manifest = reference_aggregate_manifest or settings[
        "inference"
    ].get("reference_aggregate_manifest")
    saved_references = (
        SavedEvaluationReferences(
            reference_evaluation_run,
            manifest,
            databases,
            aggregate_manifest=reference_aggregate_manifest,
        )
        if reference_evaluation_run and not (planning_only or execution_only)
        else None
    )
    if saved_references is not None:
        reference_reuse_only = True
        # The saved run pins the original relation bindings as well as SQL bytes.
        rewrite_relations = False
    ranker = None if request_guided else load_inference_model(checkpoint, settings)
    if not request_guided and not (ranker and ranker.value_available):
        raise ValueError("Evaluation requires a trained value model")
    if teacher is None:
        teacher_file = teacher_path(settings)
        teacher = load_teacher(
            default_teacher_model(teacher_file), models, teacher_file
        )
    if reference_teacher is None and not planning_only and saved_references is None:
        reference_teacher = (
            load_teacher(
                default_teacher_model(reference_teacher_config),
                models,
                reference_teacher_config,
            )
            if reference_teacher_config is not None
            else teacher
        )
    reference_teacher_config = reference_teacher_config or teacher_path(settings)
    quality_policy = None
    if not planning_only:
        quality_policy = (
            saved_references.policy
            if saved_references is not None
            else reference_policy(reference_teacher, settings)
        )
    validate_measured_epoch(run_dir, models, teacher)
    calibration_path = calibration_paths(run_dir)["global"]
    prior_teacher = teacher
    if calibration_path.exists():
        prior_teacher = TeacherModel(
            **json.loads(calibration_path.read_text())["contract"]["teacher"]
        )
    profile = GlobalCalibration(calibration_path, models, prior_teacher).snapshot()
    # Validate the saved full calibration contract before restricting candidates.
    # Existing model statistics remain valid when using a subset of that pool.
    if candidate_models is not None:
        if (
            not isinstance(candidate_models, (list, tuple))
            or not candidate_models
            or any(not isinstance(name, str) for name in candidate_models)
            or len(set(candidate_models)) != len(candidate_models)
            or any(name not in models for name in candidate_models)
        ):
            raise ValueError(
                "Candidate models must be a nonempty unique subset of the saved model pool"
            )
        models = {name: models[name] for name in candidate_models}
        catalog = {name: catalog[name] for name in candidate_models}
    models, catalog = models_for_mode(models, catalog, settings, plan_generation)
    teacher.apply_profile(profile)
    constraints: dict = {"allowed_models": list(models)}
    if minimum_accuracy is not None:
        constraints["a_min"] = minimum_accuracy
    profile.update(models=catalog, constraints=constraints)
    transport_models = teacher.transport_models(models)
    output = Path(output) if output is not None else run_dir / "evaluation"
    reference_output = (
        Path(reference_output)
        if reference_output is not None
        else output.parent / "references"
    )
    freeze_evaluation_contract(
        output,
        checkpoint,
        {
            "models": models,
            "heldout_manifest": manifest,
            "query_order": order,
            "training_profile": profile,
            "source_databases": {name: str(path) for name, path in databases.items()},
            "input_scope": "original tables, all rows",
            "timeout": timeout,
            "beam_width": beam_width,
            "search_seconds": search_seconds,
            "max_expansions": max_expansions,
            "sampling_config": sampling_config,
            "execution_cpu_parallelism": execution_cpu_parallelism,
            "max_concurrent_requests_per_model": max_concurrent_requests_per_model,
            "enforce_model_operator_order": enforce_model_operator_order,
            "preference": preference,
            "layer_pruning": layer_pruning,
            "rewrite_relations": rewrite_relations,
            "cache_policy": (
                "cold_before_evaluation_then_shared_across_queries"
                if cache_reset_scope == "evaluation"
                else "cold_before_each_query_then_search_execution_continuous"
            ),
            "reference_policy": quality_policy,
            "plan_generation": plan_generation,
            **(
                {"operator_priority_mode": operator_priority_mode}
                if operator_priority_mode is not None
                else {}
            ),
            **({"fixed_layout": fixed_layout} if fixed_layout is not None else {}),
            **({"reference_reuse_only": True} if reference_reuse_only else {}),
            **({"execution_only": True} if execution_only else {}),
            **({"planning_only": True} if planning_only else {}),
            **(
                {"minimum_accuracy": minimum_accuracy}
                if minimum_accuracy is not None
                else {}
            ),
        },
    )
    terminal_statuses = {"planned", "infeasible"} if planning_only else {"completed"}
    completed = {}
    if output.exists():
        for path in output.glob("**/attempt-*/summary.json"):
            entry = json.loads(path.read_text())
            if entry.get("status") in terminal_statuses:
                completed[entry["query"]] = entry
    results = [
        row
        for row in completed.values()
        if planning_only
        or execution_only
        or (
            row.get("reference_policy") == quality_policy
            and row.get("reference_status") == "completed"
            and row.get("accuracy") is not None
        )
    ]
    # Score each new successful execution before publishing and moving on.
    # Repair resumed or failed reference/F1 work after pending executions,
    # preserving the frozen relative order within both remaining queues.
    pending_execution = {key for key in order if key not in completed}
    work = [(key, False) for key in order if key in pending_execution]
    if not (planning_only or execution_only):
        work.extend((key, True) for key in order)
    if update:
        update(
            pending_execution_queries=[key for key in order if key in pending_execution]
        )

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
                            search = ParetoBeamSearch(
                                ranker,
                                SearchConfig.configured(
                                    settings,
                                    plan_generation,
                                    beam_width=beam_width,
                                    max_seconds=search_seconds,
                                    max_expansions=max_expansions,
                                    preference=preference,
                                    layer_pruning=layer_pruning,
                                    operator_priority_mode=operator_priority_mode,
                                ),
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
