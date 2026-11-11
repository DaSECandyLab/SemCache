"""§4.4: calibrated estimate pretraining, then iterative native-SQL feedback."""

import fcntl
import gc
import json
import os
from datetime import datetime
from pathlib import Path

from planning.backends.execution import DuckDBEvaluator
from planning.execution.timeouts import TimeoutPolicy
from planning.learning.features.schema import FEATURE_VERSION
from planning.learning.training.data import dataset, validate_dataset
from planning.observations.cache import CacheHistory
from planning.search.beam import SearchConfig
from planning.settings.configuration import ROOT as SEMCACHE_ROOT
from planning.settings.configuration import (
    activate_native,
    load_config,
    load_sampling_config,
    runtime_path,
)
from planning.settings.model_config import discover_models, unavailable_models
from planning.settings.parallelism import configure_cpu_resources
from planning.settings.teacher import TeacherModel, default_teacher_model, load_teacher

from training.artifacts import read, save
from training.calibration.calibrate import calibrate_models
from training.calibration.epoch import (
    calibration_paths,
    epoch_identity,
    validate_measured_epoch,
)
from training.calibration.global_profile import GlobalCalibration
from training.collection.collector import (
    COLLECTION_PROTOCOL,
    collect_native,
    reset_context,
)
from training.collection.features import normalize_explain
from training.collection.feedback import collect_feedback_query, rollout_entries
from training.collection.policy import (
    FEEDBACK_COLLECTION_ORDER,
    FEEDBACK_PROTOCOL,
    PLAN_GENERATION_POLICIES,
    ordered_workloads,
)
from training.collection.recovery import collect_with_retry, scientific_protocol
from training.collection.resume import begin_resume_segment, validate_measured_prefix
from training.datasets import publish_model, select_manifest, validate_scope
from training.fitting import fit
from training.query_manifest import (
    count_operators,
    frozen_workload_versions,
    heldout_manifest,
    load_manifest,
    validate_round_workload,
    workload_counts,
)
from training.references.answers import TrainingAnswers
from training.references.scoring import (
    PRE_AGGREGATE_ACCURACY_METRIC,
    RESULT_ACCURACY_METRIC,
    finalize_measured_accuracy,
)
from training.source_data import inspect_corpora
from training.workloads.checks import audit_workloads, validate_native_budget
from training.workloads.provenance import load_template_rewrites


def candidate_models(args, run_dir):
    """Resume from saved model metadata without querying the serving endpoints."""
    models = run_dir / "models.json"
    if models.is_file():
        return read(models), read(run_dir / "model-catalog.json")
    if getattr(args, "model_config", None):
        from planning.settings.model_config import discover_configured_models

        return discover_configured_models(args.model_config)
    return discover_models(args.endpoints, max_output_tokens=args.max_output_tokens)


def calibration_progress_writer(path):
    """Publish live calibration progress so 进度.md can show partial state."""
    started = datetime.now().astimezone()

    def write(entry):
        payload = dict(entry)
        payload["started_at"] = started.isoformat()
        payload["updated_at"] = datetime.now().astimezone().isoformat()
        payload["elapsed_seconds"] = round(
            (datetime.now().astimezone() - started).total_seconds(), 1
        )
        temporary = path.with_suffix(".tmp")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(payload, ensure_ascii=False))
        temporary.replace(path)

    return write


def run(args):
    configured_settings = load_config(getattr(args, "config", None))
    settings = {
        **configured_settings,
        "response_cache": {
            **configured_settings["response_cache"],
            "directory": None,
            "verify_on_second_use": False,
        },
    }
    activate_native(settings)
    selected_dataset = settings["training"].get("dataset")
    configured_counts = workload_counts(args.queries_per_dataset)
    canonical_path = SEMCACHE_ROOT / settings["training"]["workload_manifest"]
    canonical_manifest = load_manifest(canonical_path)
    collect_only = getattr(
        args, "collect_only", settings["training"].get("collect_only", False)
    )
    if collect_only and (
        args.pretrain_rounds != 1 or args.rounds != 0 or args.evaluate_selected
    ):
        raise ValueError(
            "Collection-only mode requires pretrain_rounds=1, rounds=0 and no evaluation"
        )
    configured_build = runtime_path(settings, "build")
    assert isinstance(configured_build, Path)
    build = Path(getattr(args, "build", configured_build)).resolve()
    run_dir = args.run.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    shared_lock = SEMCACHE_ROOT.parent / "experiments/results/semcache/model_training/.collection.lock"
    shared_lock.parent.mkdir(parents=True, exist_ok=True)
    with (
        shared_lock.open("a") as global_lock,
        (run_dir / "pipeline.lock").open("a") as lock,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (run_dir / "pipeline.pid").write_text(str(os.getpid()) + "\n")
        previous_state = (
            read(run_dir / "pipeline-status.json")
            if (run_dir / "pipeline-status.json").exists()
            else {}
        )
        state = {
            "status": "running",
            "phase": "initializing",
            "pid": os.getpid(),
            "completed_queries": 0,
        }

        def update(**values):
            state.update(values, updated_at=datetime.now().astimezone().isoformat())
            save(run_dir / "pipeline-status.json", state)

        try:
            update(phase="waiting_shared_resources", current_query=None)
            fcntl.flock(global_lock, fcntl.LOCK_EX)
            update(phase="initializing")
            save(
                run_dir / "cpu-resources.json",
                configure_cpu_resources(settings, configure_torch=True),
            )
            # Fitting a finished bootstrap uses its frozen observations only;
            # no model service, teacher, SQL binding or sampling is needed.
            bootstrap = run_dir / "estimated/round-1"
            if (
                not collect_only
                and args.pretrain_rounds == 1
                and args.rounds == 0
                and not args.evaluate_selected
                and (bootstrap / "experience.json").is_file()
            ):
                completed = read(bootstrap / "experience.json")
                validate_dataset(completed)
                validate_round_workload(completed, read(bootstrap / "workloads.json"))
                checkpoint_out = bootstrap / "model/value.pt"
                update(
                    phase="value_update",
                    round=1,
                    global_round=1,
                    total_rounds=1,
                    total_queries=len(read(bootstrap / "workloads.json")["workloads"]),
                    records=len(completed["records"]),
                    current_query=None,
                    model_training="running",
                )
                if not checkpoint_out.exists():
                    fit(
                        completed["records"],
                        bootstrap / "model",
                        epochs=args.epochs,
                        seed=args.seed,
                        source="estimated",
                        settings=settings,
                    )
                update(
                    status="complete",
                    phase="complete",
                    checkpoint=str(checkpoint_out),
                    model_training="complete",
                )
                return
            SearchConfig(args.candidates, args.max_expansions, args.search_seconds)
            models, catalog = candidate_models(args, run_dir)
            teacher_config = getattr(args, "teacher_config", None)
            teacher_name = getattr(
                args, "teacher_model", None
            ) or default_teacher_model(teacher_config)
            teacher = load_teacher(teacher_name, models, teacher_config)
            teacher_epoch = validate_measured_epoch(run_dir, models, teacher)
            teacher_epoch_id = epoch_identity(teacher_epoch)
            if len(models) < 2:
                raise ValueError(
                    "Training requires at least two distinct available models"
                )
            policy = TimeoutPolicy(
                args.timeout, args.minimum_timeout, args.timeout_slack
            )
            estimated_policy = TimeoutPolicy(
                settings["training"]["estimated_timeout"],
                min(args.minimum_timeout, settings["training"]["estimated_timeout"]),
                args.timeout_slack,
            )
            sampling_path = getattr(args, "sampling_config", None)
            sampling_config = load_sampling_config(sampling_path, settings)
            if not isinstance(sampling_config, dict):
                raise TypeError("Sampling configuration must be a JSON object")
            from planning.settings.parallelism import sampling_parallelism

            resolved_sampling = sampling_parallelism(
                sampling_config, override=getattr(args, "parallelism", None)
            )
            if getattr(args, "parallelism", None) is not None:
                sampling_config["cpu_parallelism"] = resolved_sampling["parallelism"]
            if args.pretrain_rounds < 1 or args.candidates < 1 or args.rounds < 0:
                raise ValueError(
                    "Training requires at least one estimated bootstrap pass, positive candidate budgets, "
                    "and nonnegative measured rounds"
                )
            if args.rounds == 0 and args.evaluate_selected:
                raise ValueError(
                    "--evaluate-selected requires measured rounds; "
                    "--rounds 0 runs estimated pretraining only"
                )
            feedback_cycle = settings["training"].get("feedback_cycle")
            if not isinstance(feedback_cycle, list) or not feedback_cycle:
                raise ValueError("training.feedback_cycle must be a nonempty list")
            if any(
                not isinstance(item, dict)
                or item.get("plan_generation") not in PLAN_GENERATION_POLICIES
                or item.get("query_order") not in ("forward", "reverse", "shuffled")
                for item in feedback_cycle
            ):
                raise ValueError(
                    "Each feedback_cycle item needs plan_generation="
                    "heuristic_requests/heuristic_value/value "
                    "and query_order=forward/reverse/shuffled"
                )
            workload_version, pretraining_workload_version = frozen_workload_versions(
                run_dir, canonical_manifest["version"]
            )
            protocol = {
                "version": COLLECTION_PROTOCOL,
                "build": str(build),
                "configuration": settings,
                "feature_version": FEATURE_VERSION,
                "workload_version": workload_version,
                "pretraining_workload_version": pretraining_workload_version,
                "models": models,
                "teacher": teacher.configuration(),
                "encoder": "GAT-PE",
                "objective": "balsa_monte_carlo_value_learning",
                "return_target": "unconditioned_mean_completion_pair",
                "input_scope": "all rows of each configured source database; no runtime input restriction",
                "simulator_bootstrap": "estimated/round-1",
                "value_guided_estimated_rounds": max(0, args.pretrain_rounds - 1),
                "pretrain_rounds": args.pretrain_rounds,
                "collect_only": collect_only,
                "rounds": args.rounds,
                "feedback_cycle": feedback_cycle,
                "execution_candidates": FEEDBACK_PROTOCOL
                if args.rounds
                else "all_beam_completions",
                "feedback_protocol": FEEDBACK_PROTOCOL,
                "searches_per_query": 2,
                "executions_per_query": 2,
                "measured_search_policy": PLAN_GENERATION_POLICIES,
                "round_cache_reset": "kv_history_prompt_answers_v1",
                "beam_width": args.candidates,
                "queries_per_dataset": args.queries_per_dataset,
                "repeats": args.repeats,
                "measured_repeats": 1,
                "seed": args.seed,
                "epochs": args.epochs,
                "timeout": vars(policy),
                "statistics": "same sampled / DuckDB estimates as online inference",
                "completion_scoring": "once after first observation; no final rescoring",
                "sketch_policy": "adaptive-v5: prefix-diversity coverage plus empirical reuse error; shared depth grid",
                "sampling_config": sampling_config,
                "sampling_request_unit": "actual LLM HTTP send attempt; cache reuse costs zero",
                "search_seconds": args.search_seconds,
                "max_expansions": args.max_expansions,
                "evaluate_selected": args.evaluate_selected,
                "quality": "read growing frozen Sema teacher answers; system_comparison.accuracy row F1; wait for pending answers before updating",
                "accuracy_metrics": {
                    "final_output": RESULT_ACCURACY_METRIC,
                    "pre_aggregate_rows": PRE_AGGREGATE_ACCURACY_METRIC,
                },
                "censor": "terminal-only lower-bound log regression",
                "collection_order": FEEDBACK_COLLECTION_ORDER,
                "calibration": "once mixed train cohort; isolated atomic cost priors plus separate descriptive online response/throughput statistics; validation read-only",
                "test_queries_excluded": True,
            }
            if selected_dataset is not None:
                protocol["dataset"] = selected_dataset
                protocol["calibration"] = "isolated dataset training cohort"
            protocol_path = run_dir / "protocol.json"
            if protocol_path.exists():
                if scientific_protocol(read(protocol_path)) != scientific_protocol(
                    protocol
                ):
                    raise ValueError(
                        "Frozen run protocol differs; use a new run directory"
                    )
            else:
                save(protocol_path, protocol)
            retry_policy = settings["training"].get("network_retry", {})
            save(run_dir / "network-retry-policy.json", retry_policy)
            save(run_dir / "models.json", models)
            save(run_dir / "model-catalog.json", catalog)
            databases = runtime_path(settings, "source_databases")
            training_databases = (
                {selected_dataset: databases[selected_dataset]}
                if selected_dataset is not None
                else databases
            )
            missing_databases = [
                name for name, path in training_databases.items() if not path.is_file()
            ]
            if missing_databases:
                raise ValueError(
                    "Missing dataset source databases: " + ", ".join(missing_databases)
                )
            source_info = {
                "databases": {
                    name: str(path) for name, path in training_databases.items()
                },
                "input_scope": "all rows of each configured source database; no runtime input restriction",
            }
            source_path = run_dir / "source-data.json"
            if source_path.exists() and read(source_path) != source_info:
                raise ValueError(
                    "Original source path changed; use a new run directory"
                )
            save(source_path, source_info)
            corpora_path = run_dir / "corpora.json"
            if not corpora_path.exists():
                save(
                    corpora_path,
                    inspect_corpora(training_databases, build),
                )
            corpora = read(corpora_path)
            if set(corpora) != set(training_databases):
                raise ValueError(
                    "Frozen corpus metadata differs from the training dataset"
                )
            manifest_path = run_dir / "workloads.json"
            generating = not manifest_path.exists()
            if generating:
                # Freeze the canonical, offline-audited SQL in each run. Never
                # silently regenerate a different workload at training startup.
                manifest = load_manifest(canonical_path)
                manifest = select_manifest(
                    manifest,
                    selected_dataset,
                    configured_counts.get(selected_dataset)
                    if selected_dataset is not None
                    else None,
                )
                queries_by_dataset = {}
                for workload in manifest["workloads"]:
                    queries_by_dataset.setdefault(workload["dataset"], []).append(
                        workload
                    )
                expected_counts = {name: configured_counts[name] for name in corpora}
                if set(queries_by_dataset) != set(corpora) or any(
                    len(group) != expected_counts[dataset]
                    or sum(w["split"] == "validation" for w in group)
                    != round(expected_counts[dataset] * args.validation_fraction)
                    for dataset, group in queries_by_dataset.items()
                ):
                    raise ValueError(
                        "Canonical workload differs from training settings; regenerate it"
                    )
                save(manifest_path, manifest)
            else:
                manifest = read(manifest_path)
            validate_scope(
                manifest,
                selected_dataset,
                configured_counts.get(selected_dataset)
                if selected_dataset is not None
                else None,
            )
            workloads = manifest["workloads"]
            queries = run_dir / "queries"
            queries.mkdir(exist_ok=True)
            for w in workloads:
                (queries / (w["id"] + ".sql")).write_text(w["sql"] + "\n")
            heldout = heldout_manifest()
            split_audit = {
                **audit_workloads(workloads, heldout=heldout,
                                  template_rewrites=load_template_rewrites(canonical_path.parent)),
                "heldout_manifest": heldout,
            }
            if (run_dir / "split-audit.json").exists() and read(
                run_dir / "split-audit.json"
            ) != split_audit:
                raise ValueError("Frozen heldout queries or generated split differ")
            save(run_dir / "split-audit.json", split_audit)
            bindings = run_dir / "query-bindings"
            bindings.mkdir(exist_ok=True)
            update(phase="binding", total_queries=len(workloads))
            budgets = {}
            for w in workloads:
                bound_path = bindings / (w["id"] + ".json")
                if not bound_path.exists():
                    with DuckDBEvaluator(
                        build,
                        models,
                        w["sql"],
                        database=databases[w["dataset"]],
                        parallelism=settings["execution"]["cpu_parallelism"],
                        max_concurrent_requests_per_model=settings["execution"][
                            "max_concurrent_requests_per_model"
                        ],
                        enforce_model_operator_order=settings["execution"][
                            "enforce_model_operator_order"
                        ],
                    ) as evaluator:
                        bound = normalize_explain(evaluator.run(explain=True))
                        save(bound_path, bound)
                bound = read(bound_path)
                if generating:
                    validate_native_budget(bound)
                budgets[w["id"]] = count_operators(bound)
                w["operator_counts"] = budgets[w["id"]]
                w["task_contracts"] = [
                    {
                        key: op.get(key)
                        for key in (
                            "operator_id",
                            "kind",
                            "task",
                            "input_field_schema",
                            "output_schema",
                        )
                    }
                    for op in bound["semantic_operators"]
                ]
                update(
                    current_query=w["id"],
                    completed_queries=state["completed_queries"] + 1,
                )
            save(
                run_dir / "binding-audit.json",
                {
                    "status": "passed",
                    "queries": len(workloads),
                    "model_requests": 0,
                    "operator_counts": budgets,
                },
            )
            save(manifest_path, manifest)
            if args.generate_only:
                update(status="ready", phase="generated", current_query=None)
                return
            checkpoint = None
            replay = []
            answers = None
            if args.rounds:
                update(phase="checking_reference_answers", current_query=None)
                reference_root = (
                    SEMCACHE_ROOT / settings["training"]["reference_answers_dir"]
                )
                pointer = run_dir / "reference_answers_path.json"
                if pointer.is_file():
                    reference_root = Path(read(pointer)["root"])
                elif (
                    not (reference_root / "contract.json").is_file()
                    and (reference_root / "latest.json").is_file()
                ):
                    reference_root = Path(read(reference_root / "latest.json")["root"])
                answers = TrainingAnswers(
                    reference_root,
                    workloads,
                    databases,
                    allow_subset=selected_dataset is not None,
                )
                # Other datasets' pending answers must not delay this run.
                for workload in workloads:
                    answers.load(workload)
                save(
                    run_dir / "reference_answers_path.json",
                    {
                        "root": str(reference_root.resolve()),
                        "system": answers.system,
                        "metric": RESULT_ACCURACY_METRIC,
                    },
                )
            history = CacheHistory(
                run_dir / "history.sqlite",
                capacity=settings["sampling"]["history_capacity"],
            )
            try:
                update(phase="initial_calibration", current_query=None)
                global_path = run_dir / "calibration/global.json"
                if not global_path.exists() and any(
                    run_dir.glob("*/round-*/*/*/collection.json")
                ):
                    raise ValueError(
                        "Missing global calibration for an existing collection; use a fresh run directory"
                    )

                def open_calibration(paths, selected_teacher):
                    return GlobalCalibration(
                        paths["global"],
                        models,
                        selected_teacher,
                        lambda: calibrate_models(
                            workloads,
                            models,
                            selected_teacher,
                            databases,
                            build,
                            paths["probes"],
                            history=history,
                            adaptive=sampling_config.get("adaptive_sketch"),
                            parallelism=resolved_sampling["parallelism"],
                            sample_rows=settings["calibration"]["sample_rows"],
                            cohort_seed=settings["calibration"]["cohort_seed"],
                            progress=calibration_progress_writer(paths["progress"]),
                            shared_cache_dir=(
                                SEMCACHE_ROOT / settings["calibration"]["cache_dir"]
                            ).resolve()
                            if settings["calibration"].get("cache_dir") is not None
                            else None,
                        ),
                    )

                if teacher_epoch is None:
                    calibration = open_calibration(calibration_paths(run_dir), teacher)
                else:
                    calibration = GlobalCalibration(
                        global_path,
                        models,
                        TeacherModel(**teacher_epoch["source_teacher"]),
                    )

                def round_tests(phase, number, checkpoint):
                    from training.evaluation.round import run_requested

                    run_requested(
                        run_dir,
                        phase=phase,
                        round_number=number,
                        checkpoint=checkpoint,
                        settings=settings,
                        workloads=workloads,
                        models=models,
                        catalog=catalog,
                        teacher=teacher,
                        databases=databases,
                        profile=calibration.snapshot(),
                        answers=answers,
                        update=update,
                    )

                for phase, rounds in (
                    ("estimated", args.pretrain_rounds),
                    ("measured", args.rounds),
                ):
                    if phase == "measured" and rounds and teacher_epoch is not None:
                        update(
                            phase="initial_calibration",
                            current_query=None,
                            teacher_epoch=teacher_epoch_id,
                            calibration_path=str(calibration_paths(run_dir)["global"]),
                        )
                        calibration = open_calibration(
                            calibration_paths(run_dir), teacher
                        )
                    for iteration in range(rounds):
                        global_round = (
                            iteration
                            + 1
                            + (args.pretrain_rounds if phase == "measured" else 0)
                        )
                        directory = run_dir / phase / f"round-{iteration + 1}"
                        checkpoint_out = directory / "model/value.pt"
                        if (directory / "experience.json").exists():
                            completed = read(directory / "experience.json")
                            validate_dataset(completed)
                            frozen_round_workload = read(directory / "workloads.json")
                            validate_round_workload(completed, frozen_round_workload)
                            if phase == "measured" and frozen_round_workload != read(
                                run_dir / "workloads.json"
                            ):
                                raise ValueError(
                                    "Completed measured round SQL differs from the current workload"
                                )
                            manifest = read(directory / "round.json")
                            update(
                                round=iteration + 1,
                                global_round=global_round,
                                total_rounds=args.pretrain_rounds + args.rounds,
                                query_order=manifest["query_order"],
                                current_query=None,
                            )
                            if phase == "measured":
                                replay.extend(completed["records"])
                            if collect_only:
                                continue
                            if not checkpoint_out.exists():
                                update(phase="value_update", round=iteration + 1)
                                fit(
                                    replay
                                    if phase == "measured"
                                    else completed["records"],
                                    directory / "model",
                                    epochs=args.epochs,
                                    checkpoint=checkpoint,
                                    seed=args.seed + iteration,
                                    source=phase,
                                    settings=settings,
                                )
                            checkpoint = checkpoint_out
                            round_tests(phase, iteration + 1, checkpoint)
                            continue
                        if checkpoint_out.exists():
                            raise RuntimeError(
                                "Checkpoint has no complete round feedback"
                            )
                        interrupted_measured = (
                            phase == "measured" and directory.exists()
                        )
                        if directory.exists() and (
                            not (directory / "round.json").exists()
                            or phase == "estimated"
                            and (directory / "stream").exists()
                        ):
                            raise RuntimeError(
                                "Interrupted round has no compatible frozen manifest"
                            )
                        cycle_item = (
                            feedback_cycle[iteration % len(feedback_cycle)]
                            if phase == "measured"
                            else {
                                "plan_generation": "heuristic_requests",
                                "query_order": "shuffled",
                            }
                        )
                        plan_generation = cycle_item["plan_generation"]
                        query_order = cycle_item["query_order"]
                        ordered = ordered_workloads(
                            workloads, query_order, args.seed + global_round
                        )
                        directory.mkdir(parents=True, exist_ok=True)
                        round_workloads = directory / "workloads.json"
                        current_workloads = read(run_dir / "workloads.json")
                        if (
                            round_workloads.exists()
                            and read(round_workloads) != current_workloads
                        ):
                            raise ValueError(
                                "Frozen round workload differs from current feedback SQL"
                            )
                        if interrupted_measured and not round_workloads.exists():
                            raise ValueError(
                                "Measured resume requires frozen round workloads"
                            )
                        if not round_workloads.exists():
                            save(round_workloads, current_workloads)
                        manifest = {
                            "round": global_round,
                            "phase": phase,
                            "phase_round": iteration + 1,
                            "query_order": query_order,
                            "query_order_seed": args.seed + global_round
                            if query_order == "shuffled"
                            else None,
                            "feedback_protocol": FEEDBACK_PROTOCOL
                            if phase == "measured"
                            else None,
                            "plan_generation": plan_generation,
                            "cycle_index": (
                                iteration % len(feedback_cycle)
                                if phase == "measured"
                                else None
                            ),
                            "queries": [w["id"] for w in ordered],
                            "checkpoint_in": str(checkpoint) if checkpoint else None,
                            "search_guide": plan_generation,
                            "cache_mode": "cold" if phase == "estimated" else "stream",
                            "cache_reset": "pending",
                        }
                        if phase == "measured" and teacher_epoch is not None:
                            manifest.update(
                                teacher_epoch=teacher_epoch_id,
                                calibration_path=str(
                                    calibration_paths(run_dir)["global"].relative_to(
                                        run_dir
                                    )
                                ),
                            )
                        manifest_path = directory / "round.json"
                        if manifest_path.exists():
                            old_manifest = read(manifest_path)
                            for key in (
                                "round",
                                "phase_round",
                                "queries",
                                "checkpoint_in",
                                "search_guide",
                                "phase",
                                "cache_mode",
                                "query_order",
                                "feedback_protocol",
                                "plan_generation",
                                "query_order_seed",
                                "cycle_index",
                                "teacher_epoch",
                                "calibration_path",
                            ):
                                if old_manifest.get(key) != manifest.get(key):
                                    raise ValueError(
                                        "Frozen cold round manifest differs"
                                    )
                            if interrupted_measured:
                                manifest = old_manifest
                        prefix = (
                            validate_measured_prefix(
                                directory,
                                manifest,
                                ordered,
                                calibration.state,
                                COLLECTION_PROTOCOL,
                            )
                            if interrupted_measured
                            else 0
                        )
                        if not interrupted_measured or prefix < len(ordered):
                            update(
                                phase="resetting_cache",
                                round=iteration + 1,
                                global_round=global_round,
                                total_rounds=args.pretrain_rounds + args.rounds,
                                current_query=None,
                            )
                        if interrupted_measured:
                            cache_segment = begin_resume_segment(
                                directory,
                                manifest,
                                prefix,
                                calibration.state,
                                previous_state,
                                reset=lambda: reset_context(models, history),
                                snapshot=history.snapshot,
                            )
                        else:
                            save(manifest_path, manifest)
                            reset_details = reset_context(models, history)
                            manifest.update(
                                cache_reset="completed",
                                cache_reset_details=reset_details,
                                reset_at=datetime.now().astimezone().isoformat(),
                            )
                            save(manifest_path, manifest)
                            cache_segment = {
                                "id": "initial",
                                "boundary": "round_start",
                                "start_index": 0,
                                "start_query": ordered[0]["id"] if ordered else None,
                            }
                        records = []
                        round_entries = []
                        update(
                            phase=phase,
                            round=iteration + 1,
                            rounds=rounds,
                            query_order=manifest["query_order"],
                            total_queries=len(workloads),
                            completed_queries=0,
                            records=0,
                            cache_segment=cache_segment
                            if phase == "measured"
                            else None,
                        )
                        for query_index, w in enumerate(ordered):
                            destination = directory / manifest["cache_mode"] / w["id"]
                            update(
                                current_query=w["id"],
                                network_retry=None,
                                plan_generation=plan_generation,
                            )
                            reused = (destination / "collection.json").exists() and (
                                destination / "experience.json"
                            ).exists()
                            if reused:
                                payload = read(destination / "experience.json")
                                validate_dataset(payload)
                                if any(
                                    r.get("metadata", {}).get("collection_protocol")
                                    != COLLECTION_PROTOCOL
                                    or r["metadata"]
                                    .get("feature_audit", {})
                                    .get("status")
                                    != "passed"
                                    for r in payload["records"]
                                ):
                                    raise ValueError(
                                        "Completed query has unaudited feedback"
                                    )
                            else:

                                def collect_query(
                                    target=destination,
                                    rollout_index=1,
                                    selection_policy="pareto_random",
                                ):
                                    missing = unavailable_models(models)
                                    if missing:
                                        raise RuntimeError(
                                            "Model endpoints unavailable: "
                                            + ", ".join(missing)
                                        )
                                    return collect_native(
                                        w,
                                        models,
                                        catalog,
                                        databases[w["dataset"]],
                                        build,
                                        target,
                                        checkpoint=checkpoint
                                        if phase == "measured"
                                        else None,
                                        phase=phase,
                                        plan_generation=plan_generation,
                                        repeats=args.repeats,
                                        candidates=args.candidates,
                                        seed=args.seed
                                        + global_round * len(ordered)
                                        + query_index
                                        + (rollout_index - 1) * 1000003,
                                        history=history,
                                        calibration_profile=calibration.snapshot(),
                                        cache_segment=cache_segment
                                        if phase == "measured"
                                        else None,
                                        timeout_policy=estimated_policy
                                        if phase == "estimated"
                                        else policy,
                                        sampling_config=sampling_config,
                                        search_seconds=args.search_seconds,
                                        max_expansions=args.max_expansions,
                                        teacher=teacher,
                                        exploration_fraction=settings["training"].get(
                                            "exploration_fraction", 0.0
                                        ),
                                        inference_settings=settings,
                                        execution_cpu_parallelism=settings["execution"][
                                            "cpu_parallelism"
                                        ],
                                        max_concurrent_requests_per_model=settings[
                                            "execution"
                                        ]["max_concurrent_requests_per_model"],
                                        enforce_model_operator_order=settings[
                                            "execution"
                                        ]["enforce_model_operator_order"],
                                        rollout_index=rollout_index,
                                        selection_policy=selection_policy,
                                    )

                                if phase == "measured":
                                    payload = collect_feedback_query(
                                        w,
                                        destination,
                                        collect=collect_query,
                                        calibration=calibration,
                                        answers=answers,
                                        teacher_config=teacher_config,
                                        round_number=iteration + 1,
                                        completed_before=2 * query_index,
                                        update=update,
                                        progress=print,
                                    )
                                else:
                                    payload = collect_with_retry(
                                        collect_query,
                                        destination=destination,
                                        run_dir=run_dir,
                                        phase=phase,
                                        policy=retry_policy,
                                        reset=lambda: reset_context(models, history),
                                        update=update,
                                        previous_failure=previous_state.get("error")
                                        if previous_state.get("current_query")
                                        == w["id"]
                                        and previous_state.get("phase") == phase
                                        and previous_state.get("round") == iteration + 1
                                        else None,
                                    )
                            validate_round_workload(payload, {"workloads": [w]})
                            if w["split"] == "train" and phase == "estimated":
                                calibration.commit(
                                    f"{phase}/{iteration + 1}/{w['id']}",
                                    **read(destination / "calibration-update.json"),
                                )
                            if phase == "estimated" and not reused:
                                reset_context(models, history)
                                save(
                                    destination / "cache-reset.json",
                                    {
                                        "status": "completed",
                                        "after_query": w["id"],
                                        "reset_at": datetime.now()
                                        .astimezone()
                                        .isoformat(),
                                    },
                                )
                            assert payload is not None
                            records.extend(payload["records"])
                            if phase == "measured":
                                round_entries.extend(rollout_entries(w, destination))
                            update(
                                completed_queries=state["completed_queries"] + 1,
                                records=len(records),
                                network_retry=None,
                            )
                            # Reclaim unreachable trajectory cycles after serialization before constructing the next sampling working set.
                            gc.collect()
                        if phase == "measured":
                            assert answers is not None
                            assert teacher_config is not None
                            update(phase="round_accuracy", current_query=None)
                            finalize_measured_accuracy(
                                round_entries,
                                reference_answers=answers,
                                teacher_config=teacher_config,
                                wait=True,
                                update=update,
                            )
                            update(phase="round_accuracy", pending_reference_answers=0)
                            save(
                                directory / "round-accuracy.json",
                                {
                                    "status": "complete",
                                    "queries": len(round_entries),
                                    "labeled": sum(
                                        r.get("accuracy") is not None for r in records
                                    ),
                                    "method": f"frozen_{answers.system}_teacher_answer",
                                    "metric": RESULT_ACCURACY_METRIC,
                                },
                            )
                            replay.extend(records)
                        save(directory / "experience.json", dataset(records))
                        if collect_only:
                            update(phase="collection_complete", current_query=None)
                            continue
                        update(phase="value_update", current_query=None)
                        fit(
                            replay if phase == "measured" else records,
                            directory / "model",
                            epochs=args.epochs,
                            checkpoint=checkpoint,
                            seed=args.seed + iteration,
                            source=phase,
                            settings=settings,
                        )
                        checkpoint = checkpoint_out
                        round_tests(phase, iteration + 1, checkpoint)
            finally:
                history.close()
            if args.evaluate_selected:
                from training.evaluation.runner import evaluate

                update(phase="testing", current_query=None)
                tested = evaluate(
                    run_dir,
                    checkpoint,
                    settings=configured_settings,
                    timeout=args.timeout,
                    beam_width=args.candidates,
                    search_seconds=args.search_seconds,
                    max_expansions=args.max_expansions,
                    sampling_config=sampling_config,
                    teacher=teacher,
                    update=update,
                    execution_cpu_parallelism=settings["execution"]["cpu_parallelism"],
                    max_concurrent_requests_per_model=settings["execution"][
                        "max_concurrent_requests_per_model"
                    ],
                    enforce_model_operator_order=settings["execution"][
                        "enforce_model_operator_order"
                    ],
                )
                update(
                    test_queries=len(tested),
                    completed_test_queries=sum(
                        t["status"] == "completed" for t in tested
                    ),
                )
                if len(tested) != len(heldout) or any(
                    t["status"] != "completed" for t in tested
                ):
                    raise RuntimeError(
                        "Heldout evaluation incomplete; inspect evaluation/results.json"
                    )
            publish_model(run_dir, checkpoint, selected_dataset, workloads)
            update(
                status="complete",
                phase="collection_complete" if collect_only else "complete",
                current_query=None,
                checkpoint=str(checkpoint) if checkpoint is not None else None,
                model_training="not_started" if collect_only else "complete",
            )
        except BaseException as error:
            update(
                status="stopped" if isinstance(error, KeyboardInterrupt) else "error",
                error=str(error),
            )
            raise
