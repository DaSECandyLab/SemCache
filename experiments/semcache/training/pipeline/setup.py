"""Pipeline initialization, workload auditing and query binding preparation."""

import fcntl
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from planning.backends.execution import DuckDBEvaluator
from planning.execution.timeouts import TimeoutPolicy
from planning.learning.features.schema import FEATURE_VERSION
from planning.learning.training.data import validate_dataset
from planning.search.beam import SearchConfig
from planning.settings.configuration import ROOT as SEMCACHE_ROOT
from planning.settings.configuration import (
    load_sampling_config,
    runtime_path,
    teacher_path,
)
from planning.settings.model_config import discover_models
from planning.settings.parallelism import configure_cpu_resources
from planning.settings.teacher import default_teacher_model, load_teacher
from reference_answers.storage.files import file_digest
from training.artifacts import read, save
from training.calibration.epoch import (
    epoch_identity,
    validate_measured_epoch,
)
from training.collection.collector import (
    COLLECTION_PROTOCOL,
)
from training.collection.features import normalize_explain
from training.collection.lifecycle.recovery import (
    scientific_protocol,
)
from training.collection.policy import (
    FEEDBACK_COLLECTION_ORDER,
    FEEDBACK_PROTOCOL,
    PLAN_GENERATION_POLICIES,
)
from training.fitting import fit
from training.references.scoring import (
    PRE_AGGREGATE_ACCURACY_METRIC,
    RESULT_ACCURACY_METRIC,
)
from training.source_data import inspect_corpora
from training.workloads.checks import audit_workloads, validate_native_budget
from training.workloads.datasets import select_manifest, validate_scope
from training.workloads.provenance import load_template_rewrites
from training.workloads.query_manifest import (
    count_operators,
    frozen_workload_versions,
    heldout_manifest,
    load_manifest,
    validate_round_workload,
)


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


def prepare_pipeline_context(
    args, configured_settings, settings, build, run_dir, previous_state, state, update,
    *, global_lock, selected_dataset, configured_counts, canonical_path, canonical_manifest, collect_only,
):
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
        return None
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
    protocol = {
        "version": COLLECTION_PROTOCOL,
        "build": str(build),
        "configuration": settings,
        "feature_version": FEATURE_VERSION,
        "workload_version": workload_version,
        "pretraining_workload_version": pretraining_workload_version,
        "models": models,
        "teacher": teacher.configuration(),
        "teacher_config_sha256": file_digest(
            teacher_config or teacher_path(settings, getattr(args, "config", None))
        ),
        "reference_root": str(reference_root.resolve()),
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
        return None
    return SimpleNamespace(
        models=models,
        catalog=catalog,
        teacher=teacher,
        databases=databases,
        sampling_config=sampling_config,
        resolved_sampling=resolved_sampling,
        retry_policy=retry_policy,
        teacher_config=teacher_config,
        teacher_epoch=teacher_epoch,
        teacher_epoch_id=teacher_epoch_id,
        workloads=workloads,
        heldout=heldout,
        feedback_cycle=feedback_cycle,
        policy=policy,
        estimated_policy=estimated_policy,
        selected_dataset=selected_dataset,
        collect_only=collect_only,
        source_path=source_path,
        source_info=source_info,
    )
