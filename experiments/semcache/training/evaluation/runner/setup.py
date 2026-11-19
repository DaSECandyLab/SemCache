"""Heldout selected text SQL evaluation after training; no labels enter replay."""

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import sqlglot
from planning.learning.devices import load_inference_model
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
    sampling_for_mode,
)
from planning.settings.teacher import TeacherModel, default_teacher_model, load_teacher
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope
from training.artifacts import save
from training.calibration.epoch import calibration_paths, validate_measured_epoch
from training.calibration.global_profile import GlobalCalibration
from training.references.evaluation import policy as reference_policy
from training.references.saved_evaluation import SavedEvaluationReferences
from training.workloads.query_manifest import heldout_manifest, ordered_heldout_manifest


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

def prepare_evaluation(
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
    layer_pruning=None,
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
    if layer_pruning not in (None, "pareto", "accuracy_time", "grouped"):
        raise ValueError(
            "layer_pruning must be None, pareto, accuracy_time or grouped"
        )
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
            "layer_pruning": layer_pruning or "auto",
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
    return SimpleNamespace(
        output=output,
        settings=settings,
        search_seconds=search_seconds,
        max_expansions=max_expansions,
        beam_width=beam_width,
        timeout=timeout,
        execution_cpu_parallelism=execution_cpu_parallelism,
        max_concurrent_requests_per_model=max_concurrent_requests_per_model,
        enforce_model_operator_order=enforce_model_operator_order,
        models=models,
        databases=databases,
        profile=profile,
        manifest=manifest,
        order=order,
        results=results,
        completed=completed,
        terminal_statuses=terminal_statuses,
        pending_execution=pending_execution,
        work=work,
        minimum_accuracy=minimum_accuracy,
        planning_only=planning_only,
        execution_only=execution_only,
        cache_reset_scope=cache_reset_scope,
        progress=progress,
        update=update,
        build=build,
        run_dir=run_dir,
        checkpoint=checkpoint,
        fixed_model_id=fixed_model_id,
        teacher=teacher,
        catalog=catalog,
        fixed_layout=fixed_layout,
        layer_pruning=layer_pruning,
        operator_priority_mode=operator_priority_mode,
        plan_generation=plan_generation,
        preference=preference,
        quality_policy=quality_policy,
        ranker=ranker,
        reference_output=reference_output,
        reference_reuse_only=reference_reuse_only,
        reference_teacher_config=reference_teacher_config,
        reference_teacher=reference_teacher,
        request_guided=request_guided,
        rewrite_relations=rewrite_relations,
        sampling_config=sampling_config,
        saved_references=saved_references,
        transport_models=transport_models,
    )
