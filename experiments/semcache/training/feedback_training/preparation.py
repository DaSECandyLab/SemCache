"""Feedback training run configuration, validation and manifest preparation."""

import copy
import fcntl
import json
import math
import os
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import torch
from planning.learning.model import PlanValueModel
from planning.learning.training.targets import TARGET_PROTOCOL
from planning.prompt_layouts import BLOCK_ORDERS
from planning.settings.configuration import (
    ROOT,
    activate_native,
    load_config,
    runtime_path,
)
from planning.settings.model_config import discover_models
from planning.settings.parallelism import configure_cpu_resources
from planning.settings.teacher import default_teacher_model, load_teacher
from reference_answers.storage.files import file_digest
from training.artifacts import read, save
from training.collection.lifecycle.recollection import start_recollection
from training.collection.policy import DUPLICATE_PLAN_POLICY
from training.references.answers import TrainingAnswers
from training.references.scoring import NON_FREE_TEXT_METRIC
from training.reporting.training import (
    saved_executions,
)
from training.workloads.query_manifest import heldout_manifest, ordered_heldout_manifest
from training.workloads.workload_resume import (
    accept_workload_update,
    round_query_counts,
)

ACCURACY_MINIMUMS = (0.2, 0.5, 0.8)
SEARCHES_PER_QUERY = len(ACCURACY_MINIMUMS)


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def initial_checkpoint(path, settings):
    if path.exists():
        restored = PlanValueModel.load(path)
        if (restored.training_info or {}).get("initialization") != "random":
            raise ValueError("Feedback training requires random initial weights")
        return
    torch.manual_seed(settings["training"]["seed"])
    model = PlanValueModel(
        **settings["value_model"], time_scale=settings["learning"]["time_scale"]
    )
    model.training_info = {
        "initialization": "random",
        "epochs": 0,
        "pretraining": False,
        "target_protocol": TARGET_PROTOCOL,
        "purpose": "feedback training from random initial weights",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)

def prepare_feedback_run(
    output,
    references,
    *,
    workloads_path=None,
    allow_heldout_training=False,
    baseline_protocol=None,
    global_http_limit=None,
    execution_timeout=None,
    resume_round=None,
    rounds=3,
    exclude_training_rounds=(),
    preserve_resume_cache=False,
    learning_rate=None,
    collect_only=False,
    collection_checkpoint=None,
    recollect_queries=(),
):
    output, references = Path(output).resolve(), Path(references).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if collect_only != (collection_checkpoint is not None):
        raise ValueError("Collection-only runs require an explicit existing checkpoint")
    recollect_queries = set(recollect_queries)
    if recollect_queries and (
        not collect_only or not (output / "protocol.json").is_file()
    ):
        raise ValueError(
            "Selective recollection requires an existing collection-only run"
        )
    if collection_checkpoint is not None:
        collection_checkpoint = Path(collection_checkpoint).resolve()
        if not collection_checkpoint.is_file():
            raise FileNotFoundError(collection_checkpoint)
    run_guard = (output / "pipeline.lock").open("a")
    fcntl.flock(run_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    resource_guard = (
        ROOT.parent / "experiments/results/semcache/model_training/.collection.lock"
    ).open("a")
    fcntl.flock(resource_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    baseline_digest = None
    if baseline_protocol:
        if Path(baseline_protocol).is_file():
            baseline = read(baseline_protocol)
            baseline_digest = file_digest(baseline_protocol)
        else:
            # The run freezes its full configuration; deleted baseline results
            # must not make an otherwise self-contained continuation unusable.
            baseline = read(output / "protocol.json")
            baseline_digest = baseline["baseline_protocol_sha256"]
        settings = copy.deepcopy(baseline["configuration"])
    else:
        settings = copy.deepcopy(load_config())
    if learning_rate is not None:
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive")
        settings["learning"]["learning_rate"] = learning_rate
    if type(rounds) is not int or rounds < 1:
        raise ValueError("Training rounds must be a positive integer")
    if preserve_resume_cache and resume_round is None:
        raise ValueError("Keeping serving state requires an explicit resume round")
    excluded_rounds = set(exclude_training_rounds)
    if any(
        type(number) is not int or not 1 <= number < rounds
        for number in excluded_rounds
    ):
        raise ValueError("Excluded training rounds must precede the last round")
    for number in excluded_rounds:
        if not (output / f"measured/round-{number}/model/value.pt").is_file():
            raise ValueError(
                "Only previously fitted rounds can be excluded from replay"
            )
    if execution_timeout is not None:
        if not math.isfinite(execution_timeout) or execution_timeout <= 0:
            raise ValueError("Execution timeout must be positive")
        # Apply the execution budget to every rollout without shortening it from earlier timings.
        settings["training"]["timeout"] = execution_timeout
        settings["training"]["minimum_timeout"] = execution_timeout
    if global_http_limit is not None:
        if not 1 <= global_http_limit <= 1024:
            raise ValueError("Global HTTP concurrency must be between 1 and 1024")
        os.environ["SEMCACHE_GLOBAL_HTTP_CONCURRENCY"] = str(global_http_limit)
        os.environ["SEMCACHE_GLOBAL_HTTP_AUDIT"] = str(
            output / "http-concurrency.jsonl"
        )
    settings["teacher_config"] = "teacher.opencode.json"
    settings["training"].update(
        pretrain_rounds=0,
        rounds=rounds,
        dataset=None,
        allow_random_initialization=True,
        feedback_rollouts_per_query=SEARCHES_PER_QUERY,
        feedback_accuracy_minimums=list(ACCURACY_MINIMUMS),
        teacher_config="teacher.opencode.json",
        feedback_cycle=[
            {
                "plan_generation": "heuristic_value",
                "operator_priority_mode": "heuristic_requests",
                "query_order": "shuffled",
            },
        ]
        + [
            {
                "plan_generation": "heuristic_value",
                "operator_priority_mode": "heuristic_value",
                "query_order": "shuffled",
            }
            for _ in range(rounds - 1)
        ],
    )
    if excluded_rounds:
        settings["training"]["replay_exclude_rounds"] = sorted(excluded_rounds)
    else:
        settings["training"].pop("replay_exclude_rounds", None)
    # Training never consumes or publishes prompt answers across requests.
    settings["response_cache"]["directory"] = None
    settings["response_cache"]["verify_on_second_use"] = False
    configure_cpu_resources(settings, configure_torch=True)
    activate_native(settings)
    teacher_config = ROOT / "config/teacher/teacher.opencode.json"
    teacher = load_teacher(default_teacher_model(teacher_config), {}, teacher_config)
    if (
        teacher.endpoint.get("thinking") != "disabled"
        or teacher.endpoint.get("reasoning_effort") != "none"
    ):
        raise ValueError("Beam training teacher must have thinking disabled")
    models, catalog = discover_models(
        settings["models"]["endpoints"],
        max_output_tokens=settings["models"]["max_output_tokens"],
    )
    if set(models) != {"gemma4-e2b", "gemma4-e4b", "gemma4-12b"}:
        raise ValueError("Feedback training requires exactly E2B, E4B and 12B")
    if workloads_path is None and not allow_heldout_training:
        default_workloads = ROOT.parent / "experiments/semcache/queries/workloads.json"
        if not default_workloads.is_file():
            raise ValueError(
                "Training on heldout queries requires allow_heldout_training=True"
            )
        workloads_path = default_workloads
    custom_workloads = read(workloads_path)["workloads"] if workloads_path else None
    manifest = (
        ordered_heldout_manifest(heldout_manifest())
        if custom_workloads is None and allow_heldout_training
        else []
    )
    workloads = []
    for item in manifest:
        sql = Path(item["path"]).read_text()
        short = item["id"].removeprefix("translated/semcache/")
        domain = short.split("/")[0]
        workloads.append(
            {
                **item,
                "sql": sql,
                "split": "train",
                "group": item["id"],
                "dataset": {"movies": "movies_nirvana"}.get(domain, domain),
                "public_query_id": short,
                "rows": None,
                "output_contract": {},
                "accuracy_metric": NON_FREE_TEXT_METRIC,
            }
        )
    if custom_workloads is not None:
        if not custom_workloads or (
            not allow_heldout_training
            and any(w.get("split") != "train" for w in custom_workloads)
        ):
            raise ValueError("Training manifest must contain training queries")
        workloads = [
            dict(w, public_query_id=w["id"], accuracy_metric=NON_FREE_TEXT_METRIC)
            for w in custom_workloads
        ]
        databases = runtime_path(settings, "source_databases")
    else:
        databases = {
            name: ROOT.parent / "benchmarks/databases/semcache" / f"{name}.duckdb"
            for name in sorted({w["dataset"] for w in workloads})
        }
    query_count = len(workloads)
    if recollect_queries - {w["id"] for w in workloads}:
        raise ValueError("Recollection query is absent from the frozen workload")
    if len({workload["id"] for workload in workloads}) != query_count:
        raise ValueError("Training workload IDs must be unique")
    if custom_workloads is not None:
        settings["training"]["queries_per_dataset"] = dict(
            sorted(Counter(workload["dataset"] for workload in workloads).items())
        )
    training_scope = (
        "training manifest; public test queries excluded"
        if custom_workloads is not None
        else "in-sample feedback; public queries intentionally used for training"
    )
    # A custom workload manifest may intentionally select a subset of the
    # frozen reference set (for example, recollecting one dataset). Keep the
    # strict full-set check for ordinary runs, while allowing the selected
    # workload IDs to be validated against the same frozen answers.
    answers = TrainingAnswers(
        references, workloads, databases, allow_subset=custom_workloads is not None
    )
    previous_protocol = read(output / "protocol.json") if recollect_queries else None
    reference_evidence = {
        w["id"]: (
            previous_protocol["answer_sha256"][w["id"]]
            if recollect_queries and w["id"] not in recollect_queries
            else answers.load(w)["answer_sha256"]
        )
        for w in workloads
    }
    frozen = {
        "experiment": "training-manifest"
        if custom_workloads is not None
        else "in-sample-feedback",
        "generalization_evaluation": False,
        "configuration": settings,
        "initialization": "random",
        "pretraining": False,
        "searches_per_query": SEARCHES_PER_QUERY,
        "minimum_accuracies": ACCURACY_MINIMUMS,
        "query_sampling_policy": "once_per_query_round_v1",
        "duplicate_plan_policy": DUPLICATE_PLAN_POLICY,
        "models": models,
        "teacher": teacher.configuration(),
        "teacher_config_sha256": file_digest(teacher_config),
        "target_protocol": TARGET_PROTOCOL,
        "layouts": BLOCK_ORDERS,
        "reference_root": str(references),
        "reference_contract_sha256": file_digest(references / "contract.json"),
        "answer_sha256": reference_evidence,
        "database_sha256": {
            key: file_digest(value) for key, value in databases.items()
        },
        "workload_sql_sha256": {w["id"]: w["sql_sha256"] for w in workloads},
        "reward": "(-observed_execution_seconds, observed_non_free_text_F1)",
        "cache_policy": "prompt answer cache disabled; reset model KV/history at round boundaries; continuous within each round",
        "round_query_counts": round_query_counts(output, query_count, rounds),
    }
    if collect_only:
        frozen.update(
            collect_only=True,
            collection_checkpoint=str(collection_checkpoint),
            collection_checkpoint_sha256=file_digest(collection_checkpoint),
        )
    if baseline_protocol:
        frozen["baseline_protocol_sha256"] = baseline_digest
    if workloads_path:
        frozen["training_manifest_sha256"] = file_digest(workloads_path)
    if global_http_limit is not None:
        frozen["global_http_limit"] = global_http_limit
    if execution_timeout is not None:
        frozen["execution_timeout_seconds"] = execution_timeout
    protocol_path = output / "protocol.json"
    # JSON normalization makes tuple-based layout constants compare consistently.
    frozen = json.loads(json.dumps(frozen))
    if protocol_path.exists() and read(protocol_path) != frozen:
        previous = read(protocol_path)
        upgraded = {
            **previous,
            "query_sampling_policy": "once_per_query_round_v1",
            "round_query_counts": frozen["round_query_counts"],
        }
        if recollect_queries:
            # New teacher answers apply only to replaced executions. Retained
            # records keep their original, explicit answer provenance.
            upgraded.update(
                answer_sha256=frozen["answer_sha256"],
                reference_contract_sha256=frozen["reference_contract_sha256"],
            )
        if collect_only:
            upgraded.update(
                collect_only=True,
                collection_checkpoint=str(collection_checkpoint),
                collection_checkpoint_sha256=file_digest(collection_checkpoint),
            )
            if upgraded == frozen and previous.get("collect_only") is not True:
                save(
                    output / "collection-only-policy.json",
                    {
                        "time": now(),
                        "collect_only": True,
                        "checkpoint": str(collection_checkpoint),
                        "checkpoint_sha256": file_digest(collection_checkpoint),
                        "previous_fitting_artifacts_used": False,
                        "existing_execution_records_preserved": True,
                    },
                )
        if learning_rate is not None and resume_round is not None:
            previous_rate = previous["configuration"]["learning"]["learning_rate"]
            if learning_rate < previous_rate:
                upgraded = copy.deepcopy(upgraded)
                upgraded["configuration"]["learning"]["learning_rate"] = learning_rate
                if upgraded == frozen:
                    save(
                        output / "learning-rate-update.json",
                        {
                            "time": now(),
                            "round": resume_round,
                            "previous_rate": previous_rate,
                            "learning_rate": learning_rate,
                            "weights": "best monitored weights in saved epoch checkpoint",
                            "optimizer_reset": True,
                            "warmup_restarted": False,
                            "completed_executions_preserved": True,
                        },
                    )
        duplicate_policy_upgrade = previous.get("duplicate_plan_policy") == (
            "random_unseen_completion_within_each_query_round"
        )
        if duplicate_policy_upgrade:
            upgraded["duplicate_plan_policy"] = DUPLICATE_PLAN_POLICY
        if upgraded != frozen:
            if not accept_workload_update(
                output,
                {**previous, "duplicate_plan_policy": DUPLICATE_PLAN_POLICY}
                if duplicate_policy_upgrade
                else previous,
                frozen,
                resume_round=resume_round,
            ):
                raise ValueError("Frozen feedback training inputs or settings changed")
        elif previous.get("query_sampling_policy") is None:
            save(
                output / "sampling-policy-update.json",
                {
                    "from": "independent_sampling_per_search",
                    "to": "once_per_query_round_v1",
                    "time": now(),
                    "preserved_executions": saved_executions(output),
                    "existing_completed_queries_are_not_rerun": True,
                },
            )
        if duplicate_policy_upgrade:
            save(
                output / "duplicate-policy-update.json",
                {
                    "time": now(),
                    "from": previous["duplicate_plan_policy"],
                    "to": DUPLICATE_PLAN_POLICY,
                    "applies_to": "newly selected rollouts; completed executions preserved",
                    "accuracy_floor_preference_unchanged": True,
                },
            )
    save(protocol_path, frozen)
    recollection_path = output / "recollection.json"
    recollection = None
    if recollect_queries:
        recollection = start_recollection(
            output, recollect_queries, rounds, collection_checkpoint, SEARCHES_PER_QUERY
        )
    save(output / "models.json", models)
    save(output / "model-catalog.json", catalog)
    save(
        output / "workloads.json",
        {
            "suite": "training-manifest"
            if custom_workloads is not None
            else "public-queries-intentionally-used-for-training",
            "workloads": workloads,
        },
    )
    save(
        output / "reference_answers_path.json",
        {"root": str(references), "system": "semcache"},
    )
    save(output / "cpu-resources.json", configure_cpu_resources(settings))
    initial = collection_checkpoint if collect_only else output / "initial/value.pt"
    if not collect_only:
        initial_checkpoint(initial, settings)
    total_executions = sum(frozen["round_query_counts"]) * SEARCHES_PER_QUERY
    restored_executions = saved_executions(output)
    state = {
        "status": "running",
        "phase": "initial_calibration",
        "pid": os.getpid(),
        "round": 0,
        "rounds": rounds,
        "query_count": query_count,
        "round_query_counts": frozen["round_query_counts"],
        "current_query": None,
        "completed_executions": restored_executions,
        "restored_executions": restored_executions,
        "total_executions": total_executions,
        "searches_per_query": SEARCHES_PER_QUERY,
        "minimum_accuracies": ACCURACY_MINIMUMS,
        "started_at": now(),
        "collect_only": collect_only,
    }
    (output / "feedback-training.pid").write_text(str(os.getpid()) + "\n")
    state_lock, stopping = threading.RLock(), threading.Event()
    return SimpleNamespace(
        output=output,
        settings=settings,
        workloads=workloads,
        manifest=manifest,
        models=models,
        catalog=catalog,
        teacher=teacher,
        databases=databases,
        references=references,
        collect_only=collect_only,
        collection_checkpoint=collection_checkpoint,
        recollect_queries=recollect_queries,
        recollection=recollection,
        resume_round=resume_round,
        preserve_resume_cache=preserve_resume_cache,
        rounds=rounds,
        query_count=query_count,
        total_executions=total_executions,
        restored_executions=restored_executions,
        execution_timeout=execution_timeout,
        global_http_limit=global_http_limit,
        baseline_protocol=baseline_protocol,
        excluded_rounds=excluded_rounds,
        learning_rate=learning_rate,
        answers=answers,
        frozen=frozen,
        recollection_path=recollection_path,
        reference_evidence=reference_evidence,
        teacher_config=teacher_config,
        training_scope=training_scope,
        run_guard=run_guard,
        resource_guard=resource_guard,
        initial=initial,
        state=state,
        state_lock=state_lock,
        stopping=stopping,
    )
