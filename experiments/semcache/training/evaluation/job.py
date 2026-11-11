"""Run heldout inference after a training pipeline completes successfully."""

import argparse
import fcntl
import os
import threading
import time
from contextlib import closing
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from planning.settings.configuration import (
    ROOT,
    activate_native,
    runtime_path,
    teacher_path,
)
from reference_answers.storage.files import file_digest

from training.artifacts import read, save
from training.query_manifest import heldout_manifest, ordered_heldout_manifest


def frozen_reference_teacher(spec):
    """Restore a credential-free historical policy only for saved answer reuse."""
    from planning.settings.teacher import KINDS, load_teacher

    if "frozen_reference_teacher" not in spec:
        return None
    if spec.get("reference_reuse_only") is not True:
        raise ValueError(
            "A frozen reference teacher requires reference_reuse_only=true"
        )
    frozen = spec["frozen_reference_teacher"]
    if not isinstance(frozen, dict) or set(frozen) != {
        "model_id",
        "endpoint",
        "supported_kinds",
    }:
        raise ValueError(
            "Frozen reference teacher must be a TeacherModel configuration"
        )
    model_id, endpoint, kinds = (
        frozen["model_id"],
        frozen["endpoint"],
        frozen["supported_kinds"],
    )
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("Frozen reference teacher requires a nonempty model_id")
    if not isinstance(endpoint, dict) or any(
        not isinstance(endpoint.get(name), str) or not endpoint[name].strip()
        for name in ("model", "base_url")
    ):
        raise ValueError("Frozen reference endpoint requires model and base_url")
    if (
        not isinstance(kinds, list)
        or not kinds
        or any(not isinstance(kind, str) or kind not in KINDS for kind in kinds)
        or len(set(kinds)) != len(kinds)
    ):
        raise ValueError("Frozen reference teacher has unsupported semantic kinds")

    def check_credentials(value):
        if isinstance(value, dict):
            for name, child in value.items():
                if not isinstance(name, str) or name.lower() in {
                    "api_key",
                    "authorization",
                }:
                    raise ValueError("Frozen reference teacher must omit credentials")
                check_credentials(child)
        elif isinstance(value, list):
            for child in value:
                check_credentials(child)

    check_credentials(endpoint)
    teacher = load_teacher(
        model_id,
        {},
        {"models": {model_id: {**endpoint, "supported_kinds": kinds}}},
    )
    if teacher.configuration() != frozen:
        raise ValueError("Frozen reference teacher must use canonical configuration")
    return teacher


def reference_directory(teacher_config):
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M")
    provider = Path(teacher_config).stem.removeprefix("teacher.")
    return (
        ROOT
        / "experiments/results/reference_answers/semcache/testing"
        / provider
        / ("answers-" + stamp)
    )


def request(run, *, timeout=600, planning_only=False):
    """Freeze the user's evaluation separately from the running training job."""
    settings = read(run / "protocol.json")["configuration"]
    manifest = heldout_manifest()
    frozen = read(run / "split-audit.json")["heldout_manifest"]
    if manifest != frozen:
        raise ValueError("Heldout queries changed since training started")
    datasets = {
        "movies_nirvana"
        if Path(q["path"]).parent.name == "movies"
        else Path(q["path"]).parent.name
        for q in manifest
    }
    databases = {
        name: str(ROOT.parent / "benchmarks/databases/semcache" / (name + ".duckdb"))
        for name in sorted(datasets)
    }
    if not all(Path(path).is_file() for path in databases.values()):
        raise FileNotFoundError("Missing full original SemCache benchmark database")
    rounds = settings["training"]["rounds"]
    mode = settings.get("inference", {}).get("plan_generation", "heuristic_requests")
    settings["sampling"]["offline_statistics_dir"] = (
        ".cache/evaluation-prefix-statistics"
    )
    spec = {
        "training_run": str(run),
        "checkpoint": str(run / f"measured/round-{rounds}/model/value.pt")
        if mode == "heuristic_value"
        else None,
        "plan_generation": mode,
        "timeout": timeout,
        "reference_timeout_seconds": None,
        "preferences": ["fastest", "accurate"]
        if mode == "heuristic_value"
        else ["fastest"],
        "settings": settings,
        "manifest": manifest,
        "query_order": [item["id"] for item in ordered_heldout_manifest(manifest)],
        "databases": databases,
        "database_sha256": {
            name: file_digest(path) for name, path in databases.items()
        },
        "teacher_config_sha256": file_digest(teacher_path(settings)),
        "input_scope": "full original benchmark databases; SQL predicates and explicit SQL limits unchanged",
        "cache_policy": "cold_before_each_query_then_search_execution_continuous",
        "reference_policy": "configured teacher uses heuristic_requests Beam, fixed teacher/GOTDF and independent operator sampling; saved once and shared between preferences",
        "reference_output": str(reference_directory(settings["teacher_config"])),
        "training_feedback_used": False,
        **({"planning_only": True} if planning_only else {}),
    }
    source = settings["inference"].get("reference_evaluation_run")
    if source and not planning_only:
        from training.references.saved_evaluation import SavedEvaluationReferences

        aggregate = settings["inference"].get("reference_aggregate_manifest")
        saved = SavedEvaluationReferences(
            source, manifest, databases, aggregate_manifest=aggregate
        )
        spec.update(
            reference_evaluation_run=str(saved.root),
            reference_aggregate_manifest=str((ROOT / aggregate).resolve())
            if aggregate
            else None,
            reference_reuse_only=True,
            reference_policy=saved.policy,
            reference_output=str(saved.root),
        )
    return spec


def render(output, spec, state):
    from experiments.semcache.monitoring.evaluation_report import (
        render as render_report,
    )

    return render_report(output, spec, state)


def evaluate_preferences(output, spec, evaluate, update, **arguments):
    """Use identical inputs/checkpoint with separate contracts and results."""
    rows = []
    for preference in spec["preferences"]:
        update(
            status="running",
            training_phase=None,
            preference=preference,
            current_query=None,
        )

        def progress(**values):
            if "completed_test_queries" in values:
                values["completed_test_queries"] += len(rows)
            values["total_test_queries"] = len(spec["manifest"]) * len(
                spec["preferences"]
            )
            update(preference=preference, **values)

        observed = evaluate(
            output=output / preference,
            preference=preference,
            update=progress,
            **arguments,
        )
        rows.extend(dict(row, preference=preference) for row in observed)
        save(
            output / "results.json",
            {
                "queries": rows,
                "preferences": spec["preferences"],
                "total_queries": len(spec["manifest"]) * len(spec["preferences"]),
                "training_feedback_used": False,
                **({"planning_only": True} if spec.get("planning_only") else {}),
            },
        )
    return rows


def evaluation_succeeded(spec, rows):
    """Require every frozen query/preference pair and a successful terminal result."""
    expected = {
        (item["id"], preference)
        for item in spec["manifest"]
        for preference in spec["preferences"]
    }
    if (
        not expected
        or len(rows) != len(expected)
        or {(row.get("query"), row.get("preference")) for row in rows} != expected
    ):
        return False
    if spec.get("planning_only", False):
        return all(
            row.get("status") in ("planned", "infeasible")
            and row.get("planning_only") is True
            and row.get("execution_status") == "not_run"
            for row in rows
        )
    return all(
        row.get("status") == "completed"
        and (spec.get("execution_only", False) or row.get("accuracy") is not None)
        for row in rows
    )


def wait_for_evaluation(spec, update):
    """A finished campaign and its released lock gate the next model workload."""
    if not spec.get("after_evaluation"):
        return
    previous = Path(spec["after_evaluation"])
    previous_spec = read(previous / "request.json")
    if not isinstance(previous_spec, dict):
        raise ValueError("Previous evaluation request must be an object")
    manifest = previous_spec.get("manifest")
    preferences = previous_spec.get("preferences")
    if (
        not isinstance(manifest, list)
        or not manifest
        or any(
            not isinstance(item, dict)
            or not isinstance(item.get("id"), str)
            or not item["id"]
            for item in manifest
        )
        or len({item["id"] for item in manifest}) != len(manifest)
        or not isinstance(preferences, list)
        or not preferences
        or any(value not in ("fastest", "accurate") for value in preferences)
        or len(set(preferences)) != len(preferences)
    ):
        raise ValueError(
            "Previous evaluation request has invalid manifest or preferences"
        )
    expected_pairs = {
        (item["id"], preference) for item in manifest for preference in preferences
    }
    expected = len(expected_pairs)
    while True:
        state = read(previous / "job-status.json")
        if state["status"] in ("complete", "finished_with_issues"):
            results = read(previous / "results.json")
            rows = results.get("queries")
            complete = (
                results.get("total_queries") == expected
                and isinstance(rows, list)
                and len(rows) == expected
                and all(
                    isinstance(row, dict)
                    and isinstance(row.get("query"), str)
                    and isinstance(row.get("preference"), str)
                    for row in rows
                )
                and {(row["query"], row["preference"]) for row in rows}
                == expected_pairs
            )
            with (previous / "job.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    if complete:
                        return
                except BlockingIOError:
                    pass
        update(
            status="waiting_for_previous_evaluation",
            previous_status=state["status"],
            previous_query=state.get("current_query"),
            after_evaluation=str(previous),
        )
        time.sleep(10)


def run_job(run, spec, output=None):
    output = Path(output) if output is not None else run / "evaluation"
    output.mkdir(exist_ok=True)
    state = {"status": "waiting_for_training", "pid": os.getpid()}
    report_lock = threading.RLock()
    stop_reports = threading.Event()

    def update(**values):
        with report_lock:
            state.update(values)
            state["updated_at"] = datetime.now().astimezone().isoformat()
            save(output / "job-status.json", state)
            render(output, spec, state)

    def refresh_reports():
        interval = (
            spec.get("settings", {}).get("monitoring", {}).get("refresh_seconds", 10)
        )
        while not stop_reports.wait(interval):
            update()

    reporter = threading.Thread(target=refresh_reports, daemon=True)
    reporter.start()

    try:
        update()
        wait_for_evaluation(spec, update)
        last = None
        while True:
            training = read(run / "pipeline-status.json")
            marker = tuple(
                training.get(k)
                for k in ("status", "phase", "round", "completed_queries")
            )
            if training["status"] == "complete":
                break
            if training["status"] in ("error", "stopped"):
                update(
                    status="blocked_by_training",
                    error=training.get("error") or "Training stopped",
                )
                return
            if marker != last:
                update(
                    training_phase=training.get("phase"),
                    training_round=training.get("round"),
                    training_completed_queries=training.get("completed_queries", 0),
                )
                last = marker
            time.sleep(10)
        mode = spec.get("plan_generation", "heuristic_requests")
        from planning.settings.modes import validate_mode

        validate_mode(mode)
        checkpoint = Path(spec["checkpoint"]) if mode == "heuristic_value" else None
        if checkpoint is not None:
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            if (
                spec.get("checkpoint_sha256") is not None
                and file_digest(checkpoint) != spec["checkpoint_sha256"]
            ):
                raise ValueError("Scheduled value model checkpoint changed")
        # Wait until the training controller has released its data/model lock.
        with (run / "pipeline.lock").open("a") as training_lock:
            while True:
                try:
                    fcntl.flock(training_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(1)
            settings = spec["settings"]
            activate_native(settings)
            from planning.backends.duckdb import DuckDBConnection
            from planning.observations.offline_statistics import (
                OfflineStatisticsStore,
                build_database,
            )
            from planning.settings.parallelism import configure_cpu_resources
            from planning.settings.teacher import default_teacher_model, load_teacher

            from training.evaluation.runner import evaluate

            configure_cpu_resources(settings, configure_torch=True)
            workload_manifest = spec.get("frozen_workload_manifest", spec["manifest"])
            if heldout_manifest() != workload_manifest:
                raise ValueError("Scheduled test SQL changed")
            if file_digest(teacher_path(settings)) != spec["teacher_config_sha256"]:
                raise ValueError("Scheduled teacher configuration changed")
            build = runtime_path(settings, "build")
            update(status="building_statistics", training_phase=None)
            for name, path in spec["databases"].items():
                if file_digest(path) != spec["database_sha256"][name]:
                    raise ValueError(f"Scheduled original database changed: {name}")
                store = OfflineStatisticsStore(
                    settings["sampling"]["offline_statistics_dir"], path
                )
                with closing(DuckDBConnection(build, path)) as db:
                    tables = [
                        row[0]
                        for row in db.execute(
                            "SELECT table_name FROM duckdb_tables() WHERE schema_name='main'"
                        )
                    ]
                missing = [
                    table for table in tables if store.load("main", table) is None
                ]
                if missing:
                    build_database(
                        path,
                        settings["sampling"]["offline_statistics_dir"],
                        build,
                        tables=missing,
                        config=settings["sampling"].get("adaptive_sketch"),
                        parallelism=settings["statistics"]["build_parallelism"],
                    )
            models = read(run / "models.json")
            teacher_file = teacher_path(settings)
            teacher = load_teacher(
                default_teacher_model(teacher_file), models, teacher_file
            )
            reference_file = None
            reference_teacher = None
            reference_output = None
            saved_source = spec.get("reference_evaluation_run") or settings[
                "inference"
            ].get("reference_evaluation_run")
            if saved_source and not (
                spec.get("planning_only", False) or spec.get("execution_only", False)
            ):
                from training.references.saved_evaluation import (
                    SavedEvaluationReferences,
                )

                saved = SavedEvaluationReferences(
                    saved_source,
                    spec["manifest"],
                    spec["databases"],
                    aggregate_manifest=spec.get("reference_aggregate_manifest")
                    or settings["inference"].get("reference_aggregate_manifest"),
                )
                if (
                    isinstance(spec.get("reference_policy"), dict)
                    and saved.policy != spec["reference_policy"]
                ):
                    raise ValueError("Scheduled saved reference evidence changed")
                reference_output = saved.root
            elif not spec.get("planning_only", False):
                reference_file = Path(
                    spec.get("reference_teacher_config", teacher_file)
                )
                if spec.get("reference_teacher_config_sha256") is not None and (
                    file_digest(reference_file)
                    != spec["reference_teacher_config_sha256"]
                ):
                    raise ValueError(
                        "Scheduled reference teacher configuration changed"
                    )
                reference_teacher = frozen_reference_teacher(spec)
                if reference_teacher is None:
                    reference_teacher = load_teacher(
                        default_teacher_model(reference_file), models, reference_file
                    )
                reference_output = Path(
                    spec.get("reference_output", output / "references")
                )
                if spec.get("reference_output"):
                    from training.references.evaluation import policy

                    reference_contract = {
                        "version": 1,
                        "queries": [
                            {"id": item["id"], "sql_sha256": item["sql_sha256"]}
                            for item in workload_manifest
                        ],
                        "database_sha256": spec["database_sha256"],
                        "policy": policy(reference_teacher, settings),
                    }
                    contract_file = reference_output / "contract.json"
                    if (
                        contract_file.exists()
                        and read(contract_file) != reference_contract
                    ):
                        raise ValueError(
                            "Independent reference inputs or teacher changed"
                        )
                    if not contract_file.exists():
                        if spec.get("reference_reuse_only"):
                            raise FileNotFoundError("Missing shared reference contract")
                        save(contract_file, reference_contract)
            save(
                output / "checkpoint.json",
                {
                    "path": str(checkpoint) if checkpoint else None,
                    "sha256": file_digest(checkpoint) if checkpoint else None,
                },
            )
            rows = evaluate_preferences(
                output,
                spec,
                evaluate,
                update,
                run_dir=run,
                checkpoint=checkpoint,
                timeout=spec["timeout"],
                beam_width=settings["search"]["beam_width"],
                search_seconds=settings["search"]["inference"]["max_seconds"],
                max_expansions=settings["search"]["inference"]["max_expansions"],
                sampling_config=settings["sampling"],
                teacher=teacher,
                reference_teacher=reference_teacher,
                reference_teacher_config=reference_file,
                reference_output=reference_output,
                reference_evaluation_run=saved_source,
                reference_aggregate_manifest=spec.get("reference_aggregate_manifest"),
                plan_generation=spec.get("plan_generation", "heuristic_requests"),
                reference_reuse_only=spec.get("reference_reuse_only", False),
                candidate_models=spec.get("candidate_models"),
                fixed_layout=spec.get("fixed_layout"),
                query_ids=spec.get("query_ids"),
                execution_only=spec.get("execution_only", False),
                planning_only=spec.get("planning_only", False),
                cache_reset_scope=spec.get("cache_reset_scope", "query"),
                minimum_accuracy=spec.get("minimum_accuracy"),
                layer_pruning=spec.get("layer_pruning", "pareto"),
                execution_cpu_parallelism=settings["execution"]["cpu_parallelism"],
                max_concurrent_requests_per_model=settings["execution"][
                    "max_concurrent_requests_per_model"
                ],
                enforce_model_operator_order=settings["execution"][
                    "enforce_model_operator_order"
                ],
                settings=settings,
                databases=spec["databases"],
                rewrite_relations=False,
                query_order=spec.get("query_order"),
            )
            succeeded = evaluation_succeeded(spec, rows)
            update(
                status="complete" if succeeded else "finished_with_issues",
                current_query=None,
            )
    except KeyboardInterrupt:
        update(status="stopped", error="Test interrupted")
    except Exception as error:
        update(status="error", error=str(error))
        raise
    finally:
        stop_reports.set()
        reporter.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--planning-only", action="store_true")
    parser.add_argument("--reference-teacher-config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    output = (
        (ROOT / args.output).resolve()
        if args.output is not None
        else run / "evaluation"
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "job.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output / "request.json"
        new_request = not path.exists()
        spec = (
            read(path)
            if not new_request
            else request(run, timeout=args.timeout, planning_only=args.planning_only)
        )
        if args.planning_only and spec.get("planning_only") is not True:
            raise ValueError(
                "Scheduled evaluation mode differs; use a new output directory"
            )
        if args.reference_teacher_config is not None:
            reference_file = (ROOT / args.reference_teacher_config).resolve()
            frozen_reference = spec.get("reference_teacher_config")
            if (
                frozen_reference is not None
                and Path(frozen_reference) != reference_file
            ):
                raise ValueError("Scheduled reference teacher differs")
            reference_sha256 = file_digest(reference_file)
            if spec.get("reference_teacher_config_sha256") not in (
                None,
                reference_sha256,
            ):
                raise ValueError("Scheduled reference teacher configuration changed")
            spec["reference_teacher_config"] = str(reference_file)
            spec["reference_teacher_config_sha256"] = reference_sha256
            if new_request:
                spec["reference_output"] = str(reference_directory(reference_file))
        spec.setdefault(
            "reference_output",
            str(
                reference_directory(
                    spec.get(
                        "reference_teacher_config", spec["settings"]["teacher_config"]
                    )
                )
            ),
        )
        spec["query_order"] = [
            item["id"] for item in ordered_heldout_manifest(spec["manifest"])
        ]
        if spec["timeout"] != args.timeout or args.timeout <= 0:
            raise ValueError("Scheduled timeout differs or is not positive")
        save(path, spec)
        if args.prepare_only:
            if not spec.get("planning_only", False):
                Path(spec["reference_output"]).mkdir(parents=True, exist_ok=True)
            render(output, spec, {"status": "waiting_for_training"})
            return
        run_job(run, spec, output=output)


if __name__ == "__main__":
    main()
