"""Resume public-test references independently of candidate evaluation."""

import argparse
import fcntl
import hashlib
import os
import threading
from datetime import datetime
from pathlib import Path

from planning.settings.configuration import activate_native, runtime_path
from planning.settings.parallelism import configure_cpu_resources
from planning.settings.teacher import default_teacher_model, load_teacher
from reference_answers.storage.files import file_digest

from training.artifacts import read, save
from training.query_manifest import heldout_manifest, ordered_heldout_manifest

from .evaluation import policy, reference


def run(spec):
    from experiments.semcache.monitoring.evaluation_report import render_references

    source = spec.get("reference_evaluation_run") or spec["settings"]["inference"].get(
        "reference_evaluation_run"
    )
    if source:
        from .saved_evaluation import SavedEvaluationReferences

        saved = SavedEvaluationReferences(
            source,
            spec["manifest"],
            spec["databases"],
            aggregate_manifest=spec.get("reference_aggregate_manifest")
            or spec["settings"]["inference"].get("reference_aggregate_manifest"),
        )
        if (
            isinstance(spec.get("reference_policy"), dict)
            and saved.policy != spec["reference_policy"]
        ):
            raise ValueError("Scheduled saved reference evidence changed")
        print(
            f"Reused {len(saved.references)} saved references; no execution or teacher requests",
            flush=True,
        )
        return
    if spec["manifest"] != heldout_manifest():
        raise ValueError("Frozen reference manifest differs from the heldout suite")
    output = Path(spec["reference_output"])
    settings = spec["settings"]
    activate_native(settings)
    configure_cpu_resources(settings)
    teacher_file = Path(spec["reference_teacher_config"])
    if file_digest(teacher_file) != spec["reference_teacher_config_sha256"]:
        raise ValueError("Frozen reference teacher configuration changed")
    teacher = load_teacher(default_teacher_model(teacher_file), {}, teacher_file)
    selected_datasets = set(spec.get("selected_datasets", ()))
    selected_ids = set(spec.get("selected_ids", ()))
    manifest_items = [
        item for item in ordered_heldout_manifest(spec["manifest"])
        if not selected_datasets
        or {"movies": "movies_nirvana"}.get(Path(item["path"]).parent.name, Path(item["path"]).parent.name)
        in selected_datasets
    ]
    manifest_items = [
        item for item in manifest_items
        if not selected_ids or item["id"] in selected_ids
    ]
    expected = {
        "version": 1,
        "queries": [
            {"id": item["id"], "sql_sha256": item["sql_sha256"]}
            for item in spec["manifest"]
        ],
        "database_sha256": spec["database_sha256"],
        "policy": policy(teacher),
    }
    contract_file = output / "contract.json"
    if contract_file.exists() and read(contract_file) != expected:
        raise ValueError("Independent reference contract changed")
    for name, database in spec["databases"].items():
        if file_digest(database) != spec["database_sha256"][name]:
            raise ValueError(f"Frozen reference database changed: {name}")
    if not contract_file.exists():
        save(contract_file, expected)
    state = {
        "status": "running",
        "pid": os.getpid(),
        "mode": "references_only",
        "reference_timeout_seconds": spec.get("reference_timeout_seconds"),
        "selected_datasets": sorted(selected_datasets) if selected_datasets else None,
        "selected_ids": sorted(selected_ids) if selected_ids else None,
        "total": len(manifest_items),
        "completed": 0,
    }
    lock, stopping = threading.RLock(), threading.Event()

    def update(**values):
        with lock:
            state.update(values, updated_at=datetime.now().astimezone().isoformat())
            save(output / "job-status.json", state)
            render_references(spec, state)

    def refresh():
        while not stopping.wait(10):
            update()

    reporter = threading.Thread(target=refresh, daemon=True)
    reporter.start()
    issues = 0
    try:
        update()
        for item in manifest_items:
            key, sql_file = item["id"], Path(item["path"])
            sql = sql_file.read_text()
            if hashlib.sha256(sql.encode()).hexdigest() != item["sql_sha256"]:
                raise ValueError(f"Frozen reference SQL changed: {key}")
            dataset = {"movies": "movies_nirvana"}.get(
                sql_file.parent.name, sql_file.parent.name
            )
            update(current_query=key, current_stage="参考执行")
            try:
                result, reused = reference(
                    output,
                    key,
                    sql,
                    spec["databases"][dataset],
                    teacher,
                    runtime_path(settings, "build"),
                    settings,
                    spec.get("reference_timeout_seconds"),
                )
                issues += result["status"] != "completed"
                update(
                    last_result={
                        "query": key,
                        "status": result["status"],
                        "reused": reused,
                    }
                )
                print(
                    f"Reference {key}: {result['status']}, reused={reused}", flush=True
                )
            except (OSError, ValueError, KeyError, RuntimeError, TimeoutError) as error:
                issues += 1
                update(
                    last_result={"query": key, "status": "failed", "error": str(error)}
                )
                print(f"Reference {key}: failed: {error}", flush=True)
        update(
            status="finished_with_issues" if issues else "complete",
            current_query=None,
            current_stage=None,
        )
    except KeyboardInterrupt:
        update(status="stopped")
        raise
    except Exception as error:
        update(status="error", error=str(error))
        raise
    finally:
        stopping.set()
        reporter.join(timeout=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args()
    spec = read(args.evaluation / "request.json")
    if spec.get("reference_evaluation_run") or spec["settings"]["inference"].get(
        "reference_evaluation_run"
    ):
        run(spec)
        return
    output = Path(spec["reference_output"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "job.lock").open("a") as task_lock:
        fcntl.flock(task_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(spec)


if __name__ == "__main__":
    main()
