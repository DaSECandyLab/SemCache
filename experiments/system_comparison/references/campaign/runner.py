"""Collect the selected five-system query outputs with one external teacher."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import threading
import time
import traceback
from pathlib import Path

from experiments.core.io import digest, save
from experiments.core.paths import (
    ROOT,
)
from experiments.core.process import process
from experiments.core.resources import configure_open_file_limit, job_resources
from experiments.system_comparison.data.memory import bind_database
from experiments.system_comparison.references.protocol import (
    complete_response,
    lotus_boolean_response,
    sema_string_response,
)
from experiments.system_comparison.scheduling.campaign import (
    cleanup_owned_processes,
    prepare,
    worker_command,
)
from experiments.system_comparison.scheduling.parallel import (
    allocate_cpu_sets,
    format_cpu_set,
)
from experiments.system_comparison.transport.gateway import PoolGateway

from .setup import (
    bind_plop_runtime,
    initialize,
    read,
    render,
    skipped,
)

SYSTEMS = ("plop", "sema", "nirvana", "lotus", "abacus")
DATASETS = ("estate", "movies", "movies_sema", "appstore", "tpch")


def execute_system(
    run, system, contract, frozen, gateway, cpu_set, response_cache=None
):
    if response_cache is not None and response_cache.system != system:
        raise ValueError("Teacher caches must not be shared between systems")
    for query_id in contract["query_ids"]:
        folder = run / system / query_id
        summary_path = folder / "summary.json"
        if summary_path.exists() and read(summary_path).get("parent_finalized"):
            continue
        folder.mkdir(parents=True, exist_ok=True)
        if skipped(system, query_id):
            save(
                summary_path,
                {
                    "status": "skipped",
                    "reason": "User excluded previously timing-out Abacus Appstore/TPC-H",
                    "parent_finalized": True,
                },
            )
            continue
        query = frozen["query_files"][system + "/" + query_id]
        if digest(query["path"]) != query["sha256"]:
            raise ValueError("Frozen native query changed: " + query["path"])
        token = system + "-" + query_id.replace("/", "-")
        endpoint = gateway.register(
            token,
            folder,
            allowed=["semantic-large"],
            aliases={
                "gpt-5-mini": "semantic-large",
                "gpt-5-mini-2025-08-07": "semantic-large",
            },
            default=system == "plop",
            upstream_concurrency_limit=16,
            response_transform=sema_string_response
            if system == "sema"
            else lotus_boolean_response
            if system == "lotus"
            else complete_response,
            response_cache=response_cache,
            retry_unavailable_format=system == "sema",
        )
        model = {
            **contract["teacher"],
            "alias": "semantic-large",
            "upstream_endpoint": contract["teacher"]["endpoint"],
            "endpoint": endpoint,
        }
        job = bind_database(
            {
                **frozen["jobs"][query_id],
                **job_resources(contract["batch_size"]),
                "system": system,
                "variant": system + "-teacher",
                "single_model": True,
                "output": str(folder),
                "query_file": query["path"],
                "query_sha256": query["sha256"],
                "models": [model],
                "model": "semantic-large",
                "teacher_model": "semantic-large",
                "endpoint": endpoint,
                "timeout": contract["query_timeout_seconds"],
                "request_timeout": contract["request_timeout_seconds"],
                "max_tokens": 4096,
                "policy": "MinTime",
                "native_mechanisms": system in {"abacus", "nirvana", "lotus"},
                "native_mechanism_profile": "single-teacher-native-v1",
                "optimizer": {
                    "do_logical_optimization": True,
                    "do_physical_optimization": True,
                    "available_models": ["semantic-large"],
                    "max_rounds": 5,
                    "sample_size": 5,
                },
                "model_costs": {
                    "semantic-large": {"Input": 1, "Output": 1, "Cache": 1}
                },
                "abacus_optimizer": {
                    "sample_budget": 32,
                    "k": 4,
                    "j": 4,
                    "judge_model": "semantic-large",
                    "seed": 42,
                    "sampling_enabled": True,
                    "max_search_seconds": contract.get("abacus_search_seconds", 100),
                },
                "embedding": {
                    "backend": "sentence_transformers",
                    "model": str(ROOT / "sglang/models/all-MiniLM-L6-v2"),
                    "device": "cpu",
                },
                "cpu_set": cpu_set,
                "gateway_port": contract["gateway_port"],
            }
        )
        job["teacher_cache"] = {
            "enabled": response_cache is not None,
            "scope": system,
            "directory": str(response_cache.directory)
            if response_cache is not None
            else None,
        }
        job = bind_plop_runtime(job, contract.get("plop_runtime"))
        from experiments.semcache.reference_answers.sql.aggregate import capture_plan

        canonical = (
            ROOT / "benchmarks/queries/translated/semcache" / (query_id + ".sql")
        )
        capture = capture_plan(canonical.read_text())
        if capture:
            import duckdb

            from experiments.system_comparison.data.inputs import output_schema

            with duckdb.connect(
                frozen["jobs"][query_id]["input_databases"]["semcache"]["path"],
                read_only=True,
            ) as connection:
                job["pre_aggregate_schema"] = output_schema(
                    connection, capture["input_sql"]
                )
            if system in {"plop", "sema"}:
                job["aggregate_capture"] = capture_plan(Path(query["path"]).read_text())
        save(folder / "job.json", job)
        command, env = worker_command(system, folder / "job.json", cpu_set=cpu_set)
        if system == "plop":
            env["PLOP_HTTP_TIMEOUT_MS"] = str(
                contract["request_timeout_seconds"] * 1000
            )
        command[command.index("experiments.system_comparison.execution.worker")] = (
            "experiments.system_comparison.references.worker"
        )
        save(folder / "launch.json", {"command": command, "started_unix": time.time()})
        print("START", system, query_id, flush=True)

        def heartbeat(elapsed, pid, query_id=query_id, folder=folder):
            save(
                run / "status" / (system + ".json"),
                {
                    "phase": "running",
                    "query": query_id,
                    "elapsed_seconds": elapsed,
                    "worker_pid": pid,
                    **gateway.summary(folder),
                },
            )

        stats = {}
        try:
            stats = process(
                command,
                folder,
                contract["query_timeout_seconds"],
                env=env,
                heartbeat=heartbeat,
            )
            native = read(summary_path) if summary_path.exists() else {}
            result = {**stats, **native, "wall_seconds": stats["seconds"]}
            if stats["status"] == "timeout":
                result.update(
                    status="timeout",
                    reason=f"Whole-query {stats['timeout_seconds']:g} second deadline",
                )
            elif not native:
                result.update(
                    status="failed", reason=(folder / "stderr.log").read_text()[-2500:]
                )
        except Exception as error:  # noqa: BLE001 -- isolate individual native-query failures
            (folder / "controller-error.txt").write_text(traceback.format_exc())
            result = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
        finally:
            gateway.close_run(token)
            save(
                folder / "process-cleanup.json",
                {"terminated_pids": cleanup_owned_processes(folder)},
            )
        deadline = (
            None
            if contract["request_timeout_seconds"] is None
            else time.monotonic() + contract["request_timeout_seconds"] + 10
        )
        while gateway.pending(token) and (
            deadline is None or time.monotonic() < deadline
        ):
            time.sleep(0.2)
        result.update(gateway.summary(folder))
        if result.get("status") == "completed" and (
            result.get("model_errors") or result.get("response_delivery_errors")
        ):
            result.update(
                status="partial", reason="Transport errors; inspect gateway.jsonl"
            )
        if (
            result.get("status") == "completed"
            and capture
            and not (folder / "pre-aggregate-result.json").exists()
        ):
            result.update(
                status="failed", reason="Required aggregate input snapshot missing"
            )
        result.update(parent_finalized=True, query_id=query_id, system=system)
        save(summary_path, result)
        if result["status"] == "completed":
            answer = {
                "system": system,
                "query_id": query_id,
                "teacher": contract["teacher"],
                "status": "teacher_generated",
                "verified_ground_truth": False,
                "query_sha256": query["sha256"],
                "database_sha256": job["input_database_sha256"],
                "result": read(folder / "result.json"),
                "result_sha256": digest(folder / "result.json"),
                "seconds": result["seconds"],
                "model_requests": result["model_requests"],
                "teacher_http_requests": result["upstream_http_attempts"],
                "cache_hits": result.get("cache_hits", 0),
            }
            pre = folder / "pre-aggregate-result.json"
            if pre.exists():
                answer["pre_aggregate"] = {"path": str(pre), "sha256": digest(pre)}
            if job.get("plop_runtime"):
                answer["plop_runtime"] = job["plop_runtime"]
                answer["original_database_sha256"] = job[
                    "original_input_database_sha256"
                ]
            save(folder / "answer.json", answer)
        save(
            run / "status" / (system + ".json"),
            {"phase": "between_queries", "query": query_id},
        )
        print(
            "DONE",
            system,
            query_id,
            result["status"],
            result.get("seconds"),
            flush=True,
        )
    save(run / "status" / (system + ".json"), {"phase": "finished"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--teacher-config",
        type=Path,
        default=ROOT / "SemCache/config/teacher/teacher.workbuddy.json",
    )
    parser.add_argument(
        "--gateway-port", type=int, default=8902, choices=[8900, 8901, 8902]
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--reasoning-effort", choices=["none", "low"], default="none")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cache", action="store_true")
    deadlines = parser.add_mutually_exclusive_group()
    deadlines.add_argument(
        "--no-timeouts",
        action="store_true",
        help="Disable whole-query and external-search deadlines",
    )
    deadlines.add_argument(
        "--timeout",
        type=int,
        help="Set query, request and external-search timeouts to this many seconds",
    )
    parser.add_argument("--request-timeout", type=int, default=3600)
    parser.add_argument("--allow-runtime-update", action="store_true")
    parser.add_argument("--plop-runtime", type=Path)
    parser.add_argument("--retry-system", choices=SYSTEMS, action="append")
    parser.add_argument("--retry-query", action="append")
    args = parser.parse_args()
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    with (run / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        configure_open_file_limit()
        contract, teacher = initialize(
            run,
            args.teacher_config.resolve(),
            args.gateway_port,
            args.reasoning_effort,
            args.batch_size,
            cache_enabled=args.cache,
            no_timeouts=args.no_timeouts,
            allow_runtime_update=args.allow_runtime_update,
            request_timeout=args.request_timeout,
            timeout=args.timeout,
            plop_runtime_path=args.plop_runtime,
        )
        frozen = prepare(run, contract)
        from experiments.system_comparison.references.runtime import prepare_runtime

        render(run, contract)
        if args.prepare_only:
            print(json.dumps({"run": str(run), "scheduled": 130, "skipped": 10}))
            return
        caches = prepare_runtime(
            run,
            contract,
            teacher,
            retry_systems=args.retry_system,
            retry_queries=args.retry_query,
        )
        save(run / "controller.json", {"pid": os.getpid(), "started_unix": time.time()})
        stop = threading.Event()

        def monitor():
            while not stop.wait(10):
                render(run, contract)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        cpu_sets = [format_cpu_set(cpus) for cpus in allocate_cpu_sets(count=1)]
        save(
            run / "cpu-sets.json",
            {
                "plop": cpu_sets[0],
                "sema": cpu_sets[0],
                "remaining_systems": cpu_sets[0],
            },
        )
        try:
            with PoolGateway(
                [teacher],
                port=args.gateway_port,
                request_timeout=contract["request_timeout_seconds"],
                upstream_concurrency_limit=16,
            ) as gateway:
                for system in SYSTEMS:
                    execute_system(
                        run,
                        system,
                        contract,
                        frozen,
                        gateway,
                        cpu_sets[0],
                        caches.get(system),
                    )
            save(
                run / "controller.json",
                {"pid": os.getpid(), "phase": "finished", "finished_unix": time.time()},
            )
        except BaseException:
            (run / "controller-error.txt").write_text(traceback.format_exc())
            raise
        finally:
            stop.set()
            thread.join(timeout=10)
            render(run, contract)
            for cache in caches.values():
                cache.close()


if __name__ == "__main__":
    main()
