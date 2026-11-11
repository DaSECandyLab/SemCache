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
from experiments.core.paths import ROOT
from experiments.core.process import process
from experiments.core.resources import configure_open_file_limit, job_resources
from experiments.system_comparison.data.memory import DATABASE_NAMES, bind_database
from experiments.system_comparison.references.progress import (
    COUNTERS,
    request_totals,
    write_progress,
)
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

SYSTEMS = ("plop", "sema", "nirvana", "lotus", "abacus")
DATASETS = ("estate", "movies", "movies_sema", "appstore", "tpch")


def read(path):
    return json.loads(Path(path).read_text())


def teacher_model(path, reasoning_effort="none"):
    settings = read(path)
    base = settings["base_url"].rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    overrides = {
        "thinking": {"type": "disabled" if reasoning_effort == "none" else "enabled"},
        "max_tokens": 4096,
    }
    if reasoning_effort != "none":
        overrides["reasoning_effort"] = reasoning_effort
    elif settings.get("reasoning_effort") == "none":
        overrides["reasoning_effort"] = "none"
    return {
        "alias": "semantic-large",
        "id": settings["model"],
        "endpoint": base,
        "context_length": settings.get("context_length", 131072),
        "reasoning_effort": reasoning_effort,
        "api_key": settings["api_key"],
        "headers": settings.get("headers", {}),
        "request_overrides": overrides,
        "disable_proxy": True,
        # WorkBuddy may report the resolved backend ID rather than its auto alias.
        # The complete provider-reported model is retained in gateway responses.
        "allow_model_alias": True,
        "incomplete_output_retries": 2,
        "max_retry_output_tokens": 16384,
        "rate_limit_retries": 8,
        "network_recovery_retries": 8,
    }


def selected_queries(manifest):
    queries = {r["id"] for r in manifest["records"] if r["system"] == "semcache"}
    if len(queries) != manifest["query_count_per_system"]["semcache"]:
        raise ValueError("Selected query count differs from the manifest")
    return sorted(
        queries, key=lambda q: (DATASETS.index(q.split("/")[0]), int(q.split("/Q")[1]))
    )


def skipped(system, query):
    return system == "abacus" and query.split("/")[0] in {"appstore", "tpch"}


def plop_runtime(path):
    path = Path(path).resolve()
    profile = read(path)
    if profile["vector_size"] != 128 or not profile["chunk_merging"]:
        raise ValueError("Expected a verified vector128 PLOP runtime")
    inputs = profile["databases"]
    files = [
        (profile["binary"], profile["binary_sha256"]),
        (profile["data_manifest"], profile["data_manifest_sha256"]),
    ]
    for source in inputs.values():
        files.append((source["path"], source["sha256"]))
        if not source.get("source_retired"):
            files.append((source["source_path"], source["source_sha256"]))
    for name, expected in files:
        if digest(name) != expected:
            raise ValueError("PLOP runtime file changed: " + name)
    return {
        "profile": str(path),
        "profile_sha256": digest(path),
        "binary": profile["binary"],
        "binary_sha256": profile["binary_sha256"],
        "vector_size": profile["vector_size"],
        "chunk_merging": True,
        "databases": inputs,
    }


def bind_plop_runtime(job, runtime):
    if job["system"] != "plop" or runtime is None:
        return job
    dataset = DATABASE_NAMES.get(job["dataset"], job["dataset"])
    source = runtime["databases"][dataset]
    if job["input_database_sha256"] not in {source["source_sha256"], source["sha256"]}:
        raise ValueError("PLOP runtime database has a different frozen source")
    return {
        **job,
        "input_database": source["path"],
        "input_database_sha256": source["sha256"],
        "input_databases": {
            **job["input_databases"],
            "plop": {"path": source["path"], "sha256": source["sha256"]},
        },
        "original_input_database": source["source_path"],
        "original_input_database_sha256": source["source_sha256"],
        "plop_runtime": {k: v for k, v in runtime.items() if k != "databases"},
    }


def initialize(
    run,
    config_path,
    port,
    reasoning_effort="none",
    batch_size=1,
    *,
    cache_enabled=False,
    no_timeouts=False,
    allow_runtime_update=False,
    request_timeout=3600,
    timeout=None,
    plop_runtime_path=None,
):
    job_resources(batch_size)
    if timeout is not None:
        if no_timeouts:
            raise ValueError("Cannot combine a unified timeout with no timeouts")
        request_timeout = timeout
    if not 60 <= request_timeout <= 2147483:
        raise ValueError("Request timeout must be between 60 and 2147483 seconds")
    query_timeout = timeout if timeout is not None else (None if no_timeouts else 1800)
    search_timeout = timeout if timeout is not None else (None if no_timeouts else 100)
    manifest_path = ROOT / "benchmarks/queries/translated/manifest.json"
    teacher = teacher_model(config_path, reasoning_effort)
    queries = selected_queries(read(manifest_path))
    contract = {
        "version": "native-five-system-teacher-v1",
        "contract": "selected19-common-numeric-v4",
        "query_ids": queries,
        "systems": list(SYSTEMS),
        "query_manifest_sha256": digest(manifest_path),
        "teacher": {
            k: teacher[k]
            for k in ("id", "endpoint", "context_length", "reasoning_effort")
        },
        "teacher_config": str(config_path),
        "teacher_config_sha256": digest(config_path),
        "input_loading": "database_memory",
        "batch_size": batch_size,
        "batch_semantics": {
            "plop": "one row per native prompt; no prompt batching option",
            "sema": f"at most {batch_size} rows per native batch prompt",
            "nirvana": f"join chunk size {batch_size}; native map/filter prompts process one row",
            "lotus": f"retrieval batch size {batch_size}; native LM prompts process one row",
            "abacus": f"operator queue batch size {batch_size}; native prompts process one row",
        },
        "native_concurrency": 16,
        "teacher_requests_per_active_system": 16,
        "gateway_concurrency_limit": 16,
        "reasoning_effort": reasoning_effort,
        "teacher_max_output_tokens": 4096,
        "teacher_incomplete_output_retries": 2,
        "teacher_max_retry_output_tokens": 16384,
        "teacher_rate_limit_retries": 8,
        "teacher_network_recovery": {
            "retries": 8,
            "backoff_seconds": [1, 2, 4, 8, 15, 15, 15, 15],
            "scope": "network down/unreachable or host unreachable; retry identical request",
        },
        "query_timeout_seconds": query_timeout,
        "request_timeout_seconds": request_timeout,
        "abacus_search_seconds": search_timeout,
        "native_transport_limits": {
            "plop_http_seconds": request_timeout,
            "sema_http_seconds": request_timeout,
        },
        "gateway_port": port,
        "scheduled_queries": 130,
        "skipped_queries": ["abacus/" + q for q in queries if skipped("abacus", q)],
        "schedule": "all five systems sequentially; one query at a time; plop, sema, nirvana, lotus, abacus",
        "cache_enabled": cache_enabled,
        "cache": "independent persistent cache per system; exact endpoint/model/request keys; no cross-system reuse"
        if cache_enabled
        else "fresh native process per query; no collector response cache; no provider KV-cache control",
        "verification": "first occurrence stores one response; second independently verifies; on mismatch one final vote; latest wins three-way tie; later hits reuse"
        if cache_enabled
        else "one generation; no secondary teacher verification or majority vote",
        "prompts": "native system prompts; no PLOP prompt rewrite",
        "sema_response_serialization": "sema-native-json-string-v1 plus sema-native-json-boolean-v1; quote strings and serialize standalone true/false/yes/no; original completion retained",
        "sema_response_format_retry": "HTTP 400 string/boolean response_format unavailable: retry once without upstream format; native prompts unchanged; downstream JSON type validated",
        "lotus": "single native teacher LM; no fictitious four-model filter cascade; native retrieval join optimization retained",
        "abacus": "native MinTime with single teacher, native MAB; no external search deadline"
        if search_timeout is None
        else f"native MinTime with single teacher, native MAB and external {search_timeout}s search fallback",
        "aggregate_capture": "materialize SQL aggregate input once; observe Python host aggregate input; do not rerun semantic operators",
        "interpretation": "independent system-specific teacher-generated candidates, not independently verified ground truth",
    }
    if plop_runtime_path is not None:
        contract["plop_runtime"] = plop_runtime(plop_runtime_path)
    run.mkdir(parents=True, exist_ok=True)
    previous = run / "contract.json"
    if previous.exists():
        if read(previous) != contract:
            if (run / "controller.json").exists():
                old = read(previous)
                allowed = {
                    "cache",
                    "cache_enabled",
                    "verification",
                    "query_timeout_seconds",
                    "request_timeout_seconds",
                    "abacus",
                    "abacus_search_seconds",
                    "native_transport_limits",
                    "plop_runtime",
                    "sema_response_format_retry",
                    "sema_response_serialization",
                    "teacher_network_recovery",
                }
                if not allow_runtime_update or {
                    k: v for k, v in old.items() if k not in allowed
                } != {k: v for k, v in contract.items() if k not in allowed}:
                    raise ValueError(
                        "Frozen teacher, queries or execution settings changed; use a new directory"
                    )
                with (run / "runtime-changes.jsonl").open("a") as log:
                    log.write(
                        json.dumps(
                            {
                                "time_unix": time.time(),
                                "before": {k: old.get(k) for k in allowed},
                                "after": {k: contract.get(k) for k in allowed},
                            }
                        )
                        + "\n"
                    )
            save(previous, contract)
    else:
        save(previous, contract)
    return contract, teacher


def render(run, contract):
    complete = failed = 0
    lines = [
        "# 五系统 Teacher 答案收集",
        "",
        "每 10 秒刷新。单位：秒；未产出结果显示 `-`。错误见文末。",
        "",
        f"Teacher：{Path(contract['teacher_config']).stem.removeprefix('teacher.')} / {contract['teacher']['id']}；思考 {contract['reasoning_effort']}，请求并发16、原生 batch_size={contract['batch_size']}。五系统依次执行，每次只运行一条查询。",
        "",
        "下表耗时包含答案执行与中间结果保存，不用于替代正式性能对比。",
        f"Sema 提示每批最多 {contract['batch_size']} 行；PLOP/LOTUS 及 Nirvana Map/Filter 仍逐行请求。Abacus batch 为队列粒度、Nirvana Join batch 为分块粒度，均不在网关合并提示。",
        "",
        "| 查询 | PLOP | Sema | Nirvana | LOTUS | Abacus |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    errors = []
    active = []
    for query in contract["query_ids"]:
        cells = []
        for system in SYSTEMS:
            folder = run / system / query
            summary = (
                read(folder / "summary.json")
                if (folder / "summary.json").exists()
                else {}
            )
            if summary.get("parent_finalized"):
                if summary["status"] == "completed":
                    complete += 1
                    cells.append(f"{summary['seconds']:.2f}")
                else:
                    if summary["status"] != "skipped":
                        failed += 1
                        reason = (
                            summary.get("reason", summary["status"])
                            .replace("|", "\\|")
                            .replace("\n", " ")
                        )
                        errors.append(
                            f"| {system}/{query} | {reason[:500]} | [日志]({folder / 'stderr.log'}) |"
                        )
                    cells.append("-")
            else:
                cells.append("-")
        lines.append(
            "| "
            + query.replace("movies/", "movies_nirvana/")
            + " | "
            + " | ".join(cells)
            + " |"
        )
    for system in SYSTEMS:
        status_path = run / "status" / (system + ".json")
        if status_path.exists():
            status = read(status_path)
            if status.get("phase") == "running":
                active.append(
                    f"{system}/{status['query']}: {status.get('elapsed_seconds', 0):.0f} 秒，模型请求 {status.get('model_requests', 0)}"
                )
    lines[2:2] = [
        f"已完成 **{complete}/130**，失败/超时 **{failed}**，明确跳过 **10**。",
        "",
    ]
    if active:
        lines += ["", "当前执行：" + "；".join(active)]
    restart_path = run / "restart-request.json"
    if restart_path.exists():
        restart = read(restart_path)
        if restart.get("phase") == "pending":
            trigger = (
                f"{restart['after_system']} 本轮全部结束后"
                if restart.get("after_system")
                else "当前查询结束后"
            )
            lines += [
                "",
                f"已安排：{trigger}，仅补跑 {', '.join(restart['queries'])}，保留缓存；完成后继续后续系统。",
            ]
    if contract.get("cache_enabled"):
        lines += [
            "",
            "缓存：五系统各自独立，不跨系统共享。首次保存，第二次独立核对；不一致时最后投票一次，三次均不同则记录平票并取第三次。之后命中直接复用。",
            "统计包含已结束查询、正在执行查询和历史中断记录；HTTP 次数在请求返回或报错后入账，历史中断未记录终态的请求不计。",
            "",
            "| 系统 | 实际 HTTP 请求 | 缓存命中 | 二次核对 | 第三次投票 |",
            "|---|---:|---:|---:|---:|",
        ]
        for system in SYSTEMS:
            totals = request_totals(run, system)
            lines.append(
                "| "
                + system
                + " | "
                + " | ".join(str(totals[key]) for key in COUNTERS)
                + " |"
            )
    query_limit = contract["query_timeout_seconds"]
    search_limit = contract.get("abacus_search_seconds", 100)
    lines += [
        "",
        (
            f"超时：整条查询 {str(query_limit) + ' 秒' if query_limit is not None else '不设总时限'}；"
            f"Abacus 每次外部规划搜索 {str(search_limit) + ' 秒' if search_limit is not None else '不设总时限'}；"
            f"单次 HTTP/SDK 请求统一 {contract['request_timeout_seconds']} 秒。"
            "PLOP 使用原有环境变量，Sema 沿用实验传输副本配置，系统源码保持不变。"
        ),
    ]
    if contract.get("plop_runtime"):
        lines += [
            "",
            (
                "PLOP 后续查询使用向量大小128的构建，启用满足原生条件时的小块合并；"
                "请求并发16、每请求一行保持不变。已完成条目保留原构建及答案，"
                "各任务 native-config.json 记录实际二进制和向量设置。"
            ),
        ]
    lines += [
        "",
        "Abacus 的 Appstore/TPC-H 按要求跳过。每系统的结果、聚合前输入、计划及逐请求日志分别保存在对应子目录。",
    ]
    if errors:
        lines += ["", "| 查询 | 原因 | 日志 |", "|---|---|---|", *errors]
    text = "\n".join(lines) + "\n"
    write_progress(
        run,
        text,
        {
            "completed": complete,
            "failed": failed,
            "skipped": 10,
            "scheduled": 130,
            "active": active,
            "updated_unix": time.time(),
        },
    )


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
