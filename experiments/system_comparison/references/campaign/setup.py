"""Collect the selected five-system query outputs with one external teacher."""

from __future__ import annotations

import json
import time
from pathlib import Path

from experiments.core.io import digest, save
from experiments.core.paths import (
    load_translated_manifest,
    translated_manifest_digest,
)
from experiments.core.resources import job_resources
from experiments.system_comparison.data.memory import DATABASE_NAMES
from experiments.system_comparison.references.progress import (
    COUNTERS,
    request_totals,
    write_progress,
)

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
    teacher = teacher_model(config_path, reasoning_effort)
    queries = selected_queries(load_translated_manifest())
    contract = {
        "version": "native-five-system-teacher-v1",
        "contract": "selected19-common-numeric-v4",
        "query_ids": queries,
        "systems": list(SYSTEMS),
        "query_manifest_sha256": translated_manifest_digest(),
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
