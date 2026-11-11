"""Execute frozen training SQLs with a local model and report real cardinalities.

This is a query-design check, not teacher reference generation or an F1 test.
Use the production fine-priority Beam planner with a fixed model and layout.
No training changes or cache resets are performed.
"""

import argparse
import csv
import fcntl
import json
import os
import shutil
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from planning.backends.execution.evaluator import DuckDBEvaluator, read_telemetry
from planning.settings.configuration import (
    ROOT,
    activate_native,
    load_config,
    runtime_path,
)
from planning.settings.model_config import discover_models
from planning.settings.teacher import TeacherModel
from reference_answers.execution.semcache import HeuristicTeacherPolicy

from training.artifacts import save
from training.query_manifest import load_manifest, sql_sha256
from training.query_validation.targets import row_verdict


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


class ValidationPolicy(HeuristicTeacherPolicy):
    def optimize(self, query):
        start = time.monotonic()
        result = super().optimize(query)
        self.planning_seconds = time.monotonic() - start
        return result


def run(
    output, *, workers=4, request_limit=64, query_ids=None, refresh_workloads=False
):
    settings = load_config()
    activate_native(settings)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    guard = (output / "job.lock").open("a")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = load_manifest(ROOT / settings["training"]["workload_manifest"])
    workloads = manifest["workloads"]
    if query_ids:
        unknown = set(query_ids) - {q["id"] for q in workloads}
        if unknown:
            raise ValueError(f"Unknown query IDs: {sorted(unknown)}")
        workloads = [q for q in workloads if q["id"] in query_ids]
    models, _ = discover_models(settings["models"]["endpoints"], max_output_tokens=64)
    model = models["gemma4-12b"]
    selected = {"gemma4-12b": model}
    with urllib.request.urlopen(
        model["base_url"] + "/get_server_info", timeout=10
    ) as response:
        server = json.load(response)
    contract = {
        "version": "full-query-design-validation-v1",
        "workload_version": manifest["version"],
        "queries": {q["id"]: q["sql_sha256"] for q in workloads},
        "model": model,
        "workers": workers,
        "client_total_request_limit": request_limit,
        "per_query_request_limit": request_limit // workers,
        "server_max_running_requests": server["max_running_requests"],
        "cpu_threads_total": 16,
        "plan": "production fine-priority Beam; fixed 12B and original-field OTD",
        "cache": "isolated persistent prompt answers; shared across these validation queries; no resets",
        "minimum_result_rows": 30,
        "target_minimum_result_rows": 50,
        "maximum_result_rows": 300,
    }
    contract_path = output / "contract.json"
    if contract_path.exists():
        previous = json.loads(contract_path.read_text())
        baseline = {k: previous.get(k) for k in contract}
        if baseline != contract:
            if refresh_workloads:
                permitted = {
                    "queries",
                    "workload_version",
                    "minimum_result_rows",
                    "maximum_result_rows",
                    "target_minimum_result_rows",
                }
                if any(
                    previous.get(k) != v
                    for k, v in contract.items()
                    if k not in permitted
                ):
                    raise ValueError(
                        "Refresh may change workload SQL and cardinality targets only"
                    )
                contract.update(
                    workloads_refreshed_at=now(),
                    changed_queries=[
                        q["id"]
                        for q in workloads
                        if previous["queries"].get(q["id"]) != q["sql_sha256"]
                    ],
                )
            else:
                raise ValueError(
                    "Validation inputs changed; choose an empty output directory"
                )
        else:
            contract = {**previous, **contract}
    save(contract_path, contract)
    save(output / "workloads.json", manifest)
    save(output / "server-info.json", server)
    if hasattr(os, "sched_getaffinity"):
        available = sorted(os.sched_getaffinity(0))
        alternate = [c for c in available if c >= 16]
        if alternate:
            os.sched_setaffinity(0, alternate[:16])
    build = runtime_path(settings, "build")
    databases = runtime_path(settings, "source_databases")
    state = {
        "status": "running",
        "pid": os.getpid(),
        "started_at": now(),
        "completed": 0,
        "total": len(workloads),
        "active": {},
        "queries": {},
    }
    for q in workloads:
        saved = output / "queries" / q["id"] / "summary.json"
        if saved.exists():
            value = json.loads(saved.read_text())
            if value["sql_sha256"] != q["sql_sha256"]:
                if not refresh_workloads:
                    raise ValueError("Saved SQL differs")
                continue
            if value.get("status") == "completed":
                value["result_acceptance"] = row_verdict(value["rows"])
            state["queries"][q["id"]] = value
    lock, stop = threading.RLock(), threading.Event()

    def report():
        with lock:
            for identity, active in list(state["active"].items()):
                telemetry = (
                    output / "queries" / identity / "native/run-1/telemetry.jsonl"
                )
                if not telemetry.exists():
                    continue
                with telemetry.open("rb") as stream:
                    stream.seek(active.get("telemetry_offset", 0))
                    while True:
                        before = stream.tell()
                        line = stream.readline()
                        if not line or not line.endswith(b"\n"):
                            active["telemetry_offset"] = before
                            break
                        event = json.loads(line)
                        if event.get("event") == "submit":
                            active["http_requests"] = active.get("http_requests", 0) + 1
            state["updated_at"] = now()
            state["completed"] = len(state["queries"])
            save(output / "report.json", state)
            lines = [
                f"# {len(workloads)}条训练查询 · 12B结果检查",
                "",
                f"状态：{state['status']}；完成 {state['completed']}/{len(workloads)}；{workers}条查询并行，总HTTP并发{request_limit}，CPU总线程16。",
                "完整执行，原SQL输入不截取；固定12B与原字段OTD；后续查询使用已有精细优先级Beam规划。已完成结果保留，各条计划来源记录在summary.json；不同计划的时间不作横向比较。独立prompt结果缓存继续复用。",
                "保留比例=最终结果数/所有语义条件按全通过时的关系候选数，仅报告，不设通过率上限。",
                "12B的零结果不等于teacher已确认无答案。算子通过率是实际执行时的条件通过率，受前序过滤影响。",
                "",
                "目标结果约50～300行；本次只继续修复执行报错和少于30行的查询，30～49行保持不动，不设单个filter通过率上限。",
                "",
                "| 查询 | 状态 | 关系候选 | 实际结果 | 保留比例 | 执行(s) | HTTP请求 | 检查 |",
                "|---|---|---:|---:|---:|---:|---:|---|",
            ]
            for q in workloads:
                r = state["queries"].get(q["id"])
                if q["id"] in state["active"]:
                    active = state["active"][q["id"]]
                    lines.append(
                        f"| {q['id']} | 规划/执行中 | {q['input_coverage']['relational_result_candidates']} | — | — | "
                        f"{time.monotonic() - active['start']:.1f} | {active.get('http_requests', 0)} | |"
                    )
                elif r:
                    ratio = r.get("survival")
                    detail = (
                        r.get("classification", r.get("error", ""))
                        .replace("\n", " ")
                        .replace("|", "\\|")
                    )
                    lines.append(
                        f"| {q['id']} | {r['status']} | {r['candidates']} | {r.get('rows', '—')} | "
                        f"{f'{ratio:.1%}' if ratio is not None else '—'} | {r.get('seconds', 0):.2f} | "
                        f"{r.get('http_requests', 0)} | {detail} |"
                    )
            lines += [
                "",
                "[出错算子明细](出错算子.md)",
                "",
                "逐算子输入行数、输出行数、条件通过率及任务描述保存在每条查询的summary.json；完整ID结果保存在native/run-1/result.csv。",
            ]
            (output / "progress.md").write_text("\n".join(lines) + "\n")
            assessed = [
                r for r in state["queries"].values() if r.get("status") == "completed"
            ]
            notes = [
                "# Filter修复验收",
                "",
                f"状态：{state['status']}；已检查{len(assessed)}/{len(workloads)}条；"
                f"符合约50～300行目标：{sum(row_verdict(r['rows'])['status'] == 'passed' for r in assessed)}条。",
                "",
                "| 查询 | 结果行数 | 验收 |",
                "|---|---:|---|",
            ]
            for r in assessed:
                notes.append(
                    f"| {r['id']} | {r['rows']} | {row_verdict(r['rows'])['classification']} |"
                )
            notes += [
                "",
                "逐查询据实报告结果数；不使用LIMIT、重复记录或修改模型判断凑足范围。",
            ]
            (output / "检查结论.md").write_text("\n".join(notes) + "\n")

    def refresh():
        while not stop.wait(10):
            report()

    def execute(q):
        folder = output / "queries" / q["id"]
        summary = folder / "summary.json"
        if summary.exists():
            saved = json.loads(summary.read_text())
            if saved["sql_sha256"] != q["sql_sha256"]:
                if not refresh_workloads:
                    raise ValueError("Saved SQL differs")
            elif saved["status"] == "completed":
                saved.pop("filter_selectivity", None)
                saved["result_acceptance"] = row_verdict(saved["rows"])
                saved["classification"] = row_verdict(saved["rows"])["classification"]
                save(summary, saved)
                return saved
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        with lock:
            state["active"][q["id"]] = {"start": time.monotonic()}
        report()
        policy = ValidationPolicy(
            TeacherModel("gemma4-12b", model),
            output=folder / "planning",
            build=build,
            database=databases[q["dataset"]],
            settings={
                "runtime_configuration": settings,
                "cpu_parallelism": max(1, 16 // workers),
                "response_cache_directory": str(output / "prompt-cache"),
                "cache_enabled": True,
            },
        )
        r = {
            "id": q["id"],
            "sql_sha256": q["sql_sha256"],
            "plan_generation": "heuristic_requests",
            "candidates": q["input_coverage"]["relational_result_candidates"],
        }
        started = time.monotonic()
        try:
            assert sql_sha256(q["sql"]) == q["sql_sha256"]
            with DuckDBEvaluator(
                build,
                selected,
                q["sql"],
                output=folder / "native",
                database=databases[q["dataset"]],
                timeout=1800,
                parallelism=max(1, 16 // workers),
                max_concurrent_requests_per_model=request_limit // workers,
            ) as evaluator:
                evaluator.response_cache.update(
                    directory=str(output / "prompt-cache"), verify_on_second_use=False
                )
                result = evaluator.run(
                    policy_override=policy, include_rows=False, allow_timeout=True
                )
            save(folder / "query_ir.json", policy.query)
            events = result["events"]
            operators = {
                op["operator_id"]: op for op in policy.query["semantic_operators"]
            }
            summaries = [e for e in events if e.get("event") == "operator_summary"]
            execution = next(
                (e for e in reversed(events) if e.get("event") == "execution_report"),
                {},
            )
            r.update(
                status=result["status"],
                seconds=(
                    execution.get("execution_wall_ms", 0) / 1000
                    or result.get("seconds")
                    or result.get("time_lower_bound")
                ),
                planning_seconds=getattr(policy, "planning_seconds", None),
                http_requests=sum(e.get("event") == "submit" for e in events),
                operators=[
                    {
                        **e,
                        "task": operators[e["operator_id"]]["task"],
                        "kind": operators[e["operator_id"]]["kind"],
                    }
                    for e in summaries
                ],
            )
            if result["status"] == "completed":
                with (Path(result["artifact"]) / "result.csv").open(newline="") as f:
                    reader = csv.reader(f)
                    r["columns"] = next(reader)
                    r["rows"] = sum(1 for _ in reader)
                r["survival"] = r["rows"] / r["candidates"] if r["candidates"] else None
                r["classification"] = row_verdict(r["rows"])["classification"]
                r["result_acceptance"] = row_verdict(r["rows"])
            else:
                r["classification"] = "超时，未判断是否为空"
        except Exception as error:
            events = read_telemetry(folder / "native/run-1/telemetry.jsonl", retries=0)
            r.update(
                status="failed",
                error=str(error),
                seconds=time.monotonic() - started,
                http_requests=sum(e.get("event") == "submit" for e in events),
            )
        save(summary, r)
        with lock:
            state["active"].pop(q["id"], None)
        return r

    reporter = threading.Thread(target=refresh, daemon=True)
    reporter.start()
    try:
        report()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(execute, q): q for q in workloads}
            for future in as_completed(futures):
                r = future.result()
                with lock:
                    state["queries"][r["id"]] = r
                print(
                    r["id"],
                    r["status"],
                    r.get("rows"),
                    r.get("classification", r.get("error")),
                    flush=True,
                )
                report()
        state["status"] = (
            "completed"
            if all(r["status"] == "completed" for r in state["queries"].values())
            else "finished_with_issues"
        )
    finally:
        stop.set()
        reporter.join(timeout=2)
        report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query", action="append")
    parser.add_argument("--refresh-workloads", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--request-limit", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16 or not args.workers <= args.request_limit <= 128:
        parser.error("Require 1..16 workers and workers..128 total request concurrency")
    run(
        args.output,
        query_ids=args.query,
        workers=args.workers,
        request_limit=args.request_limit,
        refresh_workloads=args.refresh_workloads,
    )


if __name__ == "__main__":
    main()
