"""Map observed execution failures to their actual bound semantic operators."""

import json
import re
from pathlib import Path


def error_operators(root, query, result):
    folder = Path(root) / "queries" / query["id"]
    query_ir = folder / "planning/query_ir.json"
    if not query_ir.exists():
        query_ir = folder / "query_ir.json"
    bound = json.loads(query_ir.read_text()) if query_ir.exists() else {}
    by_id = {op["operator_id"]: op for op in bound.get("semantic_operators", [])}
    details = []
    telemetry = folder / "native/run-1/telemetry.jsonl"
    if telemetry.exists():
        for line in telemetry.read_bytes().splitlines(keepends=True):
            if not line.endswith(b"\n"):
                continue
            event = json.loads(line)
            if not (
                event.get("event") == "error"
                or event.get("event") == "semantic_row"
                and event.get("parse_status") == "failed"
            ):
                continue
            op = by_id.get(event.get("operator_id"))
            if op is None:
                continue
            error = event.get("error", result.get("error", ""))
            failure_role = "连带中止" if "query already failed" in error else "原始报错"
            fields = next(
                (
                    task["input_fields"]
                    for task in query.get("semantic_tasks", [])
                    if task["task"] == op["task"]
                ),
                [],
            )
            details.append(
                {
                    "query": query["id"],
                    "phase": "完整执行",
                    "operator_id": op["operator_id"],
                    "kind": op["kind"],
                    "task": op["task"],
                    "fields": fields,
                    "request_seq": event.get("request_seq"),
                    "parse_status": event.get("parse_status"),
                    "error": error,
                    "failure_role": failure_role,
                    "raw_result": event.get("raw_result"),
                    "evidence": str(telemetry.relative_to(root)),
                }
            )
    # A sampler binds a new one-operator query. Its operator=1 is local to
    # that query and must never be looked up as operator 1 of the full plan.
    if not details:
        error = result.get("error", "")
        operator = re.search(r"\boperator=(\d+)", error)
        request = re.search(r"\brequest_seq=(\d+)", error)
        planning = "optimizer rejected QueryIR" in error or "optimizer RPC" in error
        details.append(
            {
                "query": query["id"],
                "phase": "规划/算子采样" if planning else "执行失败（阶段未确认）",
                "operator_id": None,
                "stage_local_operator_id": int(operator[1]) if operator else None,
                "kind": None,
                "task": None,
                "fields": [],
                "request_seq": int(request[1]) if request else None,
                "error": error,
                "failure_role": "原始报错",
                "mapping_status": "采样局部编号；现有日志未保存该采样请求的任务，无法可靠映射到原查询算子"
                if planning and operator
                else "现有日志不足以定位算子",
            }
        )
    unique = {}
    for entry in details:
        key = (
            entry["operator_id"],
            entry.get("stage_local_operator_id"),
            entry["phase"],
            entry["failure_role"],
        )
        prior = unique.get(key)
        if prior is None or entry.get("raw_result") and not prior.get("raw_result"):
            unique[key] = entry
    return list(unique.values())


def render_errors(root, state, manifest):
    root = Path(root)
    entries = []
    for query in manifest["workloads"]:
        result = state["queries"].get(query["id"])
        if (
            not result
            or result["status"] == "completed"
            or result["sql_sha256"] != query["sql_sha256"]
        ):
            continue
        entries.extend(error_operators(root, query, result))

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "# 出错算子",
        "",
        "只列执行或采样报错；最终结果不在50～300行属于验收不达标，不在此表内。",
        "编号以各条查询实际绑定计划为准，不代表第几个执行。采样重新绑定的局部编号不当作完整查询算子编号。",
        "",
        "“连带中止”表示其他算子先报错，调度器随后拒绝该算子的请求，不表示这个filter自身回答格式出错。",
        "",
        "| 查询 | 阶段 | 错误性质 | 完整查询算子编号 | 类型 | 输入字段 | 过滤条件 | 请求 |",
        "|---|---|---|---:|---|---|---|---:|",
    ]
    for entry in entries:
        oid = entry["operator_id"]
        label = (
            str(oid)
            if oid is not None
            else f"未确认（采样局部{entry.get('stage_local_operator_id')}）"
        )
        task = entry.get("task") or entry.get("mapping_status", "未确认")
        lines.append(
            f"| {entry['query']} | {entry['phase']} | {entry['failure_role']} | {label} | "
            f"{entry.get('kind') or '未确认'} | {cell(', '.join(entry['fields']))} | "
            f"{cell(task)} | {entry.get('request_seq') or '—'} |"
        )
    for entry in entries:
        if not entry.get("raw_result"):
            continue
        lines += [
            "",
            f"## {entry['query']} · 算子{entry['operator_id']}",
            "",
            f"条件：{entry['task']}",
            "",
            "模型实际输出（截取）：",
            "",
            "> " + entry["raw_result"][:1200].replace("\n", "\n> "),
            "",
            f"[原始日志]({entry['evidence']})",
        ]
    if not entries:
        lines += ["", "目前没有已记录的执行/采样报错。"]
    path = root / "出错算子.md"
    temp = path.with_suffix(".md.tmp")
    temp.write_text("\n".join(lines) + "\n")
    temp.replace(path)
    return entries
