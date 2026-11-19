"""Render the v4 campaign as an atomic Markdown table every ten seconds."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

from experiments.system_comparison.accuracy.reporting.artifacts import (
    timing_fields,
)


def monitor_save(path, value):
    """Do not share temporary filenames with the controller's renderer."""
    path = Path(path)
    tmp = path.with_name(path.name + f".monitor-{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str)
        + "\n"
    )
    tmp.replace(path)


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default



def append_tables_and_save(run, ctx, lines):
    config = ctx.config
    control = ctx.control
    lane_states = ctx.lane_states
    counts = ctx.counts
    records = ctx.records
    state = ctx.state
    tasks = ctx.tasks
    task_lookup = ctx.task_lookup
    variants = ctx.variants
    model_count = ctx.model_count
    names = [v["name"] for v in variants]
    def phase_value(record, field):
        value = record.get(field)
        if value is None:
            value = (record.get("timing") or {}).get(field)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            return None
        return float(value)

    def phase_table(kind):
        rows = ["| 查询 | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for query in config.get("query_ids", []):
            cells = []
            for name in names:
                task = task_lookup.get((name, query))
                record = records.get(task["task_id"]) if task else None
                if task and kind == "overhead" and task["system"] == "moar":
                    folder = run / task["folder"]
                    reuse = read(folder / "plan-reuse-input.json", {})
                    search = read(folder / "moar-search-summary.json", {})
                    previous = reuse.get("previous_search_seconds")
                    current = search.get("seconds")
                    if type(previous) in (int, float) and math.isfinite(previous) and previous >= 0:
                        cells.append(f"[{previous:.2f}†]({task['folder']}/plan-reuse-input.json)")
                        continue
                    if (type(current) in (int, float) and math.isfinite(current) and current >= 0
                            and search.get("iterations", 0) > 0):
                        cells.append(f"[{current:.2f}]({task['folder']}/moar-search-summary.json)")
                        continue
                if not (task and record and record.get("status") == "completed"
                        and task["system"] not in control.get("excluded_systems", [])
                        and (run / task["folder"] / "result.json").exists()):
                    cells.append("-")
                    continue
                if kind == "execution":
                    value = phase_value(record, "execution_seconds")
                else:
                    planning = phase_value(record, "planning_seconds")
                    sampling = phase_value(record, "sampling_seconds")
                    method = (record.get("timing") or {}).get("method") or ""
                    # MOAR's measured optimization interval already includes
                    # candidate sampling/evaluation; it is not a missing zero.
                    includes_sampling = (task["system"] == "moar" and
                                         "optimization including candidate evaluation" in method)
                    value = (planning if includes_sampling else planning + sampling
                             if planning is not None and sampling is not None else None)
                if value is None:
                    cells.append("-")
                    continue
                timing_path = run / task["folder"] / "timing.json"
                target = "timing.json" if timing_path.exists() else "summary.json"
                cells.append(f"[{value:.2f}]({task['folder']}/{target})")
            rows.append("| " + query + " | " + " | ".join(cells) + " |")
        return rows

    lines += [
        "",
        "最终执行时间（秒；不包含采样、计划搜索和输入加载；仅显示有有效最终输出且记录了该阶段的查询）。",
        "",
        *phase_table("execution"),
        "",
        "采样与计划搜索总时间（秒；不包含最终执行和输入加载）。MOAR 搜索计时已包含候选采样与评估，不再重复相加；即使最终执行失败，只要搜索完成也保留其时间。标记 † 表示复用计划的原始搜索时间，本次未重新搜索；其他系统按记录的采样和规划时间相加。阶段未完整记录时显示 `-`，不将缺失值当作 0。",
        "",
        *phase_table("overhead"),
    ]
    adjustments = [(task, records[task["task_id"]]) for task in tasks
                   if task["task_id"] in records
                   and timing_fields(records[task["task_id"]])["time_basis"] == "adjusted_exclusive_wait_estimate"]
    if adjustments:
        lines += ["", "历史查询总时间修正记录（不进入上述执行时间和采样／规划时间表）；原始实测时间保留，不等同于重新测量。", "",
                  "| 配置/查询 | 原始实测（秒） | 扣除等待（秒） | 修正估算（秒） | 明细 |",
                  "|---|---:|---:|---:|---|"]
        for task, record in adjustments:
            adjustment = record["timing_adjustment"]
            lines.append(f"| {task['task_id']} | {record['seconds']:.2f} | {adjustment['excluded_seconds']:.2f} | {adjustment['adjusted_query_seconds']:.2f} | [计算记录]({task['folder']}/wait-adjustment.json) |")
    repeats = [
        (task, records[task["task_id"]])
        for task in tasks
        if task["task_id"] in records
        and records[task["task_id"]].get("average_seconds") is not None
    ]
    if repeats:
        lines += [
            "",
            "历史查询总时间重复测量（不进入上述阶段时间表；第一轮清缓存，第二轮保留模型服务缓存；单位为秒）。",
            "",
            "| 配置/查询 | 第一轮 | 第二轮 | 平均 | 记录 |",
            "|---|---:|---:|---:|---|",
        ]
        for task, record in repeats:
            lines.append(
                f"| {task['task_id']} | {record['initial_seconds']:.2f} | "
                + f"{record['second_seconds']:.2f} | {record['average_seconds']:.2f} | "
                + f"[两轮明细]({task['folder']}/repeat-manifest.json) |"
            )
    phase_rows = []
    for variant in names:
        variant_records = [
            records[t["task_id"]]
            for t in tasks
            if t["variant"] == variant and t["task_id"] in records
        ]
        if not any((r.get("sampling_seconds") or 0) > 0 for r in variant_records):
            continue

        def total(name, variant_records=variant_records):
            values = [
                float(r[name]) for r in variant_records if r.get(name) is not None
            ]
            return "-" if not values else f"{sum(values):.2f}"

        phase_rows.append(
            (
                variant,
                total("sampling_seconds"),
                total("planning_seconds"),
                total("execution_seconds"),
                total("load_seconds"),
                len(variant_records),
            )
        )
    if phase_rows:
        lines += [
            "",
            "原生采样阶段汇总（秒；只列出实际暴露采样阶段的配置；详细数据见各任务 `timing.json`）。",
            "",
            "| 配置 | 已记录查询 | 采样合计 | 规划/选案合计 | 最终执行合计 | 输入加载合计 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in phase_rows:
            lines.append(
                f"| {row[0]} | {row[5]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |"
            )
    lines += [
        "",
        "请求统计（每格为：逻辑调用 / 实际上游HTTP请求 / prompt cache命中；没有最终记录时为 `-`）。",
        "",
        "| 查询 | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for query in config.get("query_ids", []):
        cells = []
        for name in names:
            task = task_lookup.get((name, query))
            record = records.get(task["task_id"]) if task else None
            cells.append(
                f"{record.get('model_requests', '-')} / {record.get('upstream_http_attempts', '-')} / {record.get('cache_hits', 0)}"
                if record else "-"
            )
        lines.append("| " + query + " | " + " | ".join(cells) + " |")
    errors = [
        (t, records[t["task_id"]])
        for t in tasks
        if t["task_id"] in records
        and records[t["task_id"]]["status"] not in {"completed", "skipped"}
    ]
    lines += ["", "| 异常任务 | 原因 | 日志 |", "|---|---|---|"]
    for task, record in errors[-30:]:
        reason = (
            str(record.get("reason", "")).replace("|", "/").replace("\n", " ")[:400]
        )
        if record.get("status") == "timeout" and record.get("timeout_seconds") is not None:
            reason = f"整条查询达到 {record['timeout_seconds']} 秒上限，已终止该任务进程组"
        lines.append(
            f"| {task['task_id']} | {reason} | [stderr]({task['folder']}/stderr.log) · [SQL stderr]({task['folder']}/native-sql/stderr.log) · [异常堆栈]({task['folder']}/error.txt) · [API 日志]({task['folder']}/gateway.jsonl) |"
        )
    lines += [
        "",
        "每项保留 job.json、stdout.log、stderr.log、summary.json、有效 result.json 及完整 gateway.jsonl。所有配置的输出位置汇总到 [outputs/index.json](outputs/index.json)，用于最终准确率对比；不把异常任务伪造成空答案。",
        "",
        ("本轮不执行 KV cache 清除；保留策略见 cache-reuse/，此前清除记录见 cache-resets/。" if control.get("preserve_cache") else f"单模型配置只清所属模型 cache，候选池配置清其使用的 {model_count} 个模型 cache；记录见 cache-resets/。")
        + "空结果可以是合法输出；本表不代表答案准确率。",
        "",
    ]
    progress = run / config.get("progress_file", "progress.md")
    tmp = progress.with_name(progress.name + f".monitor-{os.getpid()}.tmp")
    tmp.write_text("\n".join(lines))
    tmp.replace(progress)
    exports = []
    for task in tasks:
        record = records.get(task["task_id"])
        result = run / task["folder"] / "result.json"
        exports.append(
            {
                **task,
                "status": record.get("status") if record else None,
                "seconds": record.get("seconds") if record else None,
                "average_seconds": record.get("average_seconds") if record else None,
                "model_requests": record.get("model_requests") if record else None,
                "upstream_http_attempts": record.get("upstream_http_attempts")
                if record
                else None,
                "upstream_retries": record.get("upstream_retries") if record else None,
                "native_mechanism_profile": record.get("native_mechanism_profile")
                if record
                else None,
                "native_format_fallbacks": record.get("native_format_fallbacks")
                if record
                else None,
                "native_plan_files": record.get("native_plan_files")
                if record
                else None,
                "requests_by_model": record.get("requests_by_model", {})
                if record
                else {},
                "prompt_tokens": record.get("prompt_tokens") if record else None,
                "completion_tokens": record.get("completion_tokens")
                if record
                else None,
                "sampling_seconds": record.get("sampling_seconds") if record else None,
                "planning_seconds": record.get("planning_seconds") if record else None,
                "execution_seconds": record.get("execution_seconds")
                if record
                else None,
                "load_seconds": record.get("load_seconds") if record else None,
                "timing_file": task["folder"] + "/timing.json"
                if (run / task["folder"] / "timing.json").exists()
                else None,
                "execution_resources": record.get("execution_resources")
                if record
                else None,
                "peak_upstream_requests": record.get("peak_upstream_requests")
                if record
                else None,
                "result_file": str(result.relative_to(run))
                if result.exists()
                else None,
                "job_file": task["folder"] + "/job.json"
                if (run / task["folder"] / "job.json").exists()
                else None,
                "summary_file": task["folder"] + "/summary.json" if record else None,
                "valid_output": bool(
                    record and record.get("status") == "completed" and result.exists()
                ),
            }
        )
        exports[-1]["formal_comparison_eligible"] = bool(
            exports[-1]["valid_output"]
            and task["system"] not in control.get("excluded_systems", [])
        )
    (run / "outputs").mkdir(exist_ok=True)
    monitor_save(
        run / "outputs/index.json",
        {
            "workload_contract": config.get("contract"),
            "query_manifest_sha256": config.get("query_manifest_sha256"),
            "input_manifest": "inputs/manifest.json",
            "model_snapshot": "models.json",
            "records": exports,
            "updated_unix": time.time(),
        },
    )
    monitor_save(
        run / "summary.json",
        {
            "planned": len(tasks),
            "finished": len(records),
            "counts": dict(counts),
            "records": list(records.values()),
            "active_lanes": lane_states,
            "updated_unix": time.time(),
        },
    )
    return state.get("phase")
