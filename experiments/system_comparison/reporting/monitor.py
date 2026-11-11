"""Render the v4 campaign as an atomic Markdown table every ten seconds."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from experiments.core.io import save
from experiments.system_comparison.accuracy.artifacts import timing_fields


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


def render(run):
    run = Path(run)
    config = read(run / "campaign.json", {})
    state = read(run / "status.json", {})
    control = read(run / "control.json", {})
    if state.get("phase") in {"running", "running_parallel"} and state.get(
        "controller_pid"
    ):
        try:
            os.kill(state["controller_pid"], 0)
        except ProcessLookupError:
            state.update(
                phase="controller_failed",
                reason="Controller exited unexpectedly; inspect controller.log and retained per-query logs",
            )
            save(run / "status.json", state)
    tasks = config.get("tasks", [])
    task_lookup = {(task["variant"], task["query_id"]): task for task in tasks}
    records = {}
    for task in tasks:
        record = read(run / task["folder"] / "summary.json")
        if record and record.get("parent_finalized"):
            records[task["task_id"]] = record
    counts = Counter(r.get("status", "failed") for r in records.values())
    active = state.get("task_id") if state.get("phase") == "running" else None
    lane_states = []
    if state.get("phase") == "running_parallel":
        lane_states = [
            read(run / path, {}) for path in state.get("lane_status_files", [])
        ]
    now = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S CST")
    models = read(run / "models.json", read(run / "preflight/models.json", []))
    model_count = len(models) or (3 if config.get("preset") == "three-model-quality-grid" else 4)
    displayed_lanes = config.get("active_model_lanes", config.get("model_lanes", []))
    lane_count = len(displayed_lanes) or model_count
    lotus_variant = next((v for v in config.get("variants", []) if v["system"] == "lotus"), {})
    lotus_model_count = len(lotus_variant.get("allowed", [])) or model_count
    lotus_description = (
        "LOTUS：E2B → 12B 两级级联，E2B为helper、12B为main；基于原生过滤、采样及阈值学习。"
        if lotus_model_count == 2
        else f"LOTUS：{lotus_model_count} 模型扩展级联，基于原生语义过滤与阈值学习构建；不是官方多模型原生模式。"
    )
    lines = [
        "# 六系统共同数值查询实验进度" if any(v["system"] == "moar" for v in config.get("variants", [])) else "# 五系统共同数值查询实验进度",
        "",
        f"更新时间：{now}；每 10 秒自动刷新。",
        "",
        f"状态：**{state.get('phase', 'preparing')}**；已结束 **{len(records)}/{len(tasks)}**；有效输出 {counts['completed']}，异常 {counts['failed']}，超时 {counts['timeout']}，部分输出 {counts['partial']}，取消 {counts['cancelled']}，跳过 {counts['skipped']}，不支持 {counts['unsupported']}。",
        "",
        "查询契约："
        + "、".join(
            config.get(
                "query_contracts",
                [config.get("contract", "selected19-common-numeric-v4")],
            )
        )
        + ("。SemCache 将在其他系统结束后独立运行，结果保存在本目录的 semcache/。" if config.get("semcache_followup") else "。SemCache 本轮不运行。" if config.get("settings_source") and config.get("semcache_followup") is False else "。SemCache 本轮使用外部 adaptive-20 结果接入准确率表，不在 compare-20 控制器中重跑。")
        + f"执行模型池为 {model_count} 个已配置服务；最终结果不在本轮重新生成参考答案。",
        "",
        lotus_description + "数值/文本 projection 使用 12B。Python 关系宿主耗时计入查询时间。",
        "",
    ]
    if config.get("next_run_timeout_seconds"):
        lines += [
            f"本次所选配置的整条查询上限为 {int(config['next_run_timeout_seconds'] // 60)} 分钟；保留任务的历史上限以各自 job.json 为准。"
            + ("当前仅准备配置，尚未启动。" if str(config.get("next_run_status", "")).startswith("prepared") else ""),
            "",
        ]
    if config.get("prompt_cache_enabled"):
        policy = config.get("system_prompt_cache", {})
        selected_systems = {v["system"] for v in config.get("variants", []) if v["name"] in config.get("rerun_variants", [])}
        enabled = sorted(system for system in selected_systems if policy.get(system, True))
        disabled = sorted(system for system in selected_systems if not policy.get(system, True))
        lines += ["本次重跑 prompt cache：开启=" + "、".join(enabled) + "；关闭=" + ("、".join(disabled) or "无")
                  + "。按模型与完整请求精确复用，不进行二次核对；开启缓存的配置开始前清一次，逐查询保留。模型端KV前缀缓存仍开启。", ""]
    recovered = [task for task in tasks if records.get(task["task_id"], {}).get("postprocessing_recovered")]
    if recovered:
        lines += ["以下任务已从同次执行保存的完整行恢复结果序列化，未重跑查询或追加HTTP请求；原始错误、返回码和计时保留在修复记录中："
                  + "、".join(f"[{task['task_id']}]({task['folder']}/postprocessing-repair.json)" for task in recovered) + "。", ""]
    if config.get("moar"):
        agent = config["moar"].get("agent", {}).get("model", "12B")
        lines += [
            f"MOAR：E2B/E4B/12B 候选池，{agent} 负责改写，原生 MCTS 搜索和 best() 选案；请求并发{config.get('system_concurrency', {}).get('moar', 16)}、CPU线程16、单条输入请求。聚合查询保存所选计划同次完整执行的聚合前行集。",
            "",
        ]
        if (run / "model-restart.json").exists():
            lines += [
                "三个执行模型已按用户要求重启，上下文65,536、Decode CUDA Graph=full。配置和长输入验证见 [模型重启记录](model-restart.json)。",
                "",
            ]
    tpch_revision = config.get("dataset_revisions", {}).get("tpch")
    if tpch_revision:
        lines += [
            f"TPC-H 输入：**SF={tpch_revision['scale_factor']}**；本轮仅重跑 TPC-H，其余数据集保留原结果。"
            + " 数据来源和完整校验见 [输入更新记录](input-refreshes/tpch-sf0005/refresh.json)。",
            "",
        ]
    if control.get("disabled_systems"):
        lines += [
            "用户已停止：**"
            + "、".join(control["disabled_systems"])
            + "**。其余系统继续；已取消 "
            + str(counts["cancelled"])
            + " 项，取消不代表查询完成。",
            "",
        ]
    if control.get("disabled_variants"):
        lines += ["当前不调度（已有结果保留）：**" + "、".join(control["disabled_variants"]) + "**。", ""]
    q7_deferred = set(control.get("q7_priority_deferred_tasks", []))
    q8_deferred = set(control.get("q8_priority_deferred_tasks", []))
    priority_deferred = q7_deferred | q8_deferred
    actually_skipped = [task for task in control.get("skipped_tasks", []) if task not in priority_deferred]
    if actually_skipped:
        lines += [
            "按用户要求跳过："
            + "、".join(actually_skipped)
            + "；时间表显示 `-`，原日志保留。",
            "",
        ]
    if q7_deferred:
        lines += ["为优先运行 MOAR appstore/Q7，暂缓 appstore/Q9 和 TPC-H 查询；Q7 结束后自动继续。", ""]
    if q8_deferred:
        lines += ["为优先运行 MOAR appstore/Q8，暂缓 TPC-H 查询；Q8 结束后自动继续。", ""]
    if control.get("abacus_no_sampling_tasks"):
        lines += [
            "Abacus 无采样试验："
            + "、".join(control["abacus_no_sampling_tasks"])
            + f"。使用原生 Dataset.run、MinTime、{model_count} 模型候选池和 Pareto 搜索；不执行 MAB、不构建训练样本、不调用 Validator。此配置与此前开启采样的查询不同。",
            "",
        ]
    if control.get("excluded_systems"):
        lines += [
            "仅供诊断、不纳入当前对比表："
            + "、".join(control["excluded_systems"])
            + "。原输出和日志保留。",
            "",
        ]
    moar_compact = read(run / "moar-compact-rerun.json", {})
    if moar_compact.get("phase") in {"waiting_for_current_moar", "starting", "running"}:
        pending = "、".join(moar_compact.get("queries", []))
        message = ("当前 MOAR 队列结束后自动补跑" if moar_compact["phase"] == "waiting_for_current_moar"
                   else "正在补跑")
        lines += [f"MOAR 简洁输出修复：{message} {pending}；复用已保存的计划，不再搜索；其余查询使用新约束继续。", ""]
    elif moar_compact.get("phase") == "failed":
        lines += ["MOAR 简洁输出补跑未能自动开始或完成：" + str(moar_compact.get("error", "未知原因")), ""]
    moar_force = read(run / "moar-force-rerun.json", {})
    if moar_force.get("phase") == "running_reused_plans":
        lines += ["MOAR 已强制重新运行：先直接执行 10 条已保存计划，不再搜索；完成后继续搜索并执行 10 条尚无计划的查询。", ""]
    elif moar_force.get("phase") in {"starting_remaining", "running_remaining"}:
        lines += ["MOAR 已执行完复用计划的查询；正在继续尚无计划的查询，保留本轮已有缓存。", ""]
    elif moar_force.get("phase") == "failed":
        lines += ["MOAR 后续队列未能自动完成：" + str(moar_force.get("error", "未知原因")), ""]
    q7_rerun = read(run / "moar-q7-format-rerun.json", {})
    if q7_rerun.get("phase") == "waiting_for_current_moar":
        lines += ["MOAR appstore/Q7 布尔输出修复：当前队列结束后，复用原计划自动重跑，保留本轮缓存。", ""]
    elif q7_rerun.get("phase") in {"starting", "running"}:
        lines += ["MOAR appstore/Q7：正在用修复后的布尔输出处理复用原计划重跑。", ""]
    elif q7_rerun.get("phase") == "failed":
        lines += ["MOAR appstore/Q7 自动重跑未能完成：" + str(q7_rerun.get("error", "未知原因")), ""]
    after_q7 = read(run / "moar-after-q7.json", {})
    if after_q7.get("phase") == "waiting_for_q7":
        lines += ["MOAR 后续队列：Q7 结束后继续 appstore/Q9 和 7 条 TPC-H 查询，不清除缓存。", ""]
    elif after_q7.get("phase") in {"starting", "running"}:
        lines += ["MOAR 正在继续 Q7 后暂缓的 appstore/Q9 和 TPC-H 查询。", ""]
    elif after_q7.get("phase") == "failed":
        lines += ["MOAR Q7 后续队列未能自动继续：" + str(after_q7.get("error", "未知原因")), ""]
    after_q8 = read(run / "moar-after-q8.json", {})
    if after_q8.get("phase") == "waiting_for_q8":
        lines += ["MOAR Q8 正在复用原计划重跑；完成后继续 7 条 TPC-H 查询，不清除缓存。", ""]
    elif after_q8.get("phase") in {"starting", "running"}:
        lines += ["MOAR 正在继续 Q8 后暂缓的 TPC-H 查询。", ""]
    elif after_q8.get("phase") == "failed":
        lines += ["MOAR Q8 后续队列未能自动继续：" + str(after_q8.get("error", "未知原因")), ""]
    if active:
        lines += [
            f"当前：**{active}**；已运行 {state.get('elapsed_seconds', 0):.1f} 秒；模型请求 {state.get('model_requests', 0)}，完成 {state.get('completed_model_requests', 0)}，传输错误 {state.get('model_errors', 0)}。",
            "",
        ]
    elif state.get("reason"):
        lines += ["说明：" + str(state["reason"]).replace("\n", " ")[:1000], ""]
    if (run / "accuracy-multireference.md").is_file():
        lines += ["准确率见 [各系统对全部可用 Teacher 答案的 F1](accuracy-multireference.md)；不包括 MOAR。", ""]
    if (run / "accuracy-own-reference.md").is_file():
        lines += ["各系统对各自 Teacher 答案的结果见 [同系统参考 F1](accuracy-own-reference.md)。", ""]
    if (run / "semcache-accuracy-augmentation.json").is_file():
        lines += ["已加入外部 SemCache adaptive-20 结果；详见上述两份准确率报告。", ""]
    if displayed_lanes:
        schedule = (
            f"所选单模型配置按模型分为 {lane_count} 条并行队列；未选配置保留结果。"
            if config.get("preserved_variants")
            else f"PLOP/Sema 按模型分为 {lane_count} 条队列；同一模型的配置逐一完成，不同模型可并行。"
        )
        lines += [
            schedule
            + ("本轮保留模型 KV cache；不同配置共享已保留的服务缓存。" if control.get("preserve_cache") else "每个配置开始前清理其使用的模型 KV cache。")
            + "候选池配置在所有模型队列全部结束后串行运行。",
            "",
            "候选池顺序："
            + " → ".join(
                variant["name"]
                for variant in config.get("variants", [])
                if variant["system"] not in {"plop", "sema"} and variant["name"] not in config.get("preserved_variants", [])
            )
            + "。",
            "",
            "| 模型队列 | 配置顺序 | taskset CPU 集合（16 个逻辑 CPU） | 网关端口 |",
            "|---|---|---|---:|",
        ]
        for lane in displayed_lanes:
            lines.append(
                f"| {lane['model']} | "
                + " → ".join(lane["variants"])
                + f" | `{lane['cpu_set']}` | {lane['gateway_port']} |"
            )
        lines.append("")
    if lane_states:
        lines += [
            "| 当前并行配置 | 查询 | 已运行（秒） | 模型请求 | 传输错误 |",
            "|---|---|---:|---:|---:|",
        ]
        for lane, live in zip(config.get("model_lanes", []), lane_states):
            running = live.get("phase") == "running"
            lines.append(
                "| "
                + str(live.get("variant", "-"))
                + " | "
                + (str(live.get("query_id", "-")) if running else "-")
                + " | "
                + (f"{live.get('elapsed_seconds', 0):.1f}" if running else "-")
                + " | "
                + (str(live.get("model_requests", 0)) if running else "-")
                + " | "
                + (str(live.get("model_errors", 0)) if running else "-")
                + " |"
            )
        lines.append("")
    variants = config.get("variants", [])
    models = read(run / "models.json", read(run / "preflight/models.json", []))
    lines += ["| 模型别名 | 实际模型 | API |", "|---|---|---|"]
    for model in models:
        lines.append(f"| {model['alias']} | {model['id']} | {model['endpoint']} |")
    lines += [
        "",
        "| 配置 | 已结束 | 模型请求 | 传输重试 | 输入/输出 tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in variants:
        selected = [t for t in tasks if t["variant"] == variant["name"]]
        done = [records[t["task_id"]] for t in selected if t["task_id"] in records]
        requests = sum(r.get("model_requests", 0) for r in done)
        prompt = sum(r.get("prompt_tokens", 0) for r in done)
        completion = sum(r.get("completion_tokens", 0) for r in done)
        retries = sum(r.get("upstream_retries", 0) for r in done)
        lines.append(
            f"| {variant['name']} | {len(done)}/{len(selected)} | {requests} | {retries} | {prompt}/{completion} |"
        )
    profiles = Counter(
        (r.get("execution_resources") or {}).get("resource_profile", "legacy")
        for r in records.values()
    )
    lines += [
        "",
        "资源配置按实际运行记录区分；CPU线程16，默认请求并发16，batch 参数为 "
        + str(config.get("execution_resources", {}).get("batch_size", "未记录"))
        + "。单模型配置默认网关上游并发上限为 "
        + str(config.get("gateway_request_limit", 32))
        + "。",
    ]
    if config.get("system_concurrency"):
        lines += ["各系统请求并发及其网关总上限：" + "、".join(
            f"{system}={limit}" for system, limit in config["system_concurrency"].items()
        ) + "；候选池内所有模型共享该上限。"]
    if config.get("system_batch_sizes"):
        limits = "、".join(
            f"{system} 的 batch 上限 {size}"
            for system, size in sorted(config["system_batch_sizes"].items())
        )
        lines += [
            limits + "；其余系统使用上述默认 batch。每个配置的网关并发上限由全部模型共同使用。",
            "",
        ]
    if profiles:
        lines += [
            "",
            "| 资源配置 | 已结束任务 |",
            "|---|---:|",
        ]
        for profile, count in sorted(profiles.items()):
            lines.append(f"| {profile} | {count} |")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    save(args.run / "monitor-launch.json", {"pid": os.getpid(), "interval_seconds": 10})
    progress = args.run / read(args.run / "campaign.json", {}).get("progress_file", "progress.md")
    while True:
        phase = render(args.run)
        if phase in {
            "finished",
            "selection_finished",
            "controller_failed",
            "cancelled",
            "repair_finished",
            "moar_finished",
            "blocked_context",
            "stopped_for_native_mechanism_repair",
        }:
            break
        written = progress.stat().st_mtime_ns
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            time.sleep(min(1, max(0, deadline - time.monotonic())))
            try:
                if progress.stat().st_mtime_ns != written:
                    break
            except FileNotFoundError:
                break


if __name__ == "__main__":
    main()
