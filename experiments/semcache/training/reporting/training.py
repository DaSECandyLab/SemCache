"""Render training execution details, global calibration and fitting reports."""

import json
import os
from datetime import datetime
from functools import lru_cache, wraps
from pathlib import Path
from zoneinfo import ZoneInfo

from planning.prompt_layouts import BLOCK_ORDERS

from training.artifacts import read
from training.reporting.telemetry import MODELS

GLOBAL_START = "<!-- global-calibration:start -->"
GLOBAL_END = "<!-- global-calibration:end -->"


def latest_file_result(loader):
    """Keep one compact result per file, invalidating only changed files.

    Reports traverse every completed episode in the same order. An LRU
    smaller than the run evicts the next episode before it can be read, so
    every refresh reparses the entire run after the 512th execution.
    """
    entries = {}

    @wraps(loader)
    def cached(path, *revision):
        previous = entries.get(path)
        if previous is None or previous[0] != revision:
            previous = (revision, loader(path, *revision))
            entries[path] = previous
        return previous[1]

    cached.cache_clear = entries.clear
    return cached


def file_revision(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino


@lru_cache(maxsize=4)
def _global_profile(path, modified_ns, size):
    return read(path)["profile"]


def global_calibration_lines(output):
    """Show the committed global banks used by search, never plan predictions."""
    path = Path(output) / "calibration/global.json"
    if not path.exists():
        return [GLOBAL_START, "## 全局算子统计", "", "全局校准尚未完成。", GLOBAL_END]
    stat = path.stat()
    profile = _global_profile(str(path), stat.st_mtime_ns, stat.st_size)
    updated = datetime.fromtimestamp(stat.st_mtime, ZoneInfo("Asia/Shanghai"))
    lines = [
        GLOBAL_START,
        "## 全局算子统计",
        "",
        f"统计更新时间：{updated:%Y-%m-%d %H:%M:%S}（北京时间）。"
        "来源：[全局统计](calibration/global.json)，查询执行完成后合并更新。",
        "准确率为全局采样中与 teacher 对比的加权一致率，按模型、算子和 layout 区分；不是查询结果 F1。",
        "语义耗时为真实请求的平均响应时间，跨 layout 合并，包含初始校准、在线采样和实际执行的有效观测。",
        "",
        "| 模型 | 算子 | OTD准确率 | ODT准确率 | DOT准确率 | 平均耗时(s/请求) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for kind, label in (
            ("SEMANTIC_FILTER", "语义过滤"),
            ("SEMANTIC_JOIN", "语义Join"),
            ("SEMANTIC_PROJECTION", "语义Projection"),
        ):
            layouts = (
                profile.get("accuracy", {})
                .get(model, {})
                .get(kind, {})
                .get("layouts", {})
            )
            accuracies = [
                layouts.get(":".join(order), {}).get("value") for order in BLOCK_ORDERS
            ]
            latency = (
                profile.get("request_profiles", {})
                .get(model, {})
                .get(kind, {})
                .get("seconds_per_request")
            )
            values = [
                model.removeprefix("gemma4-"),
                label,
                *["—" if value is None else f"{value:.2%}" for value in accuracies],
                display(latency),
            ]
            lines.append("| " + " | ".join(values) + " |")
    labels = {
        "SCAN": "基表扫描",
        "FILTER": "关系过滤",
        "COMPARISON_JOIN": "关系Join",
        "CROSS_PRODUCT": "笛卡尔积",
        "PROJECTION": "关系Projection",
        "CTE_SCAN": "CTE扫描",
        "AGGREGATE": "聚合",
        "ORDER_BY": "排序",
        "LIMIT": "限制行数",
    }
    relational = profile.get("relational_profiles", {})
    if relational:
        lines.extend(
            [
                "",
                "关系算子使用全局校准的平均输入行耗时，单位为秒/输入行。",
                "",
                "| 关系算子 | 平均耗时(s/输入行) |",
                "|---|---:|",
            ]
        )
        for kind, record in sorted(relational.items()):
            if kind == "default":
                continue
            value = record.get("seconds_per_input")
            value = "—" if value is None else f"{value:.6f}"
            lines.append(f"| {labels.get(kind, kind)} ({kind}) | {value} |")
    lines.extend(
        [
            "",
            "O=算子说明，T=任务说明，D=输入数据；平均耗时不等于整条查询执行时间。",
            GLOBAL_END,
        ]
    )
    return lines


def with_global_calibration(lines, output):
    """Replace one header block; remain idempotent for live report refreshes."""
    lines = list(lines)
    if GLOBAL_START in lines:
        start = lines.index(GLOBAL_START)
        end = lines.index(GLOBAL_END, start)
        del lines[start : end + 1]
    header_end = 4
    if len(lines) > 4 and lines[4].startswith(("当前执行", "恢复校验")):
        header_end = 5
    tail = lines[header_end:]
    while tail and not tail[0]:
        tail.pop(0)
    return lines[:header_end] + [""] + global_calibration_lines(output) + [""] + tail


@latest_file_result
def _http_counts(path, *revision):
    counts = dict.fromkeys(MODELS, 0)
    with Path(path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            if (
                event.get("event")
                in (
                    "sampling_llm_submit",
                    "sampling_llm_retransmit",
                )
                and event.get("model_id") in counts
            ):
                counts[event["model_id"]] += 1
    return counts


def episode_metrics(leaf):
    leaf = Path(leaf)
    paths = [leaf / name for name in (
        "experience.json", "selected-plan.json", "sampling.json", "timing.json"
    )]
    revisions = tuple(file_revision(p) if p.exists() else None for p in paths)
    record = read_experience(paths[0])["records"][0]
    artifacts = record["metadata"].get("timing_artifacts", [])
    telemetry = Path(artifacts[0]) / "telemetry.jsonl" if artifacts else None
    telemetry_revision = (
        file_revision(telemetry) if telemetry is not None and telemetry.exists() else None
    )
    return _episode_metrics(str(leaf), revisions, telemetry_revision)


@latest_file_result
def _episode_metrics(leaf, revisions, telemetry_revision):
    leaf = Path(leaf)
    selected_path, sampling_path = leaf / "selected-plan.json", leaf / "sampling.json"
    selected = read(selected_path) if selected_path.exists() else {}
    scores = selected.get("rollout_selection", {}).get("predicted_scores")
    sampling = read(sampling_path) if sampling_path.exists() else {}
    timing_path = leaf / "timing.json"
    timing = read(timing_path) if timing_path.exists() else {}
    beam = timing.get("phases", {}).get("beam_search", {})
    sample_counts = sampling.get("http_requests_by_model")
    record = read_experience(leaf / "experience.json")["records"][0]
    artifacts = record["metadata"].get("timing_artifacts", [])
    execution_counts = None
    if artifacts:
        telemetry = Path(artifacts[0]) / "telemetry.jsonl"
        if telemetry.exists():
            execution_counts = _http_counts(str(telemetry.resolve()), *file_revision(telemetry))
    return {
        "beam_seconds": beam.get("wall_seconds"),
        "minimum_accuracy": selected.get(
            "minimum_accuracy", record["metadata"].get("minimum_accuracy")
        ),
        "duplicate_replaced": selected.get("rollout_selection", {}).get(
            "duplicate_replaced", False
        ),
        "replacement_scope": selected.get("rollout_selection", {}).get("replacement_scope"),
        "accuracy_fallback": selected.get("rollout_selection", {}).get(
            "accuracy_fallback", False
        ),
        "predicted_seconds": -scores[0] if scores is not None else None,
        "predicted_accuracy": scores[1] if scores is not None else None,
        "predicted_scores": scores,
        "model_http_requests": {
            model: {
                "sampling": sample_counts.get(model, 0)
                if sample_counts is not None
                else None,
                "execution": execution_counts[model]
                if execution_counts is not None
                else None,
            }
            for model in MODELS
        },
    }


def enrich_rows(rows):
    return [{**row, **episode_metrics(row["artifact"])} for row in rows]


@latest_file_result
def _experience(path, *revision):
    record = read(path)["records"][0]
    compact = {
        key: record.get(key)
        for key in ("status", "reward", "times", "accuracy", "time_lower_bound")
    }
    compact["metadata"] = {
        key: record["metadata"].get(key)
        for key in ("selection_policy", "timing_artifacts", "minimum_accuracy")
    }
    return {"records": [compact]}


def read_experience(path):
    """Reuse unchanged completed artifacts across progress refreshes."""
    path = Path(path)
    return _experience(str(path), *file_revision(path))


def saved_executions(output):
    selected = {}
    for path in Path(output).glob("measured/round-*/round.json"):
        selected[path.parent.name] = set(read(path)["query_ids"])
    return sum(
        path.parent.name.startswith("search-")
        and path.parent.name.removeprefix("search-").isdigit()
        and (path.parent / "collection.json").exists()
        and (
            path.parents[3].name not in selected
            or path.parents[1].name in selected[path.parents[3].name]
        )
        for path in Path(output).glob("measured/round-*/stream/**/experience.json")
    )


def display(value):
    return "—" if value is None else f"{value:.4f}"


def effective_epochs(epochs):
    """Display the retained weight lineage while preserving raw resume history."""
    retained, restarts, seen = [], [], set()
    discarded_seconds = 0.0
    for entry in epochs:
        resume = entry.get("resume") or {}
        restart = resume.get("learning_rate_restart")
        if restart:
            key = (resume["from_epoch"], restart["weights_from_epoch"], restart["base_rate"])
            if key not in seen:
                seen.add(key)
                restored_seconds = next(
                    row["display_seconds"] for row in retained
                    if row["epoch"] == restart["weights_from_epoch"]
                )
                discarded_seconds += retained[-1]["display_seconds"] - restored_seconds
                retained = [
                    row for row in retained
                    if row["epoch"] <= restart["weights_from_epoch"]
                ]
                restarts.append({**restart, "from_epoch": resume["from_epoch"],
                                 "first_epoch": entry["epoch"]})
        retained.append({**entry, "display_seconds": entry["seconds"] - discarded_seconds})
    return retained, restarts


def write_training_report(output, state, settings):
    """Summarize parameter fitting separately from SQL execution details."""
    output = Path(output)
    policy = settings["training"]["round_update"]
    searches = state.get(
        "searches_per_query", settings["training"].get("feedback_rollouts_per_query", 2)
    )
    rounds = state.get("rounds", settings["training"].get("rounds", 3))
    excluded = settings["training"].get("replay_exclude_rounds", [])
    total = state.get("total_executions", state.get("query_count", 28) * searches * rounds)
    phases = {
        "initial_calibration": "准备与恢复",
        "measured": "收集执行反馈",
        "value_update": "拟合模型参数",
        "complete": "完成",
        "completed": "完成",
    }
    lines = [
        "# 模型训练",
        "",
        f"状态：{state['status']}；阶段：{phases.get(state['phase'], state['phase'])}；"
        f"当前轮次：{state['round']}/{rounds}；执行反馈：{state['completed_executions']}/{total}。",
        f"更新：{state['updated_at']}",
        "",
        "随机初始化，无预训练。各轮查询数："
        + " / ".join(map(str, state.get("round_query_counts", [state.get("query_count", 28)] * rounds)))
        + f"；每条查询独立搜索并执行{searches}次。后续拟合只合并当前清单中SQL一致的历轮经验。",
        ("后续拟合排除第 " + "、".join(map(str, excluded)) + " 轮数据，保留已有模型权重。"
         if excluded else "后续拟合使用当前清单内的历轮执行经验。"),
        "训练标签按相同查询、中间状态及可比较缓存/统计特征合并，对同组完整执行配对取平均时间与平均F1；模型不输入准确率下限。",
        f"每轮至少训练 {policy['min_epochs']} 个 epoch，最多 {policy['max_epochs']} 个；"
        f"达到最少轮数后，连续 {policy['patience']} 个 epoch 没有累计超过 "
        f"{policy['relative_min_delta']:.1%} 的监控 loss 改善则停止。",
        "监控 loss 在固定参数下重新评估训练集，用于判断是否停止；它与训练过程中各批次平均的 loss 不同。",
        "耗时仅统计模型拟合，不包含查询采样、搜索和执行。",
        "",
        "[查询执行明细](执行明细.md)",
    ]
    for number in range(1, rounds + 1):
        directory = output / f"measured/round-{number}/model"
        epoch_file = directory / "epochs.jsonl"
        epochs = []
        if epoch_file.exists():
            epochs = [
                json.loads(line)
                for line in epoch_file.read_text().splitlines(keepends=True)
                if line.endswith("\n") and line.strip()
            ]
        lines.extend(["", f"## 第 {number} 轮", ""])
        if not epochs:
            active = (
                state["status"] == "running"
                and state["phase"] == "value_update"
                and state["round"] == number
            )
            lines.append(
                "拟合中；正在完成第1个epoch，完成后写入loss。"
                if active else "尚未开始拟合。"
            )
            continue
        report_file = directory / "training.json"
        final = read(report_file) if report_file.exists() else None
        user_stop = (final or {}).get("user_stop")
        if user_stop:
            epochs = [row for row in epochs if row["epoch"] <= user_stop["internal_epoch"]]
        last = epochs[-1]
        displayed_epochs, restarts = effective_epochs(epochs)
        display_numbers = {
            row["epoch"]: index for index, row in enumerate(displayed_epochs, 1)
        }
        display_budget = last["epochs"] - last["epoch"] + len(displayed_epochs)
        if user_stop:
            display_budget = user_stop["document_epoch"]
        complete = final is not None and (directory / "value.pt").exists()
        active = (
            state["status"] == "running"
            and state["phase"] == "value_update"
            and state["round"] == number
        )
        status = "已完成" if complete else "拟合中" if active else "尚未完成"
        coverage = last.get("epoch_coverage", {})
        lines.append(
            f"{status}；epoch {len(displayed_epochs)}/{display_budget}；"
            f"训练状态 {coverage.get('states', '—')}；"
            f"均值训练样本 {coverage.get('training_rows', '—')}；"
            f"累计有效拟合耗时 {displayed_epochs[-1]['display_seconds']:.1f} 秒。"
        )
        if complete and final is not None:
            reason = (final.get("early_stopping") or {}).get("reason") or ""
            reason = {
                "loss_plateau": "达到收敛停止条件",
                "max_epochs": "达到最大 epoch 数",
                "user_epoch_limit": "按用户指定的文档轮数停止",
            }.get(reason, reason or "拟合完成")
            lines.extend(
                [
                    "",
                    f"停止原因：{reason}；最佳 epoch："
                    f"{display_numbers.get(final.get('best_epoch'), '—')}。"
                    f"[模型文件](measured/round-{number}/model/value.pt)",
                ]
            )
        for restart in restarts:
            restored = restart["weights_from_epoch"]
            lines.extend([
                "",
                f"降学习率续训：从第 {restart['from_epoch']} 个检查点中保存的"
                f"第 {restored} 个 epoch 最佳权重恢复；基础学习率 "
                f"{restart['previous_base_rate']:g} → {restart['base_rate']:g}，"
                f"显示序号从 {display_numbers.get(restored, restored) + 1} 继续，不重新预热。",
                f"原高学习率分支的内部 epoch {restored + 1}～{restart['from_epoch']} "
                "已回退，原始日志保留。本表只显示保留的权重链，序号与累计耗时均连续衔接，扣除已回退分支的耗时。",
            ])
        if restarts:
            lines.extend(["", f"原始日志累计拟合耗时（含已回退分支）：{last['seconds']:.1f} 秒。"])
        lines.extend(
            [
                "",
                "| Epoch | 总 loss | 时间头 loss | 准确率头 loss | 监控 loss | 累计拟合(s) |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for entry in displayed_epochs:
            terms = entry.get("terms", {})
            values = [
                display_numbers[entry["epoch"]],
                *[
                    "—" if value is None else f"{value:.6f}"
                    for value in (
                        entry.get("value_loss"),
                        terms.get("time"),
                        terms.get("accuracy"),
                        entry.get("monitor_loss"),
                    )
                ],
                f"{entry['display_seconds']:.1f}",
            ]
            lines.append("| " + " | ".join(map(str, values)) + " |")
    if state.get("error"):
        lines.extend(["", "错误：" + state["error"]])
    temporary = output / f"模型训练.md.{os.getpid()}.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(output / "模型训练.md")


def render_table(rows):
    lines = [
        "调用次数仅统计计划执行阶段的真实HTTP请求（含重试）。",
        "Beamsearch包含本次搜索的选择率/准确率采样和计划搜索，不含整轮启动校准；执行(s)仅为实际SQL执行。",
        "预测时间和预测准确率取自该次搜索所选计划的value model评分(-预测秒数,+预测准确率)，不使用执行后的实测值。",
        "执行中的行每10秒更新累计耗时和HTTP请求数；结束后填入实测执行耗时与F1。",
        "",
        "| 轮 | 查询 | 次 | 下限 | 选择 | 状态 | Beamsearch(s) | 预测时间(s) | 预测准确率 | 执行(s) | F1 | E2B调用 | E4B调用 | 12B调用 |",
        "|---:|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        reward = row["reward"]
        calls = []
        for model in MODELS:
            count = row["model_http_requests"][model]["execution"]
            calls.append("—" if count is None else str(count))
        values = [
            row["round"],
            row["query"].removeprefix("translated/semcache/"),
            row["execution"],
            "—"
            if row.get("minimum_accuracy") is None
            else f"{row['minimum_accuracy']:.0%}",
            "随机换结构"
            if row.get("duplicate_replaced") and row.get("replacement_scope") == "logical_structure"
            else "随机换实现"
            if row.get("duplicate_replaced") and row.get("replacement_scope") == "semantic_implementation"
            else "随机换重"
            if row.get("duplicate_replaced")
            else "最高质量回退"
            if row.get("accuracy_fallback")
            else "fastest"
            if row["selection"] == "minimum_accuracy"
            else row["selection"],
            {
                "running": "执行中",
                "searching": "采样与搜索中",
                "preparing": "准备执行中",
                "finalizing": "保存结果中",
            }.get(row["status"], row["status"]),
            display(row.get("beam_seconds")),
            display(row["predicted_seconds"]),
            display(row["predicted_accuracy"]),
            f"{row['elapsed_seconds']:.1f}（进行中）"
            if row.get("elapsed_seconds") is not None
            else display(reward["latency_seconds"]),
            display(reward["accuracy"]),
            *calls,
        ]
        lines.append("| " + " | ".join(map(str, values)) + " |")
    return lines
