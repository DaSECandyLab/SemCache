"""Render the training loss report (``<run>/模型训练.md``).

Sources, in order of availability:

* ``<round>/epochs.jsonl`` — one line per finished epoch, written live by
  ``training.pipeline.fit`` so the curve is visible while the model is fitting.
* ``<round>/loss-checks.jsonl`` — loss checks of the newest checkpoint,
  aggregated here once ``model/value.pt`` exists.
* ``<round>/model/training.json`` — the finished round's summary.

No plotting dependency: curves are ASCII sparklines.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from .files import publish

BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float]) -> str:
    if not values:
        return ""
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return BLOCKS[0] * len(values)
    return "".join(
        BLOCKS[
            min(
                len(BLOCKS) - 1,
                int((value - low) / (high - low) * (len(BLOCKS) - 1) + 0.5),
            )
        ]
        for value in values
    )


def read_checks(run: Path):
    rows = []
    for round_dir in sorted(run.glob("*/round-*")):
        journal = round_dir / "loss-checks.jsonl"
        if not journal.exists():
            continue
        for line in journal.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "ok":
                row["round"] = str(round_dir.relative_to(run))
                rows.append(row)
    return rows


def read_epochs(round_dir: Path):
    candidates = [round_dir / "model" / "epochs.jsonl", round_dir / "epochs.jsonl"]
    journal = next((path for path in candidates if path.exists()), None)
    if journal is None:
        return []
    epochs = []
    for line in journal.read_text().splitlines():
        if not line.strip():
            continue
        try:
            epochs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return epochs


def family_of(query: str) -> str:
    for marker in ("-train-", "-validation-", "-evaluation-"):
        if marker in query:
            return query.split(marker, 1)[1].split("-", 1)[-1]
    return "?"


def dataset_of(query: str) -> str:
    for marker in ("-train-", "-validation-", "-evaluation-"):
        if marker in query:
            return query.split(marker, 1)[0]
    return "?"


def coverage_section(lines, coverage):
    if not coverage:
        return
    lines.extend(
        [
            f"- 本 Epoch 全量遍历：基础状态 {coverage['visited_states']}/{coverage['states']}，"
            f"条件监督行 {coverage['visited_condition_rows']}/{coverage['condition_rows']}，"
            f"完成 {coverage['updates']} 个计算批次。批大小不限制查询的训练状态总数。",
            "",
        ]
    )


def epoch_section(lines, round_dir: Path, epochs):
    report_path = round_dir / "model" / "training.json"
    running = not report_path.exists()
    label = "训练中" if running else "已完成"
    values = [float(entry.get("value_loss") or 0.0) for entry in epochs]
    latest = epochs[-1]
    training_monitor = latest.get("monitor_source") == "training"
    total_epochs = latest.get("epochs")
    lines.append(
        f"## {round_dir.relative_to(round_dir.parent.parent)} 逐 epoch（{label}）"
    )
    lines.append("")
    lines.append(f"`{sparkline(values)}`")
    lines.append("")
    lines.append(
        f"已跑 {latest.get('epoch')}/{total_epochs} epoch · 最新 value_loss {values[-1]:.4f} · "
        f"最小 {min(values):.4f} · 累计 {float(latest.get('seconds') or 0):.1f} s"
    )
    lines.append("")
    if training_monitor:
        lines += [
            f"- 平台期监控：全量训练数据、dropout 关闭后的复核 loss "
            f"{latest['monitor_loss']:.6f}；不使用测试查询。",
            "",
        ]
    lines.append(
        "| epoch | train_loss | time | accuracy | censor | "
        + ("复核loss" if training_monitor else "validation")
        + " | 训练(s) | "
        + ("复核(s)" if training_monitor else "验证(s)")
        + " | 累计(s) |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for entry in epochs:
        terms = entry.get("terms") or {}
        validation = (
            entry.get("monitor_loss") if training_monitor else entry.get("validation")
        )
        lines.append(
            f"| {entry.get('epoch')} | {float(terms.get('total') or entry.get('value_loss') or 0):.4f} | "
            f"{float(terms.get('time') or 0):.4f} | {float(terms.get('accuracy') or 0):.4f} | "
            f"{float(terms.get('time_lower_bound') or 0):.4f} | "
            f"{'—' if validation is None else f'{float(validation):.4f}'} | "
            f"{format(float(entry['training_seconds']), '.1f') if 'training_seconds' in entry else '—'} | "
            f"{format(float(entry['validation_seconds']), '.1f') if 'validation_seconds' in entry else '—'} | "
            f"{float(entry.get('seconds') or 0):.1f} |"
        )
    lines.append("")

    censoring = latest.get("estimated_censoring") or {}
    if censoring.get("threshold_seconds") is not None:
        lines.append(
            f"- 估计超限阈值 {censoring['threshold_seconds']:g} 秒；"
            f"超限完整状态 {censoring.get('terminal_states', 0)}；"
            f"屏蔽时间标签的部分状态 {censoring.get('prefix_time_targets_omitted', 0)}；"
            f"验证集超限完整状态 {censoring.get('validation_terminal_states', 0)}。"
            "这些下界来自估计环境，真实超时另行记录。"
        )
        lines.append("")

    coverage_section(lines, latest.get("epoch_coverage"))
    saved = [e["checkpoint"] for e in epochs if e.get("checkpoint")]
    if saved:
        lines.extend(
            [
                "- 已保存 Epoch 检查点："
                + "、".join(str(e["epoch"]) for e in saved)
                + "。",
                "",
            ]
        )

    try:
        report = json.loads(report_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        report = None
    if report:
        stopping = report.get("early_stopping") or {}
        metric = "训练复核" if stopping.get("monitor") == "training" else "验证"
        lines.append(
            f"- 最佳{metric}检查点：best_epoch {report.get('best_epoch')} · "
            f"train value_loss {report.get('training', {}).get('value_loss'):.4f} · "
            f"validation value_loss {report.get('validation', {}).get('value_loss')}"
        )
        if stopping:
            lines.append(
                "- 停止原因："
                + (
                    "loss 进入平台期"
                    if stopping["reason"] == "loss_plateau"
                    else "达到最大 Epoch 预算"
                )
                + f"；至少 {stopping['min_epochs']} Epoch，连续 {stopping['patience']} Epoch "
                + f"未达到 {stopping['relative_min_delta']:.2%} 的累计相对改善。"
            )
        lines.append("")


def query_loss(row, name):
    target = {
        "time": "time_targets",
        "accuracy": "accuracy_targets",
        "time_lower_bound": "censor_targets",
    }.get(name)
    if row.get("states") == 0 or (target in row and row[target] == 0):
        return None
    return float(row[name]) if row.get(name) is not None else None


def check_section(lines, rows):
    latest = rows[-1]
    missing = sum(row.get("time_targets") == 0 for row in rows)
    lines.append(
        f"- 检查点 {len(rows)} 条 · 最近 `{latest['query']}` · "
        f"检查点 `{latest.get('checkpoint')}` · 更新 {latest.get('checked_at')}"
    )
    lines.append(
        f"- 无点估计时间监督 {missing} 条；无适用监督的损失显示为 —，不计作零误差。time/accuracy/censor 均为损失，不是耗时或准确率。"
    )
    lines.append("")

    for name in ("total", "time", "accuracy"):
        values = [value for row in rows if (value := query_loss(row, name)) is not None]
        if not values:
            continue
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"`{sparkline(values)}`")
        lines.append("")
        lines.append(
            f"最小 {min(values):.4f} · 最大 {max(values):.4f} · 均值 {sum(values) / len(values):.4f}"
        )
        lines.append("")

    for grouping, label in ((dataset_of, "数据集"), (family_of, "族")):
        buckets = defaultdict(list)
        for row in rows:
            buckets[grouping(row["query"])].append(row)
        lines.append(f"## 按{label}汇总")
        lines.append("")
        lines.append(
            f"| {label} | 条数 | states 均值 | time 均值 | accuracy 均值 | total 均值 |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|")
        for name in sorted(buckets):
            group = buckets[name]
            count = len(group)

            def mean_loss(term):
                values = [
                    value
                    for row in group
                    if (value := query_loss(row, term)) is not None
                ]
                return f"{sum(values) / len(values):.4f}" if values else "—"

            lines.append(
                f"| {name} | {count} | "
                f"{sum(r.get('states') or 0 for r in group) / count:.1f} | "
                f"{mean_loss('time')} | {mean_loss('accuracy')} | {mean_loss('total')} |"
            )
        lines.append("")


def render(run: Path) -> Path:
    lines = [f"# 模型训练（{run.name}）", ""]
    protocol = run / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()).get("collect_only"):
        lines.append("本次仅收集预训练数据，不启动模型训练或真实执行反馈。")
        target = run / "模型训练.md"
        return publish(target, "\n".join(lines) + "\n")
    configuration = json.loads(protocol.read_text()) if protocol.exists() else {}
    if configuration.get("pretraining_workload_version") != configuration.get(
        "workload_version"
    ):
        lines.extend(
            [
                f"- 保留预训练模型及 v{configuration.get('pretraining_workload_version')} 原始数据；"
                f"后续真实反馈使用 v{configuration.get('workload_version')} 查询。各轮 SQL 见相应 `workloads.json`。",
                "",
            ]
        )
    epochs_by_round = [
        (round_dir, read_epochs(round_dir))
        for round_dir in sorted(run.glob("*/round-*"))
    ]
    epochs_by_round = [(path, epochs) for path, epochs in epochs_by_round if epochs]
    rows = read_checks(run)

    if not epochs_by_round and not rows:
        lines.append(
            "尚无记录：拟合开始后这里会按 epoch 刷新；检查点生成后显示损失汇总。"
        )
        target = run / "模型训练.md"
        return publish(target, "\n".join(lines) + "\n")

    for round_dir, epochs in epochs_by_round:
        epoch_section(lines, round_dir, epochs)
    if rows:
        check_section(lines, rows)

    target = run / "模型训练.md"
    return publish(target, "\n".join(lines) + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: loss_report.py <run-dir>")
    print(render(Path(sys.argv[1]).resolve()))
