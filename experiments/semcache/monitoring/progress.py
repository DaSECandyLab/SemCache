"""Render compact pretraining progress from saved query artifacts."""

from __future__ import annotations

import itertools
import json
import statistics
import sys
import time
from pathlib import Path

from optimizer.training.calibration.epoch import calibration_paths

from .files import publish, report_writer

STAGES = [
    ("setup_and_explain", "setup"),
    ("independent_calibration", "calibration"),
    ("search_problem_setup", "search"),
    ("sampler_initialization", "sampler"),
    ("context_setup", "context"),
    ("features_and_audit", "features"),
    ("objective_estimation", "estimate"),
    ("beam_search", "beam_cpu"),
    ("beam_search", "beam_llm"),
    ("freeze_and_replay", "freeze"),
    ("fit", "fit"),
    ("unfinished_or_cleanup", "cleanup"),
]
LABEL = dict(STAGES)


def read(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def split_id(name: str):
    for marker, split in (
        ("-train-", "train"),
        ("-validation-", "validation"),
        ("-evaluation-", "evaluation"),
    ):
        if marker in name:
            dataset, rest = name.split(marker, 1)
            index, _, family = rest.partition("-")
            return dataset, split, index, family
    return name, "?", "", ""


def llm_seconds(folder: Path) -> float:
    """Model-request time observed during sampling for one query."""
    sampling = read(folder / "sampling.json") or {}
    latency = sampling.get("request_latency") or {}
    value = latency.get("seconds")
    return float(value) if isinstance(value, (int, float)) else 0.0


def stages_of(data: dict, folder: Path) -> tuple[dict[str, float], float]:
    """Stage seconds plus the total wall time (beam split into CPU and LLM)."""
    stages: dict[str, float] = {}
    total = 0.0
    for name, value in (data.get("phases") or {}).items():
        wall = value.get("wall_seconds", 0.0)
        total += wall
        if name == "beam_search":
            if "llm_wall_seconds" in value:
                stages["beam_cpu"] = value["non_llm_wall_seconds"]
                stages["beam_llm"] = value["llm_wall_seconds"]
            else:
                stages["beam_wall"] = wall
                stages["beam_process_cpu"] = value.get("cpu_seconds", 0.0)
                stages["beam_llm_sum"] = llm_seconds(folder)
        else:
            stages[LABEL.get(name, name)] = wall
    for name, value in (data.get("backend_timings") or {}).items():
        wall = value.get("wall_seconds", 0.0)
        # Backend timings are nested diagnostics (for example observe and
        # quality inside beam_search). Keep them as display columns, but do
        # not add them to the elapsed phase total a second time.
        stages[LABEL.get(name, name)] = wall
    return stages, total


def sampling_counts(folder: Path) -> dict:
    """Read the counters shown in pretraining progress, preferring compact metadata."""
    calls = (read(folder / "sampling.json") or {}).get("calls") or {}
    requests = calls.get("model_http_requests", calls.get("model_requests", 0))
    collection = read(folder / "collection.json") or {}
    search = collection.get("search") or {}
    if not isinstance(search.get("expansions"), (int, float)):
        search = read(folder / "search.json") or {}
    records = collection.get("records")
    if not isinstance(records, (int, float)):
        records = len((read(folder / "experience.json") or {}).get("records", []))
    return {
        "requests": int(requests) if isinstance(requests, (int, float)) else 0,
        "expansions": int(search.get("expansions", 0)),
        "records": int(records),
    }


def calibration_progress(run: Path):
    """Count actual calibrated relational kinds and models, rather than files."""
    models = set(read(run / "models.json") or {})
    relational_kinds = {
        "FILTER",
        "PROJECTION",
        "ORDER_BY",
        "LIMIT",
        "AGGREGATE",
        "CROSS_PRODUCT",
    }
    relational = set()
    semantic = {
        kind: set()
        for kind in ("SEMANTIC_FILTER", "SEMANTIC_PROJECTION", "SEMANTIC_JOIN")
    }
    legacy_semantic = {kind: set() for kind in semantic}
    layouts = {
        ":".join(order)
        for order in itertools.permutations(
            ("OPERATOR_SPECIFICATION", "TASK_DESCRIPTION", "DATA_BLOCK")
        )
    }
    live = read(calibration_paths(run)["progress"]) or {}
    layout_protocol = bool(live.get("layout"))
    for path in calibration_paths(run)["probes"].glob("*.json"):
        probe = read(path) or {}
        if path.name.startswith("relational-"):
            relational.update(relational_kinds.intersection(probe.get("profile", {})))
        else:
            kind = probe.get("contract", {}).get("operator_kind")
            if kind in semantic:
                if isinstance(probe.get("layouts"), dict):
                    layout_protocol = True
                    for layout, evidence in probe["layouts"].items():
                        if layout in layouts:
                            semantic[kind].update(
                                (model, layout)
                                for model in models.intersection(
                                    evidence.get("models", {})
                                )
                            )
                else:
                    legacy_semantic[kind].update(
                        models.intersection(probe.get("models", {}))
                    )
    if not layout_protocol:
        semantic = legacy_semantic
    rows = [
        (
            "关系算子单位成本",
            len(relational),
            len(relational_kinds),
            "Filter、Projection、Order By、Limit、Aggregate、Cross Product",
        )
    ]
    for kind, label in (
        ("SEMANTIC_FILTER", "语义 Filter"),
        ("SEMANTIC_PROJECTION", "语义 Projection"),
        ("SEMANTIC_JOIN", "语义 Join"),
    ):
        rows.append(
            (
                label,
                len(semantic[kind]),
                len(models) * (6 if layout_protocol else 1),
                "模型 × 六种布局的准确率；平均单请求耗时"
                if layout_protocol
                else "旧版逐模型准确率与平均单请求耗时",
            )
        )
    return rows, sum(bool(total) and done == total for _, done, total, _ in rows)


def render(run: Path) -> Path:
    status = read(run / "pipeline-status.json") or {}
    monitor = read(run / "monitor-status.json") or {}
    current = status.get("current_query") or monitor.get("current_query")
    total = status.get("total_queries") or monitor.get("total_queries") or 0
    # Pipeline status is written in-process and is fresher than the monitor's
    # snapshot; prefer it for the counters and fall back to the monitor.
    completed = status.get("completed_queries")
    if completed is None:
        completed = monitor.get("completed_queries")
    records = status.get("records")
    if records is None:
        records = monitor.get("records")
    failed = max(monitor.get("failed_queries", 0), int(status.get("status") == "error"))
    alerts = monitor.get("alerts") or []

    lines = [
        f"# 预训练数据收集（{run.name}）",
        "",
    ]
    lines.append(
        f"- 整体流程状态 `{status.get('status')}` · 阶段 `{status.get('phase')}` · "
        f"轮次 `{status.get('global_round', status.get('round'))}/{status.get('total_rounds', '?')}` · 当前流程轮次完成 **{completed}/{total}** · "
        f"记录 {records} · 失败 {failed}"
    )
    lines.append(
        "- 本报告仅列预训练；[真实训练数据收集](真实训练数据收集.md)记录后续轮次的逐查询进度。"
    )
    protocol = read(run / "protocol.json") or {}
    if protocol.get("collect_only"):
        lines.append(
            "- 本次仅收集预训练数据；采集结束后不会启动模型训练或真实执行反馈。"
        )
    if protocol.get("pretraining_workload_version") is not None and protocol[
        "pretraining_workload_version"
    ] != protocol.get("workload_version"):
        lines.append(
            f"- 本页保留预训练查询 v{protocol.get('pretraining_workload_version')} 的原始统计；"
            f"真实反馈已切换为 v{protocol.get('workload_version')}，不会重采或覆盖这些预训练记录。"
        )
    if current:
        lines.append(f"- 当前：`{current}`")
    if status.get("error"):
        lines.append(f"- 错误：`{status['error']}`")
    if status.get("updated_at"):
        lines.append(f"- 更新时间 {status['updated_at']}")
    from .calibration_report import display_model

    services = [
        f"{display_model(name)} {'✔' if info.get('healthy') else '✘'}"
        for name, info in (monitor.get("model_services") or {}).items()
    ]
    if services:
        lines.append("- 服务：" + " · ".join(services))
    lines.append("- 告警：" + ("；".join(alerts) if alerts else "无"))
    lines.append(
        "- 缓存：预训练每条查询结束后清空候选模型 KV cache 和请求历史；查询内部连续采样。"
        "模型损失见 [模型训练.md](模型训练.md)。"
    )
    lines.append(
        "- 计时（秒）：Beam 包含采样和搜索；标签为估算监督标签的生成耗时；"
        "总计还包含初始化和结果保存。"
    )
    lines.append(
        "- HTTP：本查询实际发送的采样、校准及 teacher 请求；响应缓存命中不计。"
        "完整统计保留在各查询的 JSON 文件中。"
    )
    if (run / "字段深度统计复核.md").exists():
        lines.append("- 离线复核：[字段深度统计复核](字段深度统计复核.md)")
    if status.get("network_retry"):
        retry = status["network_retry"]
        lines.append(
            f"- 网络恢复：第 {retry['attempt']} 次失败，等待 {retry['wait_seconds']} 秒后重试；失败记录 `{retry['archive']}`"
        )
    if status.get("query_order"):
        lines.append(
            f"- 本轮查询顺序：`{status['query_order']}`；完整顺序见各轮的 `round.json`"
        )
    _, calibration_done = calibration_progress(run)
    live = read(calibration_paths(run)["progress"]) or {}
    lines.extend(["## 初始全局校准", ""])
    lines.append(
        f"- 完成 **{calibration_done}/4** 组校准 "
        "（1 组关系成本，含 6 类关系算子；3 类语义算子，每类覆盖候选模型池）"
    )
    if "cache_hits" in live:
        lines.append(
            f"- 本地校准缓存：命中 **{live['cache_hits']}/{live.get('groups', 4)}** 组；"
            "复用初始统计，不恢复模型 KV cache 或请求历史。"
        )
    if status.get("phase") == "initial_calibration":
        lines.append("- 当前阶段：正在建立关系单位成本及模型准确率、单请求耗时先验")
        kind_label = {
            "SEMANTIC_FILTER": "语义 Filter",
            "SEMANTIC_PROJECTION": "语义 Projection",
            "SEMANTIC_JOIN": "语义 Join",
        }
        if live.get("phase") == "relational":
            lines.append(
                f"- 进度：关系算子单位成本 {live.get('done')}/{live.get('total')} 类"
                f" · 已用 {live.get('elapsed_seconds')} 秒"
            )
        elif live.get("phase") == "semantic":
            done, total = live.get("done") or 0, live.get("total") or 0
            filled = int(20 * done / total) if total else 0
            bar = "█" * filled + "░" * (20 - filled)
            lines.append(
                f"- 进度：{kind_label.get(live.get('kind'), live.get('kind'))}"
                f"（第 {live.get('kind_index')}/{live.get('kinds')} 类）"
                f" `{bar}` {done}/{total} 次测量"
                f" · 模型 `{live.get('model')}`"
                f" · 每项 {live.get('rows')} 行"
                f" · 已用 {live.get('elapsed_seconds')} 秒"
            )
    elif calibration_done == 4:
        lines.append("- 当前阶段：初始全局校准已完成")
    elif calibration_done:
        lines.append("- 当前阶段：初始全局校准已部分完成")
    else:
        lines.append("- 当前阶段：尚无已完成的初始全局校准结果")
    from .calibration_report import calibration_tables

    calibration_tables(run, lines, live)

    rounds = sorted(
        {
            path.parent.parent.parent
            for path in run.glob("estimated/round-*/**/timing.json")
        },
        key=str,
    )
    for round_dir in rounds:
        rows = []
        query_order = (read(round_dir / "round.json") or {}).get("queries", [])
        positions = {query: index for index, query in enumerate(query_order)}
        paths = sorted(
            round_dir.glob("*/**/timing.json"),
            key=lambda path: (
                positions.get(path.parent.name, len(positions)),
                str(path),
            ),
        )
        for path in paths:
            data = read(path) or {}
            name = path.parent.name
            stages, wall_total = stages_of(data, path.parent)
            rows.append(
                {
                    "name": name,
                    "stage": stages,
                    "sampling": sampling_counts(path.parent),
                    "beam_seconds": (
                        (data.get("phases") or {})
                        .get("beam_search", {})
                        .get("wall_seconds")
                    ),
                    "complete": bool(data.get("complete")),
                    "seconds": wall_total,
                }
            )
        if not rows:
            continue
        active_round = str(round_dir.relative_to(run)) == (
            f"{status.get('phase')}/round-{status.get('round')}"
        )
        for row in rows:
            row["status"] = (
                "completed"
                if row["complete"]
                else {
                    "running": "running",
                    "error": "failed",
                    "stopped": "stopped",
                }.get(status.get("status"), "incomplete")
                if active_round and row["name"] == current
                else "incomplete"
            )
        counts: dict[str, int] = {}
        for row in rows:
            key = row["status"]
            counts[key] = counts.get(key, 0) + 1
        beam_times = [
            row["beam_seconds"]
            for row in rows
            if row["complete"] and row["beam_seconds"] is not None
        ]

        lines.append(f"## {round_dir.relative_to(run)}")
        lines.append("")
        lines.append(
            f"已结束 **{sum(row['complete'] for row in rows)}/{total or '?'}**；状态计数：`{counts}`"
            + (
                f"；Beam 中位数 {statistics.median(beam_times):.1f} 秒"
                if beam_times
                else ""
            )
        )
        lines.append("")
        keys = ("requests", "expansions", "records")
        totals = {key: sum(row["sampling"][key] for row in rows) for key in keys}
        lines.append(
            f"合计：计划 {totals['records']} · HTTP {totals['requests']} · "
            f"扩展 {totals['expansions']}"
        )
        lines.append("")
        headers = [
            "查询",
            "类型",
            "状态",
            "计划数",
            "HTTP",
            "Beam(s)",
            "标签(s)",
            "总计(s)",
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append(
            "|"
            + "|".join(
                ("---" if index < 3 else "---:") for index in range(len(headers))
            )
            + "|"
        )
        for row in rows:
            _, _, index, family = split_id(row["name"])
            state = row["status"]
            sampling = row["sampling"]
            beam = row["beam_seconds"]
            labels = row["stage"].get("estimated_labels")
            values = [
                index or row["name"],
                family,
                state,
                str(sampling["records"])
                if row["complete"] or sampling["records"]
                else "—",
                str(sampling["requests"])
                if row["complete"] or sampling["requests"]
                else "—",
                f"{beam:.1f}" if beam is not None else "—",
                f"{labels:.1f}" if labels is not None else "—",
                f"{row['seconds']:.1f}",
            ]
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    target = run / "预训练数据收集.md"
    publish(target, "\n".join(lines) + "\n")
    (run / "实验数据采集.md").unlink(missing_ok=True)
    return target


def refresh_reports(run: Path, checked=None):
    """Refresh Markdown; evaluate loss only when a fitted checkpoint changes."""
    with report_writer(run) as acquired:
        if not acquired:
            return checked
        return _refresh_reports(run, checked)


def _refresh_reports(run: Path, checked=None):
    from .loss_report import render as render_loss

    checkpoints = {
        str(p): p.stat().st_mtime_ns for p in run.glob("*/round-*/model/value.pt")
    }
    waiting = (read(run / "pipeline-status.json") or {}).get(
        "phase"
    ) == "waiting_shared_resources"
    if not waiting and checkpoints != (checked or {}):
        from .loss_check import check

        check(run)
    from .collection_report import render as render_collection

    render_loss(run)
    render(run)
    render_collection(run)
    return (checked or {}) if waiting else checkpoints


def follow(run: Path):
    """Refresh alongside an older monitor without restarting its launcher."""
    target = render(run)

    def versions():
        paths = calibration_paths(run)
        sources = [
            target,
            paths["global"],
            paths["progress"],
            run / "calibration/measured-epoch.json",
            run / "pipeline-status.json",
        ]
        return tuple(p.stat().st_mtime_ns if p.exists() else None for p in sources)

    previous = versions()
    while (read(run / "pipeline-status.json") or {}).get("status") == "running":
        time.sleep(1)
        current = versions()
        if current != previous:
            render(run)
            previous = versions()
    render(run)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3) or (
        len(sys.argv) == 3 and sys.argv[2] != "--follow"
    ):
        raise SystemExit("usage: progress.py <run-dir> [--follow]")
    run = Path(sys.argv[1]).resolve()
    if len(sys.argv) == 3:
        follow(run)
    else:
        print(render(run))
