"""Render compact response timing artifacts; raw telemetry is read only on request."""

import json
from pathlib import Path

from .progress import read
from .tables import table


def rebuild_online_history(run):
    """Explicit, restartable derivation; calibration commits and labels are unchanged.

    Saved per-query legacy phases are checkpoints. Rebuild the summary from all
    checkpoints every time, including after an interrupted or deleted summary.
    New collector phases belong to the live global bank and are never recounted.
    """
    from planning.observations.request_timing import merge_phase, record_phase
    from training.artifacts import save

    run = Path(run)
    destination = run / "calibration/online-history.json"
    added = 0
    for telemetry in sorted(
        (run / "measured").glob("round-*/*/*/sql/run-*/telemetry.jsonl")
    ):
        artifact = telemetry.parents[2] / "online-timing.json"
        phases = read(artifact) or {}
        key = "legacy-measured-" + telemetry.parent.name
        if "measured" in phases or key in phases:
            continue
        metadata = {"wall": None, "invalid_lines": 0}

        def events():
            with telemetry.open("rb") as stream:
                for lineno, line in enumerate(stream, 1):
                    try:
                        event = json.loads(line)
                    except (ValueError, UnicodeDecodeError) as error:
                        if not line.endswith(b"\n"):
                            metadata["invalid_lines"] += 1
                            break  # Only an incomplete final record is tolerated.
                        raise ValueError(
                            f"Invalid telemetry at {telemetry}:{lineno}"
                        ) from error
                    if not isinstance(event, dict):
                        raise ValueError(
                            f"Non-object telemetry at {telemetry}:{lineno}"
                        )
                    if event.get("event") == "execution_report":
                        value = event.get("execution_wall_ms")
                        if isinstance(value, (int, float)) and not isinstance(
                            value, bool
                        ):
                            metadata["wall"] = value / 1000
                    if event.get("event") in (
                        "semantic_row",
                        "submit",
                        "sampling_llm_submit",
                        "sampling_llm_retransmit",
                    ):
                        yield event

        summary = record_phase(
            {},
            events(),
            source="measured",
            scope=str(telemetry.parent.resolve()),
            wall_seconds=lambda: metadata["wall"],
            wall_basis="SQL execution_report.execution_wall_ms",
        )
        if not summary["reported_responses"] and not metadata["invalid_lines"]:
            continue
        summary["derived_from"] = str(telemetry.relative_to(run))
        summary["invalid_lines"] = metadata["invalid_lines"]
        summary["input_status"] = "partial" if metadata["invalid_lines"] else "complete"
        phases[key] = summary
        save(artifact, phases)
        added += 1
    history = {"profile": {}, "derived_phases": 0, "partial_phases": 0}
    for artifact in sorted((run / "measured").glob("round-*/*/*/online-timing.json")):
        for key, summary in (read(artifact) or {}).items():
            if not key.startswith("legacy-measured-"):
                continue
            merge_phase(history["profile"], summary)
            history["derived_phases"] += 1
            history["partial_phases"] += summary.get("input_status") == "partial"
    history["basis"] = (
        "explicit legacy measured SQL telemetry derivation; saved calibration commits, labels and "
        "initial priors unchanged; missing sampling archives unknown; incomplete trailing lines marked partial"
    )
    save(destination, history)
    return {
        "artifact": str(destination),
        "added_phases": added,
        "derived_phases": history["derived_phases"],
        "partial_phases": history["partial_phases"],
    }


def response_summary(profiles, model, kind):
    entries = [
        p.get("online_request_profiles", {}).get(model, {}).get(kind, {})
        for p in profiles
    ]
    samples = sum(e.get("samples", 0) for e in entries)
    seconds = sum(e.get("response_seconds_sum", 0.0) for e in entries)
    active = sum(e.get("phase_interval_union_seconds_sum", 0.0) for e in entries)
    covered = sum(e.get("interval_responses", 0) for e in entries)
    return (
        seconds / samples if samples else None,
        samples if samples else None,
        covered / active if active else None,
    )


def online_tables(run, lines, profile, *, display_model, cell):
    """Only compact phase files and fixed global moments are read during refresh."""
    history = read(run / "calibration/online-history.json") or {}
    profiles = [profile, history.get("profile", {})]
    if history:
        lines.extend(
            [
                "",
                f"历史在线观测来源：显式重新统计的 SQL 遥测，共 {history.get('derived_phases', 0)} 个阶段；"
                f"其中 {history.get('partial_phases', 0)} 个阶段有截断末行，统计仅包含已保存完整响应。"
                "重统计文件为 calibration/online-history.json；旧原始遥测、校准提交和训练标签保持原样。",
            ]
        )
    rows = []
    for bank in profiles:
        for model, kinds in bank.get("online_request_profiles", {}).items():
            for kind, entry in kinds.items():
                for source, source_entry in entry.get("sources", {}).items():
                    for cohort in source_entry.get("contexts", {}).values():
                        context = cohort["context"]
                        layout = context.get("layout", {})
                        label = ":".join(layout.get("block_order") or [])
                        label = (
                            label.replace("OPERATOR_SPECIFICATION", "O")
                            .replace("TASK_DESCRIPTION", "T")
                            .replace("DATA_BLOCK", "D")
                        )
                        fields = "/".join(
                            ",".join(layout.get(k) or [])
                            for k in (
                                "field_order",
                                "left_field_order",
                                "right_field_order",
                            )
                        )
                        rows.append(
                            [
                                display_model(model),
                                kind.removeprefix("SEMANTIC_"),
                                source,
                                label + " " + fields,
                                context.get("cache_condition", "unknown"),
                                context.get("sampling_batch_size", 1),
                                cell(cohort.get("samples")),
                                cell(cohort.get("seconds_per_response")),
                                cell(cohort.get("stddev_seconds")),
                                cell(cohort.get("min_seconds"))
                                + "–"
                                + cell(cohort.get("max_seconds")),
                            ]
                        )
    lines.extend(
        [
            "",
            "在线响应耗时与初始隔离校准分别记录。响应耗时包含排队和并发等待，"
            "不能除以并发数作为单请求服务时间。以下仅按相同模型、来源、完整布局、"
            "批大小和已报告缓存类别分桶；cached 只表示缓存 token 大于零。"
            "未知缓存或布局的响应纳入总计和描述性均值，条件分布仅保留在各查询 online-timing.json，"
            "不与已知条件桶混合。",
            "",
            "<details>",
            "<summary>在线响应耗时分布（秒）</summary>",
            "",
        ]
    )
    table(
        lines,
        [
            "模型",
            "算子",
            "来源",
            "布局/字段顺序",
            "缓存",
            "批大小",
            "样本数",
            "均值(s)",
            "标准差(s)",
            "最小–最大(s)",
        ],
        rows,
        right=(5, 6, 7, 8),
    )
    lines.extend(["</details>", ""])
    phase_rows = []
    paths = sorted(run.glob("*/round-*/*/*/online-timing.json"))
    for path in sorted(paths, key=lambda p: p.stat().st_mtime)[-8:]:
        for phase, summary in (read(path) or {}).items():
            for model, kinds in summary.get("models", {}).items():
                for kind, group in kinds.items():
                    latency = group.get("response_latency", {})
                    phase_rows.append(
                        [
                            path.parent.name,
                            phase,
                            display_model(model),
                            kind.removeprefix("SEMANTIC_"),
                            cell(latency.get("samples")),
                            cell(latency.get("seconds_per_response")),
                            cell(group.get("request_interval_union_seconds")),
                            cell(group.get("phase_wall_seconds")),
                            cell(group.get("responses_per_active_second")),
                            cell(group.get("responses_per_wall_second")),
                        ]
                    )
    lines.extend(
        [
            "<details>",
            "<summary>最近八条查询的阶段吞吐</summary>",
            "",
            "最近八条查询的阶段吞吐：分母分别为同一阶段该模型/算子请求区间的并集、"
            "整个阶段的墙钟时间。分子只统计已返回响应；区间缺失时不推算活跃吞吐。"
            "sampling墙钟包括采样初始化与清理，measured墙钟为SQL执行耗时；具体口径保存于phase_wall_basis。"
            "阶段均值是描述性混合值，不回写初始校准。",
            "",
        ]
    )
    table(
        lines,
        [
            "查询",
            "阶段",
            "模型",
            "算子",
            "响应数",
            "响应均值(s)",
            "请求并集(s)",
            "阶段墙钟(s)",
            "活跃响应/s",
            "墙钟响应/s",
        ],
        phase_rows,
        right=(4, 5, 6, 7, 8, 9),
    )

    lines.extend(["</details>", ""])
