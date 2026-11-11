"""Read-only, incremental progress for measured training feedback collection."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from optimizer.training.calibration.epoch import calibration_paths, measured_epoch

from .files import publish, report_writer
from .progress import read
from .tables import table

DEFAULT_MODEL_IDS = (
    "qwen3-1.7b",
    "gemma4-e2b",
    "gemma4-e4b",
    "gemma4-12b",
)
MODEL_LABELS = dict(zip(DEFAULT_MODEL_IDS, ("Q1.7B", "E2B", "E4B", "12B"), strict=True))


class Telemetry:
    """Read only newly appended complete JSON lines; retain no prompt or row data."""

    def __init__(self):
        self.files = {}

    def read(self, path):
        empty = {
            "requests": 0,
            "responses": 0,
            "reuse": 0,
            "reports": 0,
            "models": Counter(),
            "first": None,
            "last": None,
            "execution_seconds": None,
        }
        try:
            stat = path.stat()
        except FileNotFoundError:
            return empty
        previous = self.files.get(path)
        identity = (stat.st_dev, stat.st_ino)
        if previous is None or previous[0] != identity or stat.st_size < previous[1]:
            offset, counts = 0, empty
        else:
            _, offset, counts = previous
        with path.open("rb") as stream:
            stream.seek(offset)
            while line := stream.readline():
                if not line.endswith(b"\n"):
                    break
                offset = stream.tell()
                try:
                    event = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    continue
                name = event.get("event")
                if name in ("sampling_llm_submit", "sampling_llm_retransmit"):
                    counts["requests"] += 1
                    counts["models"][event.get("model_id", "?")] += 1
                elif name == "complete":
                    counts["responses"] += 1
                elif name == "semantic_result_reuse":
                    counts["reuse"] += 1
                elif name == "execution_report":
                    counts["reports"] += 1
                    if isinstance(event.get("execution_wall_ms"), (int, float)):
                        counts["execution_seconds"] = event["execution_wall_ms"] / 1000
                timestamp = event.get("observed_unix_seconds")
                if isinstance(timestamp, (int, float)):
                    counts["first"] = counts["first"] or timestamp
                    counts["last"] = timestamp
        self.files[path] = identity, offset, counts
        return counts


TELEMETRY = Telemetry()


def execution(folder):
    counts = TELEMETRY.read(folder / "telemetry.jsonl")
    if (folder / "timeout.json").exists():
        state = "timeout"
    elif (folder / "stderr.log").exists():
        if (
            counts["reports"]
            and (folder / "profile.json").exists()
            and (folder / "result.csv").exists()
        ):
            state = "completed"
        else:
            state = "未确认成功（检查stderr）"
    else:
        state = "执行中/未写完"
    now = datetime.now().timestamp()
    seconds = (
        (now if state == "执行中/未写完" else counts["last"]) - counts["first"]
        if counts["first"]
        else None
    )
    return state, counts, counts["execution_seconds"] or seconds


def query_headers(models):
    return [
        "查询",
        "次",
        "选择",
        "轮",
        "顺序",
        "进度",
        "Beam",
        "状态",
        "执行",
        "上限",
        "评分",
        "总计",
        "F1",
        "扩展",
        "候选",
        "数据",
        *(MODEL_LABELS.get(model, model) for model in models),
    ]


def render(run: Path) -> Path:
    status = read(run / "pipeline-status.json") or {}
    protocol = read(run / "protocol.json") or {}
    models = list(
        read(run / "models.json") or protocol.get("models") or DEFAULT_MODEL_IDS
    )
    headers = query_headers(models)
    lines = [
        f"# 真实训练数据收集（{run.name}）",
        "",
        f"- 流程状态：`{status.get('status', '等待启动')}`；阶段 `{status.get('phase', '?')}`；"
        f"本轮查询完成 {status.get('completed_queries', 0)}/{status.get('total_queries', '?')}；"
        f"真实轮次 {status.get('round', '?')}/{status.get('rounds', '?')}；"
        f"查询顺序 {status.get('query_order', '未知')}",
        f"- 当前查询：`{status.get('current_query') or '无'}`；流程阶段变更时间：{status.get('updated_at', '未知')}",
        "- [预训练数据收集](预训练数据收集.md) · [模型训练与损失](模型训练.md)",
        "",
        "真实反馈共三轮，每轮使用不同固定种子随机打乱查询。第一轮由启发式限定当前最高优先级算子组，"
        "价值模型对候选结构、模型和 Prompt layout 评分；后两轮由价值模型共同搜索结构与实现。"
        "Beam 保留配置的随机探索比例。每条查询独立搜索、执行、评分两次：第一次从最终前沿随机选择；"
        "若它已是第一次搜索中预测质量最高的计划，第二次重新搜索后选最快，否则选预测质量最高的。"
        "F1 使用预先保存的 teacher 参考答案及 "
        "system_comparison.accuracy 行比较，查询完成后立即评分；缺失答案记为待评分，轮末等待并补齐后更新模型。",
        "",
        "下表在查询结束前持续更新。搜索状态数来自 search.json；采样请求来自 sampling.json；"
        "请求和响应从执行日志增量读取，不重复计数。耗时单位均为秒，"
        "Beam 搜索单独计时；总耗时包含采集各阶段及行 F1 比较。"
        "身份不匹配的行计入多返回／漏返回惩罚；同身份内重复行一对一匹配，不复用答案。"
        "超时按每条计划单独判断；超时没有完整结果，F1 保持未知。未落盘的指标显示 —。"
        "SQL 执行序号 1 为 EXPLAIN，之后仅有一次候选执行；参考答案从共享缓存读取，"
        "不会在训练轮中调用 PLOP 或 teacher 生成整查询参考答案。Beam 内有界准确率采样保持独立。",
        "四模型 HTTP 次数为每条查询的 Beam 采样与正式执行之和，含重传，"
        "缓存命中不计；不把 teacher API 计入候选模型。按模型采样统计尚未落盘时显示 —，"
        "统计已落盘但某模型没有调用时显示 0；正式执行部分随日志追加增量更新。",
        "",
    ]
    if protocol.get("pretraining_workload_version") != protocol.get("workload_version"):
        lines.extend(
            [
                f"- 初始化模型使用预训练查询 v{protocol.get('pretraining_workload_version')}；"
                f"本报告的真实反馈查询使用 v{protocol.get('workload_version')}，与参考答案逐条校验 SQL。",
                "",
            ]
        )
    if status.get("error"):
        lines.extend([f"- 错误：`{status['error']}`", ""])
    if status.get("phase") == "waiting_reference_answers":
        pending = status.get("pending_reference_queries", [])
        lines.extend(
            [
                f"- 本轮真实执行已完成，模型更新等待 {len(pending)} 条参考答案；"
                f"参考生成状态：`{status.get('reference_generation_status', '未知')}`。",
                "- 待补参考查询：" + "、".join(pending),
                "",
            ]
        )
    calibration_path = calibration_paths(run)["global"]
    calibration = read(calibration_path) or {}
    live = read(calibration_paths(run)["progress"]) or {}
    if calibration or live:
        from .calibration_report import calibration_snapshot, calibration_tables

        applied = calibration.get("applied", {})
        displayed, pending = calibration_snapshot(run, status, calibration)
        updated = (
            datetime.fromtimestamp(
                calibration_path.stat().st_mtime, ZoneInfo("Asia/Shanghai")
            ).isoformat(timespec="seconds")
            if calibration_path.exists()
            else "-"
        )
        lines.extend(
            [
                "## 当前全局准确率与请求耗时",
                "",
                f"- 全局统计更新：{updated}；已提交预训练查询 "
                f"{sum(key.startswith('estimated/') for key in applied)} 条、真实训练搜索执行 "
                f"{sum(key.startswith('measured/') for key in applied)} 次。",
                f"- 最近提交：`{next(reversed(applied), '-')}`。",
                "- Beam 按模型、语义算子类型和 Prompt layout 读取全局先验；"
                "查询内准确率采样更新当前上下文，查询完成后提交全局统计，供后续查询使用。"
                "下表包含当前查询 Beam 已落盘但尚未全局提交的观测；"
                "Beam 仍在运行时，未落盘观测暂不显示。",
                "- 此处是算子采样输出与 teacher 的一致率；下方逐查询表的准确率 F1 "
                "来自完整结果与冻结参考答案比较，用于价值模型训练，不直接回写为算子全局准确率。",
                "",
            ]
        )
        epoch = measured_epoch(run)
        if epoch is not None:
            lines.extend(
                [
                    f"- 当前校准段：`{epoch['id']}`；teacher thinking=disabled、"
                    "reasoning_effort=none；此段从独立先验开始，"
                    "原预训练全局统计保留在 `calibration/global.json`。",
                    "",
                ]
            )
        if pending:
            saved = datetime.fromtimestamp(
                pending["path"].stat().st_mtime, ZoneInfo("Asia/Shanghai")
            ).isoformat(timespec="seconds")
            lines.extend(
                [
                    f"- 已保存当前 Beam 统计：`{pending['query']}`；保存时间 {saved}；"
                    f"新增判定 {pending['new_judgments']} 条，尚未全局提交。"
                    "表格直接读取该累计快照，查询提交后切回全局值，不重复累加。",
                    "",
                ]
            )
        calibration_tables(run, lines, live, state=displayed, include_relational=False)
    rounds = sorted((run / "measured").glob("round-*"))
    if not rounds:
        lines.extend(
            [
                "本次仅收集预训练数据，未安排真实执行反馈。"
                if protocol.get("collect_only")
                else "真实反馈采集尚未开始。",
                "",
            ]
        )
    rows = []
    for round_dir in rounds:
        meta = read(round_dir / "round.json") or {}
        for path in sorted((round_dir / "cache-segments").glob("resume-*.json")):
            segment = read(path) or {}
            lines.extend(
                [
                    f"- 缓存恢复段 `{segment.get('id', path.stem)}`：第 "
                    f"{round_dir.name.removeprefix('round-')} 轮从 "
                    f"`{segment.get('start_query', '未知')}` 继续，保留此前 "
                    f"{segment.get('start_index', '?')} 条反馈；冷缓存重置状态 "
                    f"`{segment.get('status', 'pending')}`。"
                    "前后两段分别使用各自真实的缓存上下文，原轮次起始记录保留。",
                    "",
                ]
            )
        folders = {
            p.name: p
            for flow in ("cold", "warm", "stream")
            for p in (round_dir / flow).glob("*")
            if p.is_dir()
        }
        queries = meta.get("queries", list(folders))
        paired = protocol.get("feedback_protocol") == "two_search_two_execution_v1"
        episodes = (1, 2) if paired else (1,)
        for name, episode in ((name, slot) for name in queries for slot in episodes):
            query_folder = folders.get(name)
            folder = (
                query_folder / f"search-{episode}"
                if paired and query_folder
                else query_folder
            )
            if folder is not None and not folder.exists():
                folder = None
            selection = read(query_folder / "selection.json") if query_folder else None
            chosen = read(folder / "selected-plan.json") if folder else None
            choice = (chosen or {}).get("selection_basis") or (
                "pareto_random"
                if episode == 1
                else (selection or {}).get("second_search_policy")
            )
            identity = [
                name,
                str(episode),
                {
                    "pareto_random": "前沿随机",
                    "highest_quality": "最高质量",
                    "fastest": "最快",
                }.get(choice, "待定"),
                round_dir.name.removeprefix("round-"),
                {"forward": "正序", "reverse": "反序", "shuffled": "乱序"}.get(
                    meta.get("query_order"), meta.get("query_order", "未知")
                ),
            ]
            if folder is None:
                rows.append(
                    [*identity, "未开始", *["—"] * (len(headers) - len(identity) - 1)]
                )
                continue
            search = read(folder / "search.json")
            candidates = read(folder / "candidate_plans.json")
            n = len(candidates) if isinstance(candidates, list) else None
            timing = read(folder / "timing.json") or {}
            payload = read(folder / "experience.json") or {}
            record = next(iter(payload.get("records", [])), {})
            runs = sorted((folder / "sql").glob("run-*"), key=lambda p: int(p.name[4:]))
            counts_by_phase = Counter()
            entries = []
            for sql in runs:
                number = int(sql.name[4:])
                if number == 1:
                    continue  # EXPLAIN is not a semantic execution.
                phase, key = "执行中", "timing"
                state, counters, seconds = execution(sql)
                if state == "completed":
                    counts_by_phase[key] += 1
                elif state == "timeout":
                    counts_by_phase[key + "_timeout"] += 1
                entries.append((sql, number, phase, state, counters, seconds))
            active = (
                status.get("phase") == "measured"
                and str(status.get("round")) == round_dir.name[6:]
                and status.get("current_query") == name
                and (not paired or status.get("rollout_index") == episode)
            )
            stage = (
                "完成"
                if timing.get("complete")
                else entries[-1][2]
                if entries
                else "待执行"
                if candidates is not None
                else "保存计划"
                if search is not None
                else "搜索中"
            )
            quality = read(folder / "reference.json") or {}
            phases = timing.get("phases", {})
            seconds = entries[-1][-1] if entries else None
            execution_state = "等待执行"
            if record.get("status") == "completed":
                execution_state = "成功"
                seconds = next(iter(record.get("times", [])), seconds)
            elif record.get("status") == "timeout" or counts_by_phase["timing_timeout"]:
                execution_state = "超时"
                seconds = record.get("time_lower_bound", seconds)
            elif entries:
                execution_state = "成功" if counts_by_phase["timing"] else "执行中"
            if active and status.get("status") in ("error", "stopped", "interrupted"):
                stage = "失败" if status.get("status") == "error" else "已停止"
                if not entries:
                    execution_state = "未执行"
            limit = record.get("metadata", {}).get("timeout_seconds")
            if limit is None:
                limit = protocol.get("timeout", {}).get("maximum")
            total = sum(p.get("wall_seconds", 0) for p in phases.values())
            if active and not timing.get("complete") and timing.get("started_at"):
                total = datetime.now().timestamp() - timing["started_at"]
            comparison = quality.get("comparison_seconds")
            if comparison is not None:
                total += comparison
            sampling = read(folder / "sampling.json") or {}
            sample_models = sampling.get("http_requests_by_model")
            execution_models = Counter()
            for _, _, _, _, counters, _ in entries:
                execution_models.update(counters["models"])
            model_requests = []
            for model in models:
                count = (
                    sample_models.get(model, 0)
                    if isinstance(sample_models, dict)
                    else None
                )
                model_requests.append(
                    count + execution_models[model]
                    if type(count) is int and count >= 0
                    else "—"
                )

            def duration(value):
                return f"{value:.2f}" if isinstance(value, (float, int)) else "—"

            rows.append(
                [
                    *identity,
                    stage,
                    duration(phases.get("beam_search", {}).get("wall_seconds")),
                    execution_state,
                    ("≥" if execution_state == "超时" else "") + duration(seconds),
                    duration(limit),
                    duration(comparison),
                    duration(total) if timing else "—",
                    f"{quality['value']:.4f}"
                    if quality.get("value") is not None
                    else "—",
                    search.get("expansions", "?") if search else "未知",
                    n if n is not None else "未知",
                    f"[目录]({folder.relative_to(run)}/)",
                    *model_requests,
                ]
            )
    if rows:
        lines.extend(
            [
                "## 查询总表",
                "",
                "耗时单位均为秒：Beam＝搜索，执行＝计划执行，评分＝F1 计算，总计＝采集总耗时。"
                "末几列为各模型的 HTTP 请求次数；F1 未知显示 —，超时由状态列标记。",
                "模型缩写："
                + "；".join(
                    f"{MODEL_LABELS.get(model, model)}＝{model}" for model in models
                )
                + "。",
                "",
            ]
        )
        table(
            lines,
            headers,
            rows,
            right=tuple(
                i
                for i, name in enumerate(headers)
                if name not in {"查询", "选择", "顺序", "进度", "状态", "数据"}
            ),
        )
    target = run / "真实训练数据收集.md"
    return publish(target, "\n".join(lines) + "\n")


def main():
    """Offline rendering, or read-only attachment to an already running controller."""
    import argparse
    import time
    from contextlib import nullcontext

    from .monitor import alive
    from .progress import _refresh_reports

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    protocol = read(run / "protocol.json") or {}
    interval = (
        protocol.get("configuration", {})
        .get("monitoring", {})
        .get("refresh_seconds", 10)
    )
    with (
        report_writer(run, blocking=True)
        if args.follow
        else nullcontext(True) as acquired
    ):
        if not acquired:
            raise RuntimeError("Reports already have an active writer")
        checked = {
            str(path): path.stat().st_mtime_ns
            for path in run.glob("*/round-*/model/value.pt")
        }
        while True:
            status = read(run / "pipeline-status.json") or {}
            running = alive(status.get("pid"), run)
            checked = _refresh_reports(run, checked)
            if not args.follow or not running:
                return
            time.sleep(max(1, interval))


if __name__ == "__main__":
    main()
