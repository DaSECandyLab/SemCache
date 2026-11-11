"""Render measured global priors and layout evidence without model calls."""

import itertools
import math
from pathlib import Path

from optimizer.training.calibration.epoch import calibration_paths

from .online_timing import online_tables, response_summary
from .progress import read
from .tables import table

KINDS = ("SEMANTIC_FILTER", "SEMANTIC_PROJECTION", "SEMANTIC_JOIN")
BLOCKS = ("OPERATOR_SPECIFICATION", "TASK_DESCRIPTION", "DATA_BLOCK")
RELATIONS = ("FILTER", "PROJECTION", "ORDER_BY", "LIMIT", "AGGREGATE", "CROSS_PRODUCT")
LAYOUTS = tuple(":".join(order) for order in itertools.permutations(BLOCKS))


def display_model(model):
    name = Path(model).name
    if name == "gemma-4-E2B-it-qat-w4a16-ct":
        return "gemma4-e2b"
    return name


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def cell(value, *, percent=False):
    if not number(value):
        return "-"
    if percent:
        return f"{value:.1%}"
    return str(value) if type(value) is int else f"{value:.4f}"


def pooled(entries, value, count):
    measured = [
        e
        for e in entries
        if number(e.get(value)) and number(e.get(count)) and e[count] > 0
    ]
    size = sum(e[count] for e in measured)
    return (
        (sum(e[value] * e[count] for e in measured) / size, size)
        if size
        else (None, None)
    )


def calibration_snapshot(run, status=None, state=None):
    """Display saved Beam evidence without committing or adding it twice."""
    state = state if state is not None else read(calibration_paths(run)["global"]) or {}
    status = status if status is not None else read(run / "pipeline-status.json") or {}
    phase, iteration, query = (
        status.get("phase"),
        status.get("round"),
        status.get("current_query"),
    )
    episode = status.get("rollout_index") if phase == "measured" else None
    suffix = f"/search-{episode}" if episode in (1, 2) else ""
    if (
        status.get("status") != "running"
        or not isinstance(state.get("profile"), dict)
        or phase not in ("estimated", "measured")
        or iteration is None
        or not query
        or f"{phase}/{iteration}/{query}{suffix}" in state.get("applied", {})
    ):
        return state, None
    for flow in ("stream", "cold", "warm"):
        folder = run / phase / f"round-{iteration}" / flow / query
        if suffix:
            folder /= f"search-{episode}"
        path = folder / "calibration-update.json"
        update = read(path) or {}
        if (
            update.get("before") != state.get("profile")
            or not isinstance(update.get("after"), dict)
            or update["after"].get("calibration_protocol") != 2
        ):
            continue

        def judgments(profile):
            return sum(
                entry.get("total", 0)
                for model in profile.get("accuracy", {}).values()
                for entry in model.values()
            )

        return {**state, "profile": update["after"]}, {
            "query": query,
            "path": path,
            "new_judgments": judgments(update["after"]) - judgments(update["before"]),
        }
    return state, None


def calibration_tables(run, lines, live, *, state=None, include_relational=True):
    if state is None:
        state = read(calibration_paths(run)["global"]) or {}
    profile = state.get("profile", {})
    history = (read(run / "calibration/online-history.json") or {}).get("profile", {})
    models = list(read(run / "models.json") or {})
    teacher = state.get("contract", {}).get("teacher", {}).get("model_id") or live.get(
        "teacher_model"
    )
    layouts = {kind: dict(live.get("measurements", {}).get(kind, {})) for kind in KINDS}
    relational = {}
    for path in sorted(calibration_paths(run)["probes"].glob("*.json")):
        probe = read(path) or {}
        kind = probe.get("contract", {}).get("operator_kind")
        if kind in layouts:
            layouts[kind].update(probe.get("layouts", {}))
            teacher = teacher or probe.get("contract", {}).get("teacher", {}).get(
                "model_id"
            )
        elif path.name.startswith("relational-"):
            relational.update(probe.get("profile", {}))
    relational.update(profile.get("relational_profiles", {}))
    if teacher and teacher not in models:
        models.append(teacher)
    lines.extend(
        [
            "",
            "初始请求耗时与样本数来自隔离校准，保留为成本先验；在线响应样本数来自后续采样和实际执行，"
            "随查询提交累计更新。响应(s)是所有在线布局/缓存条件的描述性混合均值；有效响应/s为各阶段"
            "有区间的响应数之和除以各自请求区间并集时长之和（不跨查询合并时间轴）。"
            "分桶分布与阶段吞吐见下方。准确率是与 teacher 的加权采样一致率。"
            "尚无观测显示 `-`，真实的零值保留；旧实验仅显示显式重新统计后保存的在线观测。",
            "",
        ]
    )
    rows = []
    for model in models:
        label = display_model(model) + ("（teacher）" if model == teacher else "")
        for kind in KINDS:
            entries = [
                v.get("models", {}).get(model, {}) for v in layouts[kind].values()
            ]
            mean, samples = pooled(entries, "seconds_per_request", "samples")
            accuracy, judged = pooled(entries, "value", "total")
            latency = profile.get("request_profiles", {}).get(model, {}).get(kind, {})
            quality = profile.get("accuracy", {}).get(model, {}).get(kind, {})
            mean, samples = (
                latency.get("seconds_per_request", mean),
                latency.get("samples", samples),
            )
            accuracy, judged = (
                quality.get("value", accuracy),
                quality.get("total", judged),
            )
            if model == teacher:
                accuracy, judged = None, None
            online_mean, online_samples, throughput = response_summary(
                [profile, history], model, kind
            )
            rows.append(
                [
                    label,
                    kind.removeprefix("SEMANTIC_"),
                    cell(mean),
                    cell(samples),
                    cell(online_mean),
                    cell(online_samples),
                    cell(throughput),
                    cell(accuracy, percent=True),
                    cell(judged),
                ]
            )
    table(
        lines,
        [
            "模型",
            "语义算子",
            "校准(s)",
            "校准n",
            "响应(s)",
            "在线n",
            "有效响应/s",
            "准确率",
            "判定n",
        ],
        rows,
        right=(2, 3, 4, 5, 6, 7, 8),
    )
    lines.extend(
        [
            "<details>",
            "<summary>六种 Prompt 布局的详细数据</summary>",
            "",
            "O＝算子说明，T＝任务描述，D＝输入数据。逐布局耗时来自初始校准；准确率优先读取最新全局布局统计。"
            "每格为“耗时 · 准确率（耗时样本数／判定样本数）”，两种样本数分别统计。"
            "教师作为比较参照，不计算其自身准确率；没有保存的教师耗时显示 `-`。",
            "",
        ]
    )
    rows = []
    for kind in KINDS:
        for model in models:
            quality = (
                profile.get("accuracy", {})
                .get(model, {})
                .get(kind, {})
                .get("layouts", {})
            )
            cells = []
            for key in LAYOUTS:
                entry = layouts[kind].get(key, {}).get("models", {}).get(model, {})
                current = quality.get(key, entry)
                latency = cell(entry.get("seconds_per_request"))
                samples = cell(entry.get("samples"))
                accuracy = cell(
                    current.get("value") if model != teacher else None, percent=True
                )
                judged = cell(current.get("total") if model != teacher else None)
                seconds = f"{latency}s" if latency != "-" else "-"
                cells.append(
                    f"{seconds} · {accuracy} ({samples}/{judged})"
                    if seconds != "-" or accuracy != "-"
                    else "-"
                )
            rows.append([display_model(model), kind.removeprefix("SEMANTIC_"), *cells])
    table(
        lines,
        [
            "模型",
            "语义算子",
            *(
                key.replace("OPERATOR_SPECIFICATION", "O")
                .replace("TASK_DESCRIPTION", "T")
                .replace("DATA_BLOCK", "D")
                .replace(":", "-")
                for key in LAYOUTS
            ),
        ],
        rows,
        right=tuple(range(2, 8)),
    )
    lines.extend(["</details>", ""])
    online_tables(run, lines, profile, display_model=display_model, cell=cell)
    if not include_relational:
        return
    lines.extend(["关系算子的单位输入耗时（不调用 LLM）：", ""])
    table(
        lines,
        ["关系算子", "单位输入耗时(μs)", "校准次数"],
        [
            [
                kind,
                cell(relational.get(kind, {}).get("seconds_per_input") * 1e6)
                if number(relational.get(kind, {}).get("seconds_per_input"))
                else "-",
                cell(relational.get(kind, {}).get("samples")),
            ]
            for kind in RELATIONS
        ],
        right=(1, 2),
    )
