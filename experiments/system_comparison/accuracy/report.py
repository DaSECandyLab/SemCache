"""Write separate time/F1 matrices and a traceable, machine-readable score table."""

from __future__ import annotations

import csv
import io
from collections import Counter
from pathlib import Path

from experiments.core.io import save
from experiments.system_comparison.accuracy.summary import summarize

REASONS = {
    "aggregate_input_missing": "系统未保存聚合前输入",
    "reference_input_missing": "参考缺少聚合前输入",
    "query_version_mismatch": "查询版本与参考不一致",
    "input_mismatch": "输入数据版本不一致",
    "result_unavailable": "运行未完成或没有保存结果",
    "teacher_pending": "teacher 无法确定自由文本是否正确",
    "invalid_output": "输出格式或类型无效",
    "aggregate_scored": "聚合输出已比较相对偏差",
}


def percent(value):
    return "-" if value is None else f"{100 * value:.2f}%"


def query_version_note(records):
    """Describe the versions actually present, including partially finished reruns."""
    parts = []
    q17 = [r for r in records if r["query_id"] == "tpch/Q17"]
    if q17:
        outdated = sum(not r["query_version_matches"] for r in q17)
        if outdated:
            parts.append(
                "TPC-H Q17 的参考已加入客户键关联；"
                f"当前 {outdated}/{len(q17)} 个配置的查询版本与参考不一致，"
                "这些输出不参与评分，处理时间保留并标注旧版。"
            )
        else:
            parts.append(
                "TPC-H Q17 当前查询已修复客户键关联，冻结 SQL 与参考版本一致。"
            )
            repeated = sum(r.get("repeat_runs") == 2 for r in q17)
            if repeated:
                parts.append(f"其中 {repeated}/{len(q17)} 个配置已按当前版本重跑两次。")
    other = [r for r in records if r["query_id"] != "tpch/Q17"]
    if other:
        query_ids = {r["query_id"] for r in other}
        mismatched = {r["query_id"] for r in other if not r["query_version_matches"]}
        parts.append(
            f"其余 {len(query_ids)} 条查询中，"
            f"{len(query_ids - mismatched)} 条的冻结 Sema SQL 与参考 SQL 完全相同。"
        )
    if all(r["input_matches"] for r in records):
        parts.append("输入数据契约已全部核对。")
    else:
        parts.append("输入数据版本不一致的记录已排除评分。")
    return "".join(parts)


def timing_note(records):
    parts = ["处理时间包含采样、规划和执行，不含输入预加载。"]
    repeated = [r for r in records if r.get("repeat_runs") == 2]
    means = sum(r.get("time_basis") == "two_run_mean" for r in repeated)
    if repeated:
        parts.append(
            f"{len(repeated)} 项记录了两次运行，两次结果分开保存，"
            "主准确率使用最新的第二次结果；"
            f"其中 {means} 项两次均完成且有合法平均时间，表中使用两次平均值。"
            "其余记录显示 summary.seconds 对应的单次时间；"
            "重复运行没有可用平均值时标注单次。"
            "CSV/JSON 保留 seconds 原值、display_seconds、两次时间及保存路径。"
        )
    else:
        parts.append("处理时间读取原 summary.seconds。")
    parts.append(
        "保留缓存重跑、Abacus 无采样诊断等历史配置差异不因重新评分而改变。"
        "固定记录集合以 teacher 参考为准，自由文本按原始来源判定；"
        "自动评审结果不等于已经独立核实的事实真值。"
    )
    return "".join(parts)


def text_review_note(report):
    profiles = report.get("evaluation_profiles", {})
    if not profiles:
        return ""
    version = next(iter(profiles.values()))["version"]
    return (
        f"四条自由文本查询使用 {version}：固定字段匹配后，teacher 仅根据原始 Details/Plot"
        "判断系统提取内容是否正确，不把参考自由文本当作事实真值。Title 仅定位，提取事实必须由"
        "实际输入 Details/Plot 支持。参考仍决定固定记录集合、分母及重复行容量；参考漏行或筛选"
        "判断分歧不由本规则修复。Estate 容许重复、同义词、来源支持的补充和营销语差异，"
        "但须覆盖核心实用设施，价格或仅写卧室数不能代替设施提取。Movies 容许合理主角多解，"
        "两个字段均须通过；错误类型、无关人物及未执行提取的整段剧情不能通过。"
        "合法空来源显式提供给 teacher，不把它当成系统输出格式错误。参考文件保持原样，"
        "其漏提取内容不会作为系统答案必须模仿的内容；原始输出和历史耗时不重写。"
    )


def markdown(report):
    records = report["records"]
    counts = Counter(r["scoring_status"] for r in records)
    relative = report.get("aggregate_metric") == "relative_deviation"
    lines = [
        "# 各系统处理时间、行 F1 与聚合相对偏差"
        if relative
        else "# 各系统处理时间与行 F1",
        "",
        (
            f"生成时间：{report['generated_at']}。"
            f"{len(report['query_ids'])} 条查询 × {len(report['variants'])} 个配置，共 {len(records)} 项。"
        ),
        "",
        (
            f"系统结果：[原实验目录]({report['run']})；"
            f"参考答案：[Sema teacher 答案]({report['references']})。"
        ),
        "",
        (
            "每行先比较全部固定字段（ID、原始字段、数值、固定类别）；"
            "固定字段均相同，才将该行全部自由文本交给 WorkBuddy teacher 判断。"
            "固定字段相同且自由文本通过对应查询规则的行才允许一对一匹配，重复次数保留。"
            "`F1 = 2 × 匹配行数 / (系统行数 + 参考行数)`；双方合法空集为 100%。"
            "不使用字段加权总分，不将未完成或未决结果当作空集。"
        ),
        "",
        (
            "Q14 固定类别及 Q15 固定标签集合按冻结词表规范化比较；"
            "错误类别或非法标签不会交给 teacher 改判。自由文本即使字面相同，也经 teacher 判断后缓存。"
            "数字按 SQL 类型规范化比较，保留 NULL 与空字符串的区别；重复列按位置保留。"
        ),
        "",
        (
            "聚合查询使用最终数值的相对偏差 `|系统值−参考值|/|参考值|`，越低越好，不与行 F1 混合平均。"
            "参考值为 0 时，两者为 0 记 0%；系统非 0 记 ∞（JSON 以标志记录）。"
            "任一值为 NULL 时偏差为 `-`，另记是否同时为 NULL；不把 NULL 当作 0。"
            "偏差允许超过 100%，不截断。"
            if relative
            else "聚合查询比较聚合算子的完整输入行。新参考已保留 10 条查询的输入，"
            "但本轮系统输出没有保存对应输入，用户确认将这些项记为 `-`；"
            "不从 COUNT 或最终 AVG/MIN/MAX/SUM 反造输入，不以最终值分数代替行 F1。"
        ),
        "",
        query_version_note(records),
        "",
        text_review_note(report),
        "",
        timing_note(records),
        "",
        "| 评分情况 | 项数 |",
        "|---|---:|",
    ]
    for status, n in sorted(counts.items()):
        lines.append(
            f"| {'已评分' if status == 'scored' else REASONS.get(status, status)} | {n} |"
        )
    audit = report.get("teacher_audit", {})
    total = audit.get("cumulative", audit)
    usage = total.get("usage", {})
    lines += [
        "",
        (
            f"去重后的自由文本评审候选：{audit.get('unique_pairs', report.get('teacher_pairs', 0))}；"
            f"teacher 累计 HTTP 请求 {total.get('http_requests', 0)} 次，"
            f"输入/输出 tokens：{usage.get('prompt_tokens', 0)}/{usage.get('completion_tokens', 0)}。"
            f"本次新增请求 {audit.get('http_requests', 0)} 次，复用 {audit.get('cache_hits', 0)} 项缓存判断。"
            f"[teacher 判断与调用记录]({report['judgments']})。"
        ),
        "",
    ]
    lookup = {(r["variant"], r["query_id"]): r for r in records}
    if relative:
        summary = summarize(report)
        lines += [
            "## 系统汇总",
            "",
            (
                "行 F1 对可评分查询取算术平均，缺失项不计作 0；覆盖数量不同，均值不是同分母排名。"
                "另报所有配置共同可评分的非空参考集合，以便同条件比较。"
                "聚合均值仅包含参考非零且双方为数值的查询；零参考和 NULL 结果在逐查询表与 CSV 中单列。"
            ),
            "",
            "| 配置 | 非空参考行 F1 | 覆盖 | 含空参考行 F1 | 共同集合行 F1 | 聚合平均偏差（非零参考） | 可计算聚合数 | NULL 不一致数 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for variant in report["variants"]:
            s = summary["variants"][variant]
            lines.append(
                f"| {variant} | {percent(s['nonempty_row_f1_mean'])} | {s['nonempty_row_f1_count']}/{s['nonempty_row_query_total']} | {percent(s['row_f1_mean'])} | {percent(s['common_row_f1_mean'])} | {percent(s['nonzero_aggregate_deviation_mean'])} | {s['nonzero_aggregate_deviation_count']}/{s['nonzero_aggregate_query_total']} | {s['aggregate_null_mismatches']} |"
            )
        lines += [
            "",
            "空参考查询："
            + "、".join(summary["empty_reference_queries"])
            + "。这些空集的语义判断尚未独立确认，逐查询 F1 保留，但同时展示排除它们的平均值。",
            "",
            "共同非空集合：" + "、".join(summary["common_nonempty_row_queries"]) + "。",
            "",
        ]

    def table(title, queries, kind):
        lines.extend(
            [
                f"## {title}",
                "",
                "| 查询 | " + " | ".join(report["variants"]) + " |",
                "|---|" + "---:|" * len(report["variants"]),
            ]
        )
        for query in queries:
            cells = []
            for variant in report["variants"]:
                r = lookup[variant, query]
                if kind == "time":
                    seconds = r.get("display_seconds", r["seconds"])
                    label = "-" if seconds is None else f"{seconds:.2f}"
                    if (
                        r.get("repeat_runs") == 2
                        and r.get("time_basis") != "two_run_mean"
                    ):
                        label += "（单次）"
                    if r["status"] == "timeout":
                        label += "（超时）"
                    if not r["query_version_matches"]:
                        label += "（旧版）"
                elif kind == "deviation":
                    label = (
                        "∞"
                        if r.get("deviation_infinite")
                        else percent(r.get("relative_deviation"))
                    )
                else:
                    label = percent(r["accuracy"])
                cells.append(label)
            lines.append(f"| {query} | " + " | ".join(cells) + " |")
        lines.append("")

    table("处理时间（秒）", report["query_ids"], "time")
    table(
        "准确率：行 F1",
        [
            q
            for q in report["query_ids"]
            if not relative or not report["reference_info"][q]["aggregate"]
        ],
        "accuracy",
    )
    aggregates = [
        q for q in report["query_ids"] if report["reference_info"][q]["aggregate"]
    ]
    table(
        "聚合相对偏差（越低越好）" if relative else "聚合前输入行 F1",
        aggregates,
        "deviation" if relative else "accuracy",
    )
    lines += [
        (
            "所有 `-` 的逐项原因、固定字段及匹配行数见同名 CSV/JSON。"
            "原始输出路径与文件哈希均保存在 JSON 中。"
        ),
        "",
    ]
    return "\n".join(lines)


def write_report(report, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {**report, "summary": summarize(report)}
    save(output.with_suffix(".json"), report)
    output.with_suffix(".md").write_text(markdown(report))
    fields = [
        "variant",
        "system",
        "query_id",
        "status",
        "seconds",
        "display_seconds",
        "time_basis",
        "repeat_runs",
        "initial_seconds",
        "second_seconds",
        "average_seconds",
        "repeat_manifest",
        "repeat_results",
        "accuracy",
        "precision",
        "recall",
        "matched_rows",
        "result_rows",
        "reference_rows",
        "accuracy_basis",
        "aggregate_value",
        "reference_value",
        "absolute_deviation",
        "relative_deviation",
        "deviation_infinite",
        "deviation_status",
        "aggregate_equal",
        "text_evaluation_profile",
        "scoring_status",
        "query_version_matches",
        "input_matches",
        "note",
        "result_path",
        "pre_aggregate_path",
        "reference_path",
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(report["records"])
    output.with_suffix(".csv").write_text(stream.getvalue())
