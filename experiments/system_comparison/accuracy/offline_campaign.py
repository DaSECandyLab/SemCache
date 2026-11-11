"""Score each saved benchmark configuration against its native Teacher answers."""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from experiments.core.io import digest, save

from .aggregates import aggregate_deviation
from .artifacts import read, timing_fields, typed_result
from .matching import finish_match, prepare_match
from .pairwise import deviation, load_answers, offline_result, pct, table


def compare_result(actual, expected, query, aggregate):
    actual = typed_result(actual, expected["columns"])
    if aggregate:
        return aggregate_deviation(actual, expected)
    case, jobs = prepare_match(
        offline_result(actual, query),
        offline_result(expected, query),
        query,
        {},
        text_fields=[],
    )
    if jobs:
        raise RuntimeError("Offline scoring cannot request model judgments")
    return finish_match(case, {})


def load_native_campaign(run, references):
    """Validate each configuration against its own system's frozen answers."""
    run, references = Path(run).resolve(), Path(references).resolve()
    contract, answers, evidence = load_answers(references)
    campaign = read(run / "campaign.json")
    if set(campaign["query_ids"]) != set(contract["query_ids"]):
        raise ValueError("Campaign and Teacher reference query sets differ")
    records, results = [], {}
    for task in campaign["tasks"]:
        query, system = task["query_id"], task["system"]
        folder = run / task["folder"]
        summary_path, result_path = folder / "summary.json", folder / "result.json"
        summary = read(summary_path) if summary_path.exists() else {}
        reference_key = f"{system}/{query}"
        item = evidence.get(reference_key)
        record = {
            "task_id": task["task_id"],
            "variant": task["variant"],
            "system": system,
            "query_id": query,
            "reference_system": system,
            "status": summary.get("status", "missing"),
            "scoring_status": "result_unavailable",
            "accuracy": None,
            **timing_fields(summary),
            "result_path": str(result_path),
            "reference_rows": item["rows"] if item else None,
            "reference_path": item["answer_path"] if item else None,
            "reference_sha256": item["answer_sha256"] if item else None,
            "aggregate": next(
                evidence[f"{s}/{query}"]["aggregate"]
                for s in contract["systems"]
                if f"{s}/{query}" in evidence
            ),
        }
        records.append(record)
        if summary.get("status") != "completed" or not result_path.exists():
            record["note"] = summary.get("reason") or record["status"]
            continue
        if item is None:
            record.update(
                scoring_status="reference_unavailable",
                note="该系统未收集此条Teacher答案",
            )
            continue
        job = read(folder / "job.json")
        answer = read(Path(item["answer_path"]))
        suffix = ".sql" if system in {"plop", "sema"} else ".py"
        frozen = run / "queries" / system / (query + suffix)
        query_matches = (
            frozen.exists()
            and digest(frozen) == job.get("query_sha256") == answer["query_sha256"]
        )
        database_hash = job.get("input_database_sha256") or job.get(
            "input_databases", {}
        ).get(system, {}).get("sha256")
        accepted_hashes = {answer["database_sha256"]}
        if answer.get("original_database_sha256"):
            accepted_hashes.add(answer["original_database_sha256"])
        record.update(
            query_version_matches=query_matches,
            input_matches=database_hash in accepted_hashes,
            input_database_sha256=database_hash,
            result_sha256=digest(result_path),
            summary_sha256=digest(summary_path),
            input_match_basis="same_physical_database"
            if database_hash == answer["database_sha256"]
            else "recorded_pre_migration_database",
        )
        if not query_matches:
            record.update(
                scoring_status="query_version_mismatch", note="查询版本不同，未评分"
            )
            continue
        if database_hash not in accepted_hashes:
            record.update(
                scoring_status="input_mismatch", note="输入数据版本不同，未评分"
            )
            continue
        try:
            results[task["task_id"]] = (
                typed_result(read(result_path), answers[reference_key]["columns"]),
                answers[reference_key],
            )
            record["scoring_status"] = "ready"
        except (ValueError, KeyError, TypeError) as error:
            record.update(scoring_status="invalid_output", note=str(error))
    return {
        "version": "offline-campaign-own-teacher-v1",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "run": str(run),
        "references": str(references),
        "campaign_sha256": digest(run / "campaign.json"),
        "reference_contract_sha256": digest(references / "contract.json"),
        "query_ids": campaign["query_ids"],
        "variants": campaign["variants"],
        "records": records,
        "reference_policy": "each configuration uses its own system's saved Teacher answer",
        "reference_info": evidence,
    }, results


def score_campaign(run, references):
    report, results = load_native_campaign(run, references)
    for record in report["records"]:
        if record["task_id"] not in results:
            continue
        try:
            record.update(
                compare_result(
                    *results[record["task_id"]], record["query_id"], record["aggregate"]
                )
            )
        except (ValueError, KeyError, TypeError) as error:
            record.update(scoring_status="invalid_output", note=str(error))
    report.update(
        http_requests=0, text_protocol="normalized-exact-text-v1; no semantic judge"
    )
    return report


def write_report(report, output):
    output.mkdir(parents=True, exist_ok=True)
    save(output / "scores.json", report)
    records, queries = report["records"], report["query_ids"]
    variants = [v["name"] for v in report["variants"]]
    index = {(r["variant"], r["query_id"]): r for r in records}
    aggregates = [q for q in queries if index[variants[0], q]["aggregate"]]
    columns = [
        "variant",
        "system",
        "reference_system",
        "query_id",
        "status",
        "scoring_status",
        "accuracy",
        "precision",
        "recall",
        "result_rows",
        "reference_rows",
        "matched_rows",
        "display_seconds",
        "aggregate_value",
        "reference_value",
        "relative_deviation",
        "deviation_infinite",
        "deviation_status",
        "result_path",
        "reference_path",
        "result_sha256",
        "reference_sha256",
        "query_version_matches",
        "input_matches",
        "teacher_unresolved",
        "note",
    ]
    with (output / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    def grid(selected, render, *, base=output):
        def cell(record):
            value = render(record)
            if value == "-":
                return value
            target = os.path.relpath(record["result_path"], base)
            return f"[{value}]({target})"

        return table(
            ["查询", *variants],
            [[q, *(cell(index[v, q]) for v in variants)] for q in selected],
        )

    def row_score(record):
        return pct(record.get("accuracy"))

    def seconds(record):
        return (
            f"{record['display_seconds']:.2f}"
            if record["status"] == "completed"
            and record.get("display_seconds") is not None
            else "-"
        )

    semantic = report.get("evaluation_mode") == "semantic"
    text_rule = (
        "固定字段先精确对齐；固定类别按现有确定性规则比较。自由文本沿用现有来源约束语义评分，由指定teacher根据原始Details/Plot判断提取是否有效；参考确定固定记录集和重复容量。"
        if semantic else
        "自由文本仅归一化Unicode、大小写、空白、列表分隔符及顺序，不判断同义或事实有效性。"
    )
    common = [
        f"正式实验：{report['run']}。",
        f"参考答案：{report['references']}。",
        "每列是独立的正式实验配置，每行是一条查询；PLOP和Sema各保留四列。",
        "每个配置对照同系统收集的Teacher答案：PLOP四配置对照PLOP Teacher，Sema四配置对照Sema Teacher，其余同理。五系统的Teacher答案不同，因此这里是各自参考下的符合程度，不能视为采用同一独立真值的系统排名。",
        (f"只读取保存的实验输出，不重跑查询。语义判断配置：{report['teacher_config']}；判断缓存、请求与用量保存在 [judgments/]({os.path.relpath(report['judgments'], output)}/)。"
         if semantic else "完全离线评分；不调用API、不运行查询、不修改实验和参考原始结果。"),
    ]
    summary_blocks = []
    pending_count = sum(r["scoring_status"] == "teacher_pending" for r in records)
    status_line = (
        f"评分尚未完成：{pending_count} 项查询等待语义判断；当前汇总仅包含已评分项。"
        if pending_count else
        f"评分完成：{sum(r['scoring_status'] == 'scored' for r in records)} 项行F1、"
        f"{sum(r['scoring_status'] == 'aggregate_scored' for r in records)} 项聚合偏差；"
        f"{sum(r['scoring_status'] not in {'scored', 'aggregate_scored'} for r in records)} 项不可评分。"
    )
    if report.get("summary"):
        summary = report["summary"]
        save(output / "summary.json", summary)
        summary_blocks = [
            "## 各配置汇总",
            "宏平均对每条可评分的非聚合查询等权；排除空参考的均值仅使用同系统参考非空的查询。微平均先汇总匹配行数再计算F1。缺失不按0计，聚合偏差不并入F1。",
            table(
                ["配置", "宏平均F1", "已评分/非聚合查询", "非空参考宏平均F1", "已评分/非空参考查询", "微平均F1", "共同非空查询F1"],
                [[v, pct(s["row_f1_mean"]), f"{s['row_f1_count']}/{s['row_query_total']}",
                  pct(s["nonempty_row_f1_mean"]),
                  f"{s['nonempty_row_f1_count']}/{s['nonempty_row_query_total']}",
                  pct(s["micro_row_f1"]),
                  pct(s["common_row_f1_mean"])]
                 for v, s in summary["variants"].items()],
            ),
            "全部配置共同可评分且各自参考非空的查询：" +
            ("、".join(summary["common_nonempty_row_queries"]) or "无") + "。",
        ]
    (output / "accuracy.md").write_text(
        "\n\n".join(
            [
                "# 正式实验逐配置、逐查询准确率",
                status_line,
                *common,
                "使用行F1 = 2×匹配行数/(实验行数+参考行数)。" + text_rule + "忽略行顺序，保留重复次数。",
                f"{len(queries)}条全部列出。{len(aggregates)}条聚合查询的行F1为-，其最终数值相对偏差单独列在聚合表。其他-表示超时、取消、缺失或不可评分，原因见明细。双方均为空集按约定F1为100%，不证明答案正确。",
                *summary_blocks,
                grid(queries, row_score),
                "[处理时间](time.md) · [聚合值相对偏差](aggregate-deviation.md) · [缺失原因](unavailable.md) · [全部明细JSON](scores.json) · [CSV](scores.csv)",
            ]
        )
        + "\n"
    )
    (output / "aggregate-deviation.md").write_text(
        "\n\n".join(
            [
                "# 正式实验逐查询聚合值相对偏差",
                *common,
                "|实验值−同系统Teacher值|/|同系统Teacher值|，越低越接近。双方0为0%，参考0而实验非0为∞；NULL单独标记，不强行赋予分数。",
                grid(aggregates, deviation),
                "[返回逐查询准确率](accuracy.md)",
            ]
        )
        + "\n"
    )
    (output / "time.md").write_text(
        "\n\n".join(
            [
                "# 正式实验逐查询处理时间",
                f"数据来源：{report['run']}。单位：秒。",
                "使用已完成任务保存的耗时；若原摘要包含有效的两次运行均值，采用该均值。超时、取消显示-；不使用Teacher收集耗时。",
                grid(queries, seconds),
                "[返回逐查询准确率](accuracy.md)",
            ]
        )
        + "\n"
    )
    missing = [
        r for r in records if r["scoring_status"] not in {"scored", "aggregate_scored"}
    ]
    (output / "unavailable.md").write_text(
        "# 不可评分项\n\n"
        + table(
            ["配置", "查询", "原因"],
            [
                [r["variant"], r["query_id"], r.get("note") or r["scoring_status"]]
                for r in missing
            ],
        )
        + "\n"
    )
    run = Path(report["run"])
    output_link = os.path.relpath(output, run)
    (run / "accuracy.md").write_text(
        "\n\n".join(
            [
                "# 五系统逐查询时间与准确率",
                status_line,
                "查询和配置顺序与 [progress.md](progress.md) 一致。PLOP、Sema各四个配置独立列出；数值可点击查看实验输出。",
                *summary_blocks,
                "## 处理时间（秒）",
                grid(queries, seconds, base=run),
                "## 准确率（行F1）",
                "正式实验输出对照同系统Teacher答案。聚合查询见下表；超时、取消或缺失为-。双方空集记100%。" + text_rule,
                grid(queries, row_score, base=run),
                "## 聚合值相对偏差",
                "相对偏差 = |实验值−Teacher值| / |Teacher值|，越低越接近；NULL单独标记。",
                grid(aggregates, deviation, base=run),
                f"参考答案：{report['references']}。",
                f"[评分规则与明细]({output_link}/accuracy.md) · [缺失原因]({output_link}/unavailable.md) · [CSV]({output_link}/scores.csv)",
            ]
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run, references = args.run.resolve(), args.references.resolve()
    output = args.output.resolve() if args.output else run / "accuracy"
    report = score_campaign(run, references)
    write_report(report, output)
    print(dict(Counter(r["scoring_status"] for r in report["records"])))
    print(run / "accuracy.md")


if __name__ == "__main__":
    main()
