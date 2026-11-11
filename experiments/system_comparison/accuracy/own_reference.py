"""Report each configuration against its own system's Teacher answer."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

from experiments.core.io import digest, save
from experiments.system_comparison.accuracy.artifacts import read
from experiments.system_comparison.accuracy.pairwise import table


def calculate(run):
    run = Path(run).resolve()
    source_path = run / "accuracy-multireference.json"
    source = read(source_path)
    if digest(run / "campaign.json") != source["campaign_sha256"]:
        raise ValueError("Campaign changed after the multi-reference scoring")
    references = Path(source["reference_root"])
    if digest(references / "contract.json") != source["reference_contract_sha256"]:
        raise ValueError("Teacher contract changed after the multi-reference scoring")
    pairs = {}
    for item in source["comparisons"]:
        key = item["task_id"], item["reference_system"]
        if key in pairs:
            raise ValueError(f"Duplicate reference comparison: {key}")
        pairs[key] = item

    records = []
    for item in source["records"]:
        key = item["task_id"], item["system"]
        pair = pairs.get(key)
        record = {
            "task_id": item["task_id"], "variant": item["variant"],
            "system": item["system"], "query_id": item["query_id"],
            "status": item["status"], "scoring_status": item["scoring_status"],
            "aggregate": item["aggregate"], "accuracy": None, "note": item.get("note"),
        }
        records.append(record)
        if item["scoring_status"] != "scored":
            continue
        if pair is None:
            if item["system"] in item["reference_systems"]:
                raise ValueError(f"Missing saved comparison for an available reference: {key}")
            record.update(scoring_status="own_reference_unavailable", note="本系统未收集此查询的 Teacher 参考答案")
            continue
        if digest(item["result_path"]) != item["result_sha256"]:
            raise ValueError(f"Result changed after the multi-reference scoring: {key}")
        if digest(pair["reference_path"]) != pair["reference_sha256"]:
            raise ValueError(f"Teacher answer changed after the multi-reference scoring: {key}")
        record.update({
            field: pair[field] for field in (
                "accuracy", "precision", "recall", "matched_rows", "result_rows",
                "reference_rows", "reference_system", "reference_path", "reference_sha256",
                "compared_fields", "excluded_free_text_fields", "comparison_basis",
            )
        })
        record.update(scoring_status="scored", result_path=item["result_path"], result_sha256=item["result_sha256"])

    variants, queries = source["variants"], source["query_ids"]
    index = {(r["variant"], r["query_id"]): r for r in records}
    common = [query for query in queries
              if all(index[variant, query]["scoring_status"] == "scored" for variant in variants)]
    summary = {}
    for variant in variants:
        scored = [r for r in records if r["variant"] == variant and r["scoring_status"] == "scored"]
        shared = [r["accuracy"] for r in scored if r["query_id"] in common]
        summary[variant] = {
            "mean_f1": mean(r["accuracy"] for r in scored) if scored else None,
            "scored_queries": len(scored), "total_queries": len(queries),
            "common_mean_f1": mean(shared) if shared else None, "common_queries": len(shared),
        }
    return {
        "version": "own-native-teacher-non-free-text-row-f1-v1",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "run": str(run), "reference_root": str(references),
        "source_multireference": str(source_path), "source_multireference_sha256": digest(source_path),
        "variants": variants, "query_ids": queries, "common_queries": common,
        "summary": summary, "records": records,
    }


def percent(value):
    return "-" if value is None else f"{100 * value:.2f}%"


def write_report(run, report):
    run = Path(run)
    json_path = run / "accuracy-own-reference.json"
    csv_path = run / "accuracy-own-reference.csv"
    md_path = run / "accuracy-own-reference.md"
    save(json_path, report)
    fields = ["variant", "system", "query_id", "status", "scoring_status", "accuracy",
              "precision", "recall", "matched_rows", "result_rows", "reference_rows",
              "reference_system", "result_path", "reference_path", "note"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["records"])
    variants, queries, records = report["variants"], report["query_ids"], report["records"]
    index = {(r["variant"], r["query_id"]): r for r in records}
    summary = table(
        ["配置", "同系统参考平均 F1", "已评分", f"共同 {len(report['common_queries'])} 条平均 F1"],
        [[v, percent(report["summary"][v]["mean_f1"]),
          f"{report['summary'][v]['scored_queries']}/{len(queries)}",
          percent(report["summary"][v]["common_mean_f1"])] for v in variants],
    )
    grid = table(
        ["查询", *variants],
        [[q, *(percent(index[v, q]["accuracy"]) for v in variants)] for q in queries],
    )
    missing = [r for r in records if r["scoring_status"] != "scored"]
    missing_table = table(
        ["配置", "查询", "状态", "原因"],
        [[r["variant"], r["query_id"], r["status"], str(r.get("note") or r["scoring_status"]).replace("|", "\\|").replace("\n", " ")]
         for r in missing],
    )
    semcache_note = (
        f"SemCache 使用外部实验 `{report['semcache_evaluation_root']}` 及其参考答案；"
        if report.get("semcache_evaluation_root") else ""
    )
    md_path.write_text("\n\n".join([
        "# 各系统对各自 Teacher 答案的 F1",
        f"PLOP 对 PLOP 参考，Sema 对 Sema 参考，Nirvana、LOTUS、Abacus 和 SemCache 各对本系统参考；不包括 MOAR。不调用模型、不重跑查询。{semcache_note}",
        "沿用多参考报告的非自由文本行 F1 口径：聚合查询比较聚合前行，忽略生成的自由文本；行顺序无关、保留重复次数。双方均为空记 100%。",
        "每个配置对其可评分查询等权平均；超时、跳过、缺少本系统参考均记为 `-`，不计为 0。不同系统参考答案各异，因此此表不是统一真值下的排名。",
        "## 汇总", summary,
        "共同查询：" + "、".join(report["common_queries"]) + "。",
        "## 逐查询 F1", grid,
        "## 未评分项", missing_table,
        "[逐查询 CSV](accuracy-own-reference.csv) · [完整 JSON](accuracy-own-reference.json) · [多系统参考的平均 F1](accuracy-multireference.md) · [实验进度](progress.md)",
    ]) + "\n")
    return md_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    report = calculate(args.run)
    print(write_report(args.run, report))
    for variant, summary in report["summary"].items():
        print(variant, percent(summary["mean_f1"]),
              f"{summary['scored_queries']}/{summary['total_queries']}",
              percent(summary["common_mean_f1"]))


if __name__ == "__main__":
    main()
