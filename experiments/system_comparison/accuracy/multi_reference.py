"""Compare saved results with every available native-system Teacher answer."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

from experiments.core.io import digest, save
from experiments.core.paths import ROOT
from experiments.system_comparison.accuracy.artifacts import read, unique_schema
from experiments.system_comparison.accuracy.matching import (
    AGGREGATE_FIELDS,
    TEXT_FIELDS,
    score_non_free_text,
)
from experiments.system_comparison.accuracy.pairwise import load_answers, table
from experiments.system_comparison.reporting.results import value_for_type
from experiments.validation.workload.cte import inline_single_use


def equivalent_program(old, new, system):
    """Accept unchanged translations or the verified single-use CTE rewrite."""
    if old == new:
        return "identical"
    try:
        if system in {"plop", "sema"}:
            rewritten, _ = inline_single_use(old, system=system)
            equal = rewritten == new
        else:
            old_tree, new_tree = ast.parse(old), ast.parse(new)

            def sql_node(tree):
                return next(
                    node for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "RELATIONAL_SQL"
                            for target in node.targets)
                )

            node = sql_node(old_tree)
            rewritten, _ = inline_single_use(ast.literal_eval(node.value))
            node.value = ast.Constant(value=rewritten)
            equal = ast.dump(old_tree, include_attributes=False) == ast.dump(new_tree, include_attributes=False)
    except (ValueError, TypeError, SyntaxError, StopIteration):
        equal = False
    if not equal:
        raise ValueError(f"Unverified translated query change for {system}")
    return "verified_single_use_cte_rewrite"


def verified_database(catalog, reference_root, system, dataset, saved_hash):
    """Check physical snapshot or its recorded, verified PLOP migration."""
    entry = catalog["systems"][system][dataset]
    if not all(table["schema_equal"] and table["bag_equal"] for table in entry["tables"].values()):
        raise ValueError(f"Database row-set equivalence is not verified: {system}/{dataset}")
    if saved_hash == entry["sha256"]:
        return "same_database"
    if system == "plop" and saved_hash == entry.get("retired_database_sha256"):
        evidence = read(reference_root / "runtime/plop-vector128/verification" / dataset / "verified.json")
        if (evidence["source_sha256"] == saved_hash and evidence["sha256"] == entry["sha256"]
                and evidence.get("catalog_equal") and evidence.get("views_equal")
                and all(item.get("schema_equal") and item.get("bag_equal")
                        and item.get("typed_rows_equal") and item.get("row_order_equal")
                        for item in evidence["tables"].values())):
            return "verified_plop_vector_migration"
    raise ValueError(f"Database hash is not covered by verified input: {system}/{dataset}")


def reference_result(root, evidence, answer, aggregate):
    if not aggregate:
        return answer["result"], Path(evidence["answer_path"]).parent / "result.json"
    info = answer.get("pre_aggregate")
    if not info:
        raise ValueError("Teacher answer has no aggregate input")
    folder = Path(evidence["answer_path"]).parent
    path = Path(info["path"])
    if not path.is_file():
        # Resolve relocated artifacts relative to their run directory.
        path = folder / path.name
    if not path.is_file() or digest(path) != info["sha256"]:
        raise ValueError(f"Teacher aggregate input missing or hash changed: {folder}")
    return read(path), path


def canonical_result(payload, canonical_schema, query, side):
    """Align original columns and numeric values without judging free text."""
    schema = unique_schema(payload["columns"])
    names = [name for name, _ in schema]
    wanted = [name for name, _ in canonical_schema]
    if set(names) != set(wanted) or len(names) != len(wanted):
        raise ValueError(f"Output columns differ: {names} != {wanted}")
    indexes = [names.index(name) for name in wanted]
    omitted = set(AGGREGATE_FIELDS[query][1] if query in AGGREGATE_FIELDS else TEXT_FIELDS.get(query, ()))
    generated_numeric = {"Price", "RuntimeMinutes", "BoxOfficeUSD"}
    invalid = Counter()
    rows = []
    for row_number, row in enumerate(payload["rows"]):
        if not isinstance(row, (list, tuple)) or len(row) != len(names):
            raise ValueError("Output row width differs from its schema")
        values = []
        for idx, (name, kind) in zip(indexes, canonical_schema, strict=True):
            value = row[idx]
            if name in omitted:
                values.append(None)
                continue
            if name in generated_numeric and isinstance(value, str) and value.strip().casefold() == "null":
                value = None
            if kind.upper() in {"VARCHAR", "TEXT"} and value is not None and not isinstance(value, str):
                # Some Python frames infer an integer type for a nullable source
                # text column. The declared source type remains authoritative.
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    value = str(value)
            try:
                value = value_for_type(value, kind)
            except (ValueError, TypeError, OverflowError, ArithmeticError):
                # Invalid numeric/model output must not accidentally match an
                # invalid value or NULL in a reference, and still counts as a row.
                invalid[name] += 1
                value = f"__invalid_{side}_{row_number}_{name}__"
            values.append(value)
        rows.append(values)
    return {"columns": canonical_schema, "rows": rows}, dict(invalid)


def calculate(run, reference_root):
    run, reference_root = Path(run).resolve(), Path(reference_root).resolve()
    campaign = read(run / "campaign.json")
    frozen = read(run / "inputs/manifest.json")
    catalog = read(ROOT / "benchmarks/databases/manifest.json")
    ref_contract, answers, evidence = load_answers(reference_root)
    if not set(campaign["query_ids"]).issubset(ref_contract["query_ids"]):
        raise ValueError("A campaign query has no reference workload")
    run_manifest = {(r["system"], r["id"]): r for r in read(run / "query-manifest.json")["records"]}
    ref_manifest = {(r["system"], r["id"]): r for r in read(reference_root / "query-manifest.json")["records"]}
    systems = ref_contract["systems"]
    ref_by_query, schemas, ref_counts = {}, {}, {}
    for query in campaign["query_ids"]:
        dataset = "movies_nirvana" if query.startswith("movies/") else query.split("/")[0]
        entries = {}
        metadata = [ref_manifest[system, query] for system in systems]
        if len({json.dumps(item.get("aggregate"), sort_keys=True) for item in metadata}) != 1:
            raise ValueError(f"Reference aggregate definitions differ: {query}")
        aggregate = bool(metadata[0].get("aggregate"))
        if aggregate != (query in AGGREGATE_FIELDS):
            raise ValueError(f"No verified non-free-text aggregate-input rule: {query}")
        for system in systems:
            key = f"{system}/{query}"
            if key not in answers:
                if key not in ref_contract["skipped_queries"]:
                    raise ValueError(f"Unexpected missing Teacher answer: {key}")
                continue
            if run_manifest[system, query]["source_sha256"] != ref_manifest[system, query]["source_sha256"]:
                raise ValueError(f"Original query differs from reference: {key}")
            answer = read(evidence[key]["answer_path"])
            db_policy = verified_database(catalog, reference_root, system, dataset, answer["database_sha256"])
            payload, path = reference_result(reference_root, evidence[key], answer, aggregate)
            entries[system] = {
                "payload": payload, "path": str(path), "sha256": digest(path),
                "answer_path": evidence[key]["answer_path"],
                "answer_sha256": evidence[key]["answer_sha256"],
                "database_equivalence": db_policy,
            }
        if not entries or "sema" not in entries:
            raise ValueError(f"No canonical Sema reference schema: {query}")
        schemas[query] = unique_schema(entries["sema"]["payload"]["columns"])
        for system, item in entries.items():
            item["canonical"], item["invalid_values"] = canonical_result(item.pop("payload"), schemas[query], query, "reference_" + system)
        ref_counts[query] = list(entries)
        ref_by_query[query] = entries

    variants = [v["name"] for v in campaign["variants"] if v["system"] != "moar"]
    records, paired, equivalence = [], [], {}
    for task in campaign["tasks"]:
        system, query = task["system"], task["query_id"]
        if system == "moar":
            continue
        folder = run / task["folder"]
        summary_path = folder / "summary.json"
        summary = read(summary_path) if summary_path.exists() else {}
        record = {
            "task_id": task["task_id"], "variant": task["variant"], "system": system,
            "query_id": query, "status": summary.get("status", "missing"),
            "scoring_status": "result_unavailable", "accuracy": None,
            "reference_count": 0, "reference_systems": ref_counts[query],
            "aggregate": query in AGGREGATE_FIELDS,
            "note": summary.get("reason"),
        }
        records.append(record)
        if summary.get("status") != "completed":
            continue
        job = read(folder / "job.json")
        query_file = Path(job["query_file"])
        run_meta, old_meta = run_manifest[system, query], ref_manifest[system, query]
        if (not query_file.is_file() or digest(query_file) != job["query_sha256"]
                or run_meta["sha256"] != job["query_sha256"]
                or run_meta["source_sha256"] != old_meta["source_sha256"]):
            raise ValueError(f"Executed query lineage failed: {task['task_id']}")
        version_key = (system, query)
        if version_key not in equivalence:
            suffix = ".sql" if system in {"plop", "sema"} else ".py"
            old = (reference_root / "queries" / system / (query + suffix)).read_text()
            equivalence[version_key] = equivalent_program(old, query_file.read_text(), system)
        dataset = "movies_nirvana" if query.startswith("movies/") else query.split("/")[0]
        expected_database = frozen["jobs"][query]["input_databases"][system]["sha256"]
        if job.get("input_database_sha256") != expected_database:
            raise ValueError(f"Executed database differs from frozen manifest: {task['task_id']}")
        db_policy = verified_database(catalog, reference_root, system, dataset, expected_database)
        if record["aggregate"]:
            info = summary.get("pre_aggregate") or {}
            path = folder / "pre-aggregate-result.json"
            if not path.is_file() or not info.get("sha256") or digest(path) != info["sha256"]:
                record.update(scoring_status="aggregate_input_missing", note="聚合前输入缺失或校验失败")
                continue
        else:
            path = folder / "result.json"
            if not path.is_file():
                record.update(note="结果文件缺失")
                continue
        try:
            actual, invalid = canonical_result(read(path), schemas[query], query, "actual")
            scores, pair_records = [], []
            for ref_system, ref in ref_by_query[query].items():
                result = score_non_free_text(actual, ref["canonical"], query)
                if result["scoring_status"] != "scored":
                    raise ValueError(f"Unexpected scoring status: {result['scoring_status']}")
                pair_records.append({
                    "task_id": task["task_id"], "variant": task["variant"],
                    "query_id": query, "reference_system": ref_system,
                    "reference_path": ref["path"], "reference_sha256": ref["sha256"],
                    "reference_invalid_values": ref["invalid_values"],
                    **result,
                })
                scores.append(result["accuracy"])
            paired.extend(pair_records)
            record.update(
                accuracy=mean(scores), reference_count=len(scores), scoring_status="scored",
                result_path=str(path), result_sha256=digest(path), result_rows=len(actual["rows"]),
                invalid_values=invalid, translated_query_equivalence=equivalence[version_key],
                database_equivalence=db_policy,
            )
        except (KeyError, ValueError, TypeError, ArithmeticError) as error:
            record.update(scoring_status="invalid_output", note=str(error))
    by_key = {(r["variant"], r["query_id"]): r for r in records}
    common = [q for q in campaign["query_ids"]
              if all(by_key[v, q]["scoring_status"] == "scored" for v in variants)]
    stats = {}
    for variant in variants:
        eligible = [r for r in records if r["variant"] == variant and r["scoring_status"] == "scored"]
        shared = [r["accuracy"] for r in eligible if r["query_id"] in common]
        stats[variant] = {
            "mean_f1": mean(r["accuracy"] for r in eligible) if eligible else None,
            "scored_queries": len(eligible), "total_queries": len(campaign["query_ids"]),
            "common_mean_f1": mean(shared) if shared else None, "common_queries": len(shared),
        }
    return {
        "version": "all-native-teacher-non-free-text-row-f1-v1",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "run": str(run), "reference_root": str(reference_root),
        "campaign_sha256": digest(run / "campaign.json"),
        "reference_contract_sha256": digest(reference_root / "contract.json"),
        "dataset_catalog_sha256": digest(ROOT / "benchmarks/databases/manifest.json"),
        "query_ids": campaign["query_ids"], "variants": variants,
        "reference_systems": systems, "available_references": ref_counts,
        "common_queries": common, "summary": stats, "records": records,
        "comparisons": paired,
    }


def pct(value):
    return "-" if value is None else f"{100 * value:.2f}%"


def write_report(run, report):
    run = Path(run)
    json_path = run / "accuracy-multireference.json"
    csv_path = run / "accuracy-multireference-pairs.csv"
    md_path = run / "accuracy-multireference.md"
    save(json_path, report)
    fields = ["variant", "query_id", "reference_system", "accuracy", "precision", "recall",
              "matched_rows", "result_rows", "reference_rows", "reference_path", "reference_sha256"]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["comparisons"])
    variants, queries, records = report["variants"], report["query_ids"], report["records"]
    index = {(r["variant"], r["query_id"]): r for r in records}
    summary = table(
        ["配置", "逐查询平均 F1", "已评分", f"共同 {len(report['common_queries'])} 条平均 F1"],
        [[v, pct(report["summary"][v]["mean_f1"]),
          f"{report['summary'][v]['scored_queries']}/{len(queries)}",
          pct(report["summary"][v]["common_mean_f1"])] for v in variants],
    )
    grid = table(
        ["查询", "参考数", *variants],
        [[q, str(len(report["available_references"][q])),
          *(pct(index[v, q]["accuracy"]) for v in variants)] for q in queries],
    )
    missing = [r for r in records if r["scoring_status"] != "scored"]
    missing_table = table(
        ["配置", "查询", "状态", "原因"],
        [[r["variant"], r["query_id"], r["status"], str(r.get("note") or r["scoring_status"]).replace("|", "\\|").replace("\n", " ")]
         for r in missing],
    )
    reference_systems = report.get("reference_systems", [])
    reference_names = "、".join(reference_systems)
    semcache_note = (
        f"另纳入外部 SemCache 参考答案，目录见 `{report['semcache_reference_root']}`；"
        if report.get("semcache_reference_root") else ""
    )
    md_path.write_text("\n\n".join([
        "# 各系统对全部可用 Teacher 答案的准确率",
        "只读取已保存结果，未重跑查询、未调用模型；不包括 MOAR。",
        f"每条查询分别与 {reference_names} 的可用 Teacher 答案计算行 F1，再对这些 F1 等权平均。{semcache_note} Appstore 和 TPC-H 没有 Abacus 参考，使用其余可用参考；其他查询使用全部可用参考。",
        "聚合查询比较同次执行保存的聚合前行，比较记录 ID、数值及其他非生成文本字段，忽略生成的自由文本。F1=2×匹配行数/(系统行数+参考行数)；忽略行顺序、保留重复行，双方空集记 100%。参考答案不是经人工验证的独立真值，分数反映与这些 Teacher 答案的一致性。",
        "数值型生成字段按数值对齐，合法 JSON `null` 转为 NULL；无效值不能匹配任何参考值。输入数据库及参考文件按哈希验证，旧 PLOP 数据库经已有完整数据迁移记录验证；翻译查询变化仅接受已验证的单次 CTE 内联。",
        "## 汇总",
        "每个配置对其可评分查询等权平均；超时、跳过、缺失不计为 0。覆盖数不同的全量均值不能直接当成同分母排名；共同查询均值使用所有配置均可评分的同一批查询。",
        summary,
        "共同查询：" + "、".join(report["common_queries"]) + "。",
        "## 逐查询平均 F1",
        "每个单元格已对该行的全部可用参考答案取平均；`-` 表示无可评分的系统结果。",
        grid,
        "## 未评分的系统结果",
        missing_table,
        f"[逐参考答案评分 CSV]({os.path.relpath(csv_path, run)}) · [完整评分及校验信息 JSON]({os.path.relpath(json_path, run)}) · [实验进度](progress.md)",
    ]) + "\n")
    return md_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--references", required=True, type=Path)
    args = parser.parse_args()
    report = calculate(args.run, args.references)
    path = write_report(args.run, report)
    print(json.dumps({"report": str(path), "summary": report["summary"],
                      "states": dict(Counter(r["scoring_status"] for r in report["records"]))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
