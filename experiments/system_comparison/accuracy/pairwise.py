"""Compare saved native-system Teacher answers without executing queries."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime
from itertools import permutations
from pathlib import Path
from statistics import mean
from zoneinfo import ZoneInfo

from experiments.core.io import digest, save

from .aggregates import aggregate_deviation
from .artifacts import read, typed_result
from .matching import TEXT_FIELDS, finish_match, prepare_match


def normalized_text(value):
    """Compare unordered lists conservatively, without semantic or API judgments."""
    if value is None:
        return None
    value = unicodedata.normalize("NFKC", value).casefold()
    parts = []
    for part in re.split(r"[,;\n\r，；]+", value):
        part = re.sub(r"^[\s•*\-]+", "", part)
        part = " ".join(part.split()).rstrip(".!。")
        if part:
            parts.append(part)
    return json.dumps(sorted(parts), ensure_ascii=False)


def offline_result(result, query):
    fields = TEXT_FIELDS.get(query, ())
    indexes = {i for i, (name, _) in enumerate(result["columns"]) if name in fields}
    return {
        "columns": result["columns"],
        "rows": [
            [
                normalized_text(value) if i in indexes else value
                for i, value in enumerate(row)
            ]
            for row in result["rows"]
        ],
    }


def load_answers(root):
    """Check frozen programs and exact saved-output hashes, including moved runs."""
    contract = read(root / "contract.json")
    manifest = {
        (item["system"], item["id"]): item
        for item in read(root / "query-manifest.json")["records"]
    }
    answers, evidence = {}, {}
    skipped = set(contract["skipped_queries"])
    for query in contract["query_ids"]:
        sources = {
            manifest[system, query]["source_sha256"] for system in contract["systems"]
        }
        if len(sources) != 1:
            raise ValueError(f"Different source queries: {query}")
        for system in contract["systems"]:
            key = f"{system}/{query}"
            if key in skipped:
                continue
            folder = root / system / query
            answer = read(folder / "answer.json")
            summary = read(folder / "summary.json")
            job = read(folder / "job.json")
            metadata = manifest[system, query]
            suffix = ".sql" if system in {"plop", "sema"} else ".py"
            query_file = root / "queries" / system / (query + suffix)
            if not (
                answer["status"] == "teacher_generated"
                and summary["status"] == "completed"
                and answer["query_id"] == query
                and answer["system"] == system
                and answer["query_sha256"]
                == job["query_sha256"]
                == metadata["sha256"]
                == digest(query_file)
                and answer["database_sha256"] == job["input_database_sha256"]
                and answer["result_sha256"] == digest(folder / "result.json")
                and answer["result"] == read(folder / "result.json")
            ):
                raise ValueError(f"Answer integrity failure: {key}")
            result = typed_result(answer["result"])
            answers[key] = result
            evidence[key] = {
                "answer_path": str(folder / "answer.json"),
                "answer_sha256": digest(folder / "answer.json"),
                "result_sha256": answer["result_sha256"],
                "database_sha256": answer["database_sha256"],
                "query_sha256": answer["query_sha256"],
                "source_sha256": metadata["source_sha256"],
                "rows": len(result["rows"]),
                "aggregate": metadata.get("aggregate"),
                "value": result["rows"][0][0] if metadata.get("aggregate") else None,
            }
    return contract, answers, evidence


def prepare(root):
    contract, answers, evidence = load_answers(root)
    records = []
    for candidate, reference in permutations(contract["systems"], 2):
        for query in contract["query_ids"]:
            left, right = f"{candidate}/{query}", f"{reference}/{query}"
            record = {
                "task_id": f"{candidate}__{reference}/{query}",
                "candidate": candidate,
                "reference": reference,
                "query_id": query,
                "aggregate": (evidence.get(left) or evidence.get(right))["aggregate"],
                "accuracy": None,
                "scoring_status": "answer_unavailable",
            }
            records.append(record)
            if left not in answers or right not in answers:
                continue
            actual, expected = answers[left], answers[right]
            if record["aggregate"]:
                record.update(aggregate_deviation(actual, expected))
            else:
                case, pending = prepare_match(
                    offline_result(actual, query),
                    offline_result(expected, query),
                    query,
                    {},
                    text_fields=[],
                )
                assert not pending, "Offline comparison must never request a teacher"
                record.update(finish_match(case, {}))
    return contract, evidence, records


def pct(value):
    return "-" if value is None else f"{value * 100:.2f}%"


def deviation(record):
    if record.get("deviation_infinite"):
        return "∞"
    if record.get("deviation_status") == "both_null":
        return "NULL= NULL"
    if record.get("deviation_status") == "null_mismatch":
        return "NULL≠数值"
    return pct(record.get("relative_deviation"))


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|",
            *("| " + " | ".join(map(str, row)) + " |" for row in rows),
        ]
    )


def write_outputs(output, report):
    output.mkdir(parents=True, exist_ok=True)
    save(output / "scores.json", report)
    with (output / "scores.csv").open("w", newline="") as stream:
        columns = [
            "candidate",
            "reference",
            "query_id",
            "scoring_status",
            "accuracy",
            "result_rows",
            "reference_rows",
            "matched_rows",
            "aggregate_value",
            "reference_value",
            "relative_deviation",
            "deviation_infinite",
            "deviation_status",
        ]
        writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["records"])
    systems, queries = report["systems"], report["query_ids"]
    records, evidence = report["records"], report["evidence"]
    by_key = {(r["candidate"], r["reference"], r["query_id"]): r for r in records}
    common = [q for q in queries if all(f"{s}/{q}" in evidence for s in systems)]
    nonaggregate = [
        q
        for q in queries
        if not next(
            v["aggregate"] for k, v in evidence.items() if k.split("/", 1)[1] == q
        )
    ]

    def matrix(select, metric):
        rows = []
        for a in systems:
            row = [a]
            for b in systems:
                if a == b:
                    row.append("—")
                    continue
                subset = [by_key[a, b, q] for q in select]
                values = [r[metric] for r in subset if r.get(metric) is not None]
                value = pct(mean(values)) if values else "-"
                infinite = sum(r.get("deviation_infinite", False) for r in subset)
                row.append(
                    f"{value} ({len(values)}/{len(select)})"
                    + (f"；∞×{infinite}" if infinite else "")
                )
            rows.append(row)
        return table(["待评系统 \\ 参考系统", *systems], rows)

    agg_queries = [q for q in queries if q not in nonaggregate]
    summary = [
        "# 五系统 Teacher 答案两两比较",
        "",
        "评分状态：已完成；完全离线，API调用0次。",
        "",
        f"参考输出：{report['references']}。完整性验证 {len(evidence)}/130 通过；明确跳过10项。",
        "",
        "行是待评系统，列是参考系统。行F1对称，聚合相对偏差有方向性。比较的是五套 Teacher 输出之间的差异，不能据此认定任何一套是独立真值；相同的错误输出也可能一致。",
        "",
        "普通查询：比较所有输出字段。固定字段按既有类型和分类格式规则匹配；四条自由文本查询只做Unicode NFKC、大小写、空白、行首列表符号、句末句点归一化，并将逗号/分号/换行分隔的条目排序后精确匹配。NULL与空字符串区分；不判断同义词、提取完整性或事实正确性。行顺序忽略、重复行按容量匹配。F1 = 2×匹配行数/(待评行数+参考行数)。双方空集记100%，单方空集记0%；下方另报排除双方空集的均值。",
        "",
        "聚合查询：最终标量相对偏差 |待评值−参考值|/|参考值|，越低越接近，不并入行F1。参考0且待评非0显示∞，双方0为0%；NULL不参与数值均值，详细表单独标记。缺失显示-，不按0处理。括号为参与均值条数/该口径总查询数。",
        "",
        "## 行F1：五系统共同覆盖",
        "",
        f"共同覆盖{len(common)}条，其中{len(set(common) & set(nonaggregate))}条非聚合查询。",
        "",
        matrix([q for q in common if q in nonaggregate], "accuracy"),
        "",
        "## 行F1：全部非聚合查询",
        "",
        "涉及Abacus时覆盖较少，不能将不同覆盖的均值直接用于系统排名。",
        "",
        matrix(nonaggregate, "accuracy"),
        "",
        "## 聚合值相对偏差",
        "",
        "仅对双方均有有限数值的条目计算均值。NULL/∞会使覆盖率下降，需结合逐查询表阅读。",
        "",
        matrix(agg_queries, "relative_deviation"),
        "",
        "## 空结果与非空覆盖",
        "",
    ]
    coverage = []
    for a, b in permutations(systems, 2):
        subset = [by_key[a, b, q] for q in nonaggregate]
        scored = [r for r in subset if r.get("accuracy") is not None]
        nonempty = [r for r in scored if r["result_rows"] + r["reference_rows"]]
        coverage.append(
            [
                a,
                b,
                len(scored),
                len(scored) - len(nonempty),
                pct(mean(r["accuracy"] for r in nonempty)) if nonempty else "-",
            ]
        )
    summary += [
        table(["待评", "参考", "已评分", "双方空集", "排除双方空集的平均F1"], coverage),
        "",
        "[逐查询行F1](row-f1.md) · [逐查询聚合偏差](aggregate-deviation.md) · [输出行数及聚合原值](outputs.md) · [评分明细](scores.json) · [CSV](scores.csv)",
        "",
        "本次直接比较保存结果，不执行查询、不调用模型、不修改参考答案。",
        "",
    ]
    (output / "summary.md").write_text("\n".join(summary))
    for filename, selected, metric in [
        ("row-f1.md", nonaggregate, "accuracy"),
        ("aggregate-deviation.md", agg_queries, "relative_deviation"),
    ]:
        lines = [
            "# " + ("逐查询行F1" if metric == "accuracy" else "逐查询聚合值相对偏差"),
            "",
            "行是待评系统，列是参考系统。—为同系统未比较，-为不可用。",
            "",
        ]
        for q in selected:
            lines += ["## " + q.replace("movies/", "movies_nirvana/"), ""]
            rows = []
            for a in systems:
                rows.append(
                    [
                        a,
                        *(
                            "—"
                            if a == b
                            else (
                                pct(by_key[a, b, q].get("accuracy"))
                                if metric == "accuracy"
                                else deviation(by_key[a, b, q])
                            )
                            for b in systems
                        ),
                    ]
                )
            lines += [table(["待评 \\ 参考", *systems], rows), ""]
        (output / filename).write_text("\n".join(lines))
    output_rows = []
    for q in queries:
        row = [q.replace("movies/", "movies_nirvana/")]
        for s in systems:
            item = evidence.get(f"{s}/{q}")
            row.append(
                "-"
                if item is None
                else (
                    ("NULL" if item["value"] is None else str(item["value"]))
                    if item["aggregate"]
                    else f"{item['rows']} 行"
                )
            )
        output_rows.append(row)
    (output / "outputs.md").write_text(
        "# 各系统保存的输出\n\n非聚合为行数，聚合为最终原值；NULL是SQL空值，-是跳过。\n\n"
        + table(["查询", *systems], output_rows)
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root, output = args.references.resolve(), args.output.resolve()
    contract, evidence, records = prepare(root)
    report = {
        "version": "offline-teacher-system-pairwise-v1",
        "status": "complete",
        "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "references": str(root),
        "contract_sha256": digest(root / "contract.json"),
        "systems": contract["systems"],
        "query_ids": contract["query_ids"],
        "evidence": evidence,
        "records": records,
        "http_requests": 0,
        "text_protocol": "normalized-exact-text-v1; no semantic judge",
    }
    write_outputs(output, report)
    print(
        f"Validated {len(evidence)} answers; {len(records)} directed query comparisons; "
        "no API calls",
        flush=True,
    )
    print(dict(Counter(r["scoring_status"] for r in records)), flush=True)
    print(f"Saved {output / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
