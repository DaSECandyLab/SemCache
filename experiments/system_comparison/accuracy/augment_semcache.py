"""Add an externally completed SemCache evaluation to comparison reports."""

from __future__ import annotations

import argparse
import csv
import json
import os
from copy import deepcopy
from pathlib import Path
from statistics import mean

from experiments.core.io import digest, save
from experiments.system_comparison.accuracy.artifacts import read, result_digest, unique_schema
from experiments.system_comparison.accuracy.matching import AGGREGATE_FIELDS, score_non_free_text
from experiments.system_comparison.accuracy.multi_reference import (
    calculate as calculate_multi_reference,
    canonical_result,
    write_report as write_multi_reference,
)
from experiments.system_comparison.accuracy.own_reference import (
    calculate as calculate_own_reference,
    write_report as write_own_reference,
)
from experiments.system_comparison.reporting.results import value_for_type


SEM_CACHE_VARIANT = "semcache-adaptive20"
SEM_CACHE_SYSTEM = "semcache"


def typed_csv(path, schema, null_value):
    """Read SemCache's CSV artifact under the declared result schema."""
    columns = unique_schema(schema)
    names = [name for name, _ in columns]
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = []
        for raw in csv.DictReader(stream):
            row = []
            for name, kind in columns:
                value = raw.get(name)
                if value == null_value:
                    value = None
                elif value is not None:
                    value = value_for_type(value, kind)
                row.append(value)
            rows.append(row)
    return {"columns": columns, "rows": rows}


def semcache_paths(eval_root, reference_root, query):
    evaluation = Path(eval_root) / "min-50" / "translated" / "semcache" / query
    reference = Path(reference_root) / "translated" / "semcache" / query
    summary_path = evaluation / "summary.json"
    answer_path = reference / "answer.json"
    summary, answer = read(summary_path), read(answer_path)
    if summary.get("status") != "completed" or answer.get("status") != "completed":
        raise ValueError(f"SemCache query is incomplete: {query}")
    schema = unique_schema(answer["columns"])
    ref_payload = {"columns": schema, "rows": answer["rows"]}
    aggregate = query in AGGREGATE_FIELDS
    if aggregate:
        info = answer.get("pre_aggregate")
        if not info:
            raise ValueError(f"SemCache aggregate reference is missing: {query}")
        ref_pre = Path(info["path"])
        if not ref_pre.is_file():
            ref_pre = reference / "execution" / "pre-aggregate-result.json"
        if not ref_pre.is_file() or result_digest(read(ref_pre)) != info["result_sha256"]:
            raise ValueError(f"SemCache aggregate reference hash mismatch: {query}")
        ref_payload = read(ref_pre)
        schema = unique_schema(info["columns"])
        ref_payload = {"columns": schema, "rows": ref_payload["rows"]}
    artifact = Path(summary["result_artifact"])
    null_value = read(artifact / "result-contract.json").get("null_value")
    if aggregate:
        actual_path = artifact / "pre-aggregate-result.json"
        actual_payload = read(actual_path)
    else:
        actual_path = artifact / "result.csv"
        actual_payload = typed_csv(actual_path, schema, null_value)
    actual_payload, _ = canonical_result(actual_payload, schema, query, "actual_semcache")
    ref_payload, _ = canonical_result(ref_payload, schema, query, "reference_semcache")
    score = score_non_free_text(actual_payload, ref_payload, query)
    return {
        "query": query,
        "summary": summary,
        "actual": actual_payload,
        "reference": ref_payload,
        "actual_path": str(actual_path),
        "reference_path": str(ref_pre if aggregate else reference / "execution" / "result.json"),
        "actual_sha256": digest(actual_path),
        "reference_sha256": digest(ref_pre if aggregate else reference / "execution" / "result.json"),
        "score": score,
        "aggregate": aggregate,
    }


def load_semcache(eval_root, reference_root, queries):
    return {query: semcache_paths(eval_root, reference_root, query) for query in queries}


def payload_from_path(path):
    return read(path)


def score_payloads(actual, expected, query):
    schema = unique_schema(expected["columns"])
    actual, _ = canonical_result(actual, schema, query, "actual_external")
    expected, _ = canonical_result(expected, schema, query, "reference_external")
    return score_non_free_text(actual, expected, query)


def pair_key(item):
    return item["task_id"], item["reference_system"]


def add_semcache_to_multi(report, semcache):
    report = deepcopy(report)
    old_variants = list(report["variants"])
    variants = [*old_variants, SEM_CACHE_VARIANT]
    queries = report["query_ids"]
    pairs = {(item["task_id"], item["reference_system"]): item for item in report["comparisons"]}
    records = {(item["variant"], item["query_id"]): item for item in report["records"]}
    available = {query: [*systems] for query, systems in report["available_references"].items()}
    for query in queries:
        if SEM_CACHE_SYSTEM not in available[query]:
            available[query].append(SEM_CACHE_SYSTEM)

    for record in list(report["records"]):
        if record["scoring_status"] != "scored":
            continue
        case = semcache[record["query_id"]]
        actual = payload_from_path(record["result_path"])
        score = score_payloads(actual, case["reference"], record["query_id"])
        item = {
            "task_id": record["task_id"], "variant": record["variant"],
            "query_id": record["query_id"], "reference_system": SEM_CACHE_SYSTEM,
            "reference_path": case["reference_path"], "reference_sha256": case["reference_sha256"],
            **score,
        }
        pairs[record["task_id"], SEM_CACHE_SYSTEM] = item

    for query in queries:
        case = semcache[query]
        task_id = f"{SEM_CACHE_VARIANT}/{query}"
        reference_items = [
            item for item in report["comparisons"]
            if item["query_id"] == query
            and item["reference_system"] in report["available_references"][query]
        ]
        # One saved reference path per system is enough; all copies are
        # identical paths/content across candidate variants.
        by_reference = {}
        for item in reference_items:
            by_reference.setdefault(item["reference_system"], item)
        new_scores = []
        for reference_system, item in by_reference.items():
            expected = payload_from_path(item["reference_path"])
            score = score_payloads(case["actual"], expected, query)
            pairs[task_id, reference_system] = {
                "task_id": task_id, "variant": SEM_CACHE_VARIANT,
                "query_id": query, "reference_system": reference_system,
                "reference_path": item["reference_path"],
                "reference_sha256": item.get("reference_sha256"),
                **score,
            }
            new_scores.append(score["accuracy"])
        # The SemCache candidate is also evaluated against its own Teacher
        # answer, just like every other candidate in the all-reference table.
        pairs[task_id, SEM_CACHE_SYSTEM] = {
            "task_id": task_id, "variant": SEM_CACHE_VARIANT,
            "query_id": query, "reference_system": SEM_CACHE_SYSTEM,
            "reference_path": case["reference_path"],
            "reference_sha256": case["reference_sha256"],
            **case["score"],
        }
        new_scores.append(case["score"]["accuracy"])
        records[SEM_CACHE_VARIANT, query] = {
            "task_id": task_id, "variant": SEM_CACHE_VARIANT,
            "system": SEM_CACHE_SYSTEM, "query_id": query,
            "status": "completed", "scoring_status": "scored",
            "aggregate": case["aggregate"], "accuracy": mean(new_scores),
            "reference_count": len(new_scores),
            "reference_systems": list(by_reference), "note": None,
            "result_path": case["actual_path"], "result_sha256": case["actual_sha256"],
            "result_rows": len(case["actual"]["rows"]),
            "database_equivalence": "external_semcache_evaluation_manifest",
            "translated_query_equivalence": "external_semcache_evaluation_manifest",
        }

    for (variant, query), record in records.items():
        if record["scoring_status"] != "scored":
            continue
        candidate_pairs = [item for (task, _), item in pairs.items() if task == record["task_id"]]
        if not candidate_pairs:
            continue
        record["accuracy"] = mean(item["accuracy"] for item in candidate_pairs)
        record["reference_count"] = len(candidate_pairs)
        record["reference_systems"] = [item["reference_system"] for item in candidate_pairs]

    report["variants"] = variants
    report["reference_systems"] = [*report["reference_systems"], SEM_CACHE_SYSTEM]
    report["available_references"] = available
    report["records"] = list(records.values())
    report["comparisons"] = list(pairs.values())
    report["semcache_evaluation_root"] = str(Path(semcache[queries[0]]["summary"]["result_artifact"]).parents[5])
    report["semcache_reference_root"] = str(Path(semcache[queries[0]]["reference_path"]).parents[4])
    common = [query for query in queries if all(records[variant, query]["scoring_status"] == "scored" for variant in variants)]
    report["common_queries"] = common
    report["summary"] = {}
    for variant in variants:
        scored = [records[variant, query] for query in queries if records[variant, query]["scoring_status"] == "scored"]
        shared = [record["accuracy"] for record in scored if record["query_id"] in common]
        report["summary"][variant] = {
            "mean_f1": mean(record["accuracy"] for record in scored) if scored else None,
            "scored_queries": len(scored), "total_queries": len(queries),
            "common_mean_f1": mean(shared) if shared else None, "common_queries": len(shared),
        }
    report["version"] = report.get("version", "") + "+semcache"
    return report


def add_semcache_to_own(report, semcache):
    report = deepcopy(report)
    records = {(item["variant"], item["query_id"]): item for item in report["records"]}
    queries = report["query_ids"]
    for query, case in semcache.items():
        score = case["score"]
        records[SEM_CACHE_VARIANT, query] = {
            "task_id": f"{SEM_CACHE_VARIANT}/{query}", "variant": SEM_CACHE_VARIANT,
            "system": SEM_CACHE_SYSTEM, "query_id": query, "status": "completed",
            "scoring_status": "scored", "aggregate": case["aggregate"],
            "accuracy": score["accuracy"], "precision": score["precision"],
            "recall": score["recall"], "matched_rows": score["matched_rows"],
            "result_rows": score["result_rows"], "reference_rows": score["reference_rows"],
            "reference_system": SEM_CACHE_SYSTEM,
            "result_path": case["actual_path"], "result_sha256": case["actual_sha256"],
            "reference_path": case["reference_path"], "reference_sha256": case["reference_sha256"],
            "note": None,
        }
    variants = [*report["variants"], SEM_CACHE_VARIANT]
    report["variants"] = variants
    report["records"] = list(records.values())
    common = [query for query in queries if all(records[variant, query]["scoring_status"] == "scored" for variant in variants)]
    report["common_queries"] = common
    report["summary"] = {}
    for variant in variants:
        scored = [records[variant, query] for query in queries if records[variant, query]["scoring_status"] == "scored"]
        shared = [record["accuracy"] for record in scored if record["query_id"] in common]
        report["summary"][variant] = {
            "mean_f1": mean(record["accuracy"] for record in scored) if scored else None,
            "scored_queries": len(scored), "total_queries": len(queries),
            "common_mean_f1": mean(shared) if shared else None, "common_queries": len(shared),
        }
    report["semcache_evaluation_root"] = str(Path(next(iter(semcache.values()))["summary"]["result_artifact"]).parents[5])
    report["semcache_reference_root"] = str(Path(next(iter(semcache.values()))["reference_path"]).parents[4])
    report["version"] = report.get("version", "") + "+semcache"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--semcache-references", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    existing_multi = read(run / "accuracy-multireference.json")
    # Restore the base report before augmentation so repeated additions are idempotent.
    if SEM_CACHE_VARIANT in existing_multi.get("variants", []):
        base_multi = calculate_multi_reference(run, Path(existing_multi["reference_root"]))
        write_multi_reference(run, base_multi)
        base_own = calculate_own_reference(run)
        write_own_reference(run, base_own)
    queries = read(run / "accuracy-multireference.json")["query_ids"]
    semcache = load_semcache(args.evaluation.resolve(), args.semcache_references.resolve(), queries)
    multi = add_semcache_to_multi(read(run / "accuracy-multireference.json"), semcache)
    own = add_semcache_to_own(read(run / "accuracy-own-reference.json"), semcache)
    multi["semcache_evaluation_root"] = str(args.evaluation.resolve())
    multi["semcache_reference_root"] = str(args.semcache_references.resolve())
    own["semcache_evaluation_root"] = str(args.evaluation.resolve())
    own["semcache_reference_root"] = str(args.semcache_references.resolve())
    write_multi_reference(run, multi)
    write_own_reference(run, own)
    save(run / "semcache-accuracy-augmentation.json", {
        "variant": SEM_CACHE_VARIANT, "system": SEM_CACHE_SYSTEM,
        "evaluation_root": str(args.evaluation.resolve()),
        "reference_root": str(args.semcache_references.resolve()),
        "queries": queries, "query_count": len(queries),
        "multi_reference_summary": multi["summary"], "own_reference_summary": own["summary"],
    })
    print(json.dumps({"multi_reference": multi["summary"], "own_reference": own["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
