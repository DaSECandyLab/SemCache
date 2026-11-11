"""Validate typed reference/campaign artifacts without replaying a query."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from experiments.core.io import digest
from experiments.system_comparison.accuracy.aggregates import aggregate_deviation
from experiments.system_comparison.reporting.results import normalize


def read(path):
    return json.loads(Path(path).read_text())


def sql_digest(sql):
    return hashlib.sha256(sql.encode()).hexdigest()


def result_digest(result):
    text = json.dumps(
        {k: result[k] for k in ("columns", "rows")},
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(text.encode()).hexdigest()


def unique_schema(columns):
    """Preserve duplicate SQL columns by position, using DuckDB-style suffixes."""
    reserved = {name for name, _ in columns}
    seen, used, output = Counter(), set(), []
    for name, kind in columns:
        candidate = name
        if name in used:
            suffix = seen[name]
            while f"{name}_{suffix}" in used or f"{name}_{suffix}" in reserved:
                suffix += 1
            candidate = f"{name}_{suffix}"
        seen[name] += 1
        used.add(candidate)
        output.append([candidate, kind])
    return output


def typed_result(payload, schema=None):
    actual = unique_schema(payload["columns"])
    if schema is not None and actual != unique_schema(schema):
        raise ValueError(f"Saved output schema mismatch: {actual} != {schema}")
    return normalize({"columns": actual, "rows": payload["rows"]})


def load_references(root):
    root = Path(root).resolve()
    contract = read(root / "contract.json")
    workloads = read(root / "workloads.json")["workloads"]
    references, evidence = {}, {}
    for workload in workloads:
        query = workload["metadata"]["id"]
        answer_path = root / workload["id"] / "answer.json"
        answer = read(answer_path)
        if answer.get("status") != "completed":
            raise ValueError(f"Incomplete reference answer: {query}")
        if not (
            answer["sql_sha256"]
            == workload["sql_sha256"]
            == sql_digest(workload["sql"])
            == contract["queries"][workload["id"]]
        ):
            raise ValueError(f"Reference SQL integrity failure: {query}")
        if answer["result_sha256"] != result_digest(answer):
            raise ValueError(f"Reference result integrity failure: {query}")
        result = typed_result(answer)
        pre = answer.get("pre_aggregate")
        pre_result = None
        if pre:
            payload = read(pre["path"])
            if result_digest(payload) != pre["result_sha256"]:
                raise ValueError(f"Pre-aggregate reference integrity failure: {query}")
            pre_result = typed_result(payload, pre["columns"])
            if len(pre_result["rows"]) != pre["row_count"]:
                raise ValueError(f"Pre-aggregate row count mismatch: {query}")
        references[query] = {
            "result": result,
            "pre_aggregate": pre_result,
            "workload": workload,
            "answer": answer,
        }
        evidence[query] = {
            "answer_path": str(answer_path),
            "answer_sha256": digest(answer_path),
            "sql_sha256": answer["sql_sha256"],
            "result_sha256": answer["result_sha256"],
            "rows": len(result["rows"]),
            "columns": result["columns"],
            "teacher": answer["teacher"],
            "aggregate": workload["metadata"].get("aggregate"),
            "aggregate_reference_value": (
                result["rows"][0][0]
                if workload["metadata"].get("aggregate")
                and len(result["rows"]) == len(result["columns"]) == 1
                else None
            ),
            "pre_aggregate_path": pre["path"] if pre else None,
            "pre_aggregate_rows": len(pre_result["rows"]) if pre_result else None,
            "pre_aggregate_sha256": digest(pre["path"]) if pre else None,
        }
    return references, evidence, contract


def saved_pre_aggregate(folder, summary):
    pre = summary.get("pre_aggregate") or {}
    candidates = [
        folder / name for name in ("pre-aggregate-result.json", "aggregate-input.json")
    ]
    if pre.get("path"):
        path = Path(pre["path"])
        candidates.insert(0, path if path.is_absolute() else folder / path)
    return next((path for path in candidates if path.is_file()), None)


def timing_fields(summary):
    """Keep measured time separate from repeat means and explicit user adjustments."""
    average = summary.get("average_seconds")
    use_average = (
        type(summary.get("repeat_runs")) is int
        and summary["repeat_runs"] == 2
        and summary.get("status") == "completed"
        and type(average) in (int, float)
        and math.isfinite(average)
        and average >= 0
    )
    fields = {
        "seconds": summary.get("seconds"),
        "display_seconds": average if use_average else summary.get("seconds"),
        "time_basis": "two_run_mean" if use_average else "single_run",
    }
    adjustment = summary.get("timing_adjustment") or {}
    adjusted = adjustment.get("adjusted_query_seconds")
    excluded = adjustment.get("excluded_seconds")
    raw = summary.get("seconds")
    if (
        adjustment.get("method") == "exclusive_slow_request_wait_from_gateway_v1"
        and summary.get("status") == "completed"
        and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in (adjusted, excluded, raw))
        and math.isclose(adjusted + excluded, raw, rel_tol=1e-9, abs_tol=1e-6)
    ):
        fields.update(display_seconds=adjusted, time_basis="adjusted_exclusive_wait_estimate", timing_adjustment=adjustment)
    for name in (
        "repeat_runs",
        "initial_seconds",
        "second_seconds",
        "average_seconds",
        "repeat_manifest",
        "repeat_results",
    ):
        if name in summary:
            fields[name] = summary[name]
    return fields


def executed_query_matches(job, run):
    """Require the job's executed hash to match its current frozen program."""
    query_file, expected = job.get("query_file"), job.get("query_sha256")
    if (
        not isinstance(query_file, str)
        or not query_file
        or not isinstance(expected, str)
    ):
        return False
    path = Path(query_file)
    if not path.is_absolute():
        path = run / path
    try:
        return path.is_file() and digest(path) == expected
    except (OSError, ValueError):
        return False


def load_campaign(
    run, references, reference_evidence, contract, *, aggregate_metric="input_rows"
):
    run = Path(run).resolve()
    campaign = read(run / "campaign.json")
    versions = {}
    for query in campaign["query_ids"]:
        # Exact equality of the frozen Sema SQL proves the common workload
        # matches the Sema reference. Other systems retain independent programs.
        frozen = run / "queries" / "sema" / (query + ".sql")
        versions[query] = digest(frozen) == references[query]["workload"]["sql_sha256"]
    records, results = [], {}
    for task in campaign["tasks"]:
        query = task["query_id"]
        folder = run / task["folder"]
        summary_path, job_path = folder / "summary.json", folder / "job.json"
        summary, job = read(summary_path), read(job_path)
        version_matches = versions[query] and executed_query_matches(job, run)
        dataset = references[query]["workload"]["dataset"]
        input_matches = (
            job.get("input_databases", {}).get("sema", {}).get("sha256")
            == contract["database_sha256"][dataset]
        )
        aggregate = reference_evidence[query]["aggregate"]
        path = folder / "result.json"
        record = {
            "task_id": task["task_id"],
            "variant": task["variant"],
            "system": task["system"],
            "query_id": query,
            "status": summary.get("status", "missing"),
            **timing_fields(summary),
            "accuracy": None,
            "precision": None,
            "recall": None,
            "matched_rows": None,
            "aggregate": aggregate,
            "accuracy_basis": (
                "aggregate_relative_deviation"
                if aggregate and aggregate_metric == "relative_deviation"
                else "pre_aggregate_rows"
                if aggregate
                else "final_output"
            ),
            "scoring_status": "result_unavailable",
            "query_version_matches": version_matches,
            "input_matches": input_matches,
            "result_path": str(path) if path.exists() else None,
            "summary_path": str(summary_path),
            "summary_sha256": digest(summary_path),
            "job_path": str(job_path),
            "job_sha256": digest(job_path),
            "reference_path": reference_evidence[query]["answer_path"],
            "reference_sha256": reference_evidence[query]["answer_sha256"],
            "mechanism_profile": summary.get("native_mechanism_profile"),
            "cache_policy": summary.get("cache_policy"),
        }
        if path.exists():
            record["result_sha256"] = digest(path)
        if not version_matches:
            record.update(
                scoring_status="query_version_mismatch",
                note="查询执行版本与参考不一致",
            )
        elif not input_matches:
            record.update(
                scoring_status="input_mismatch", note="系统与参考输入版本不同"
            )
        elif record["status"] != "completed" or not path.exists():
            record["note"] = summary.get("reason") or record["status"]
        elif aggregate and aggregate_metric == "relative_deviation":
            try:
                actual = typed_result(
                    read(path), references[query]["result"]["columns"]
                )
                record.update(aggregate_deviation(actual, references[query]["result"]))
            except (ValueError, KeyError, TypeError) as error:
                record.update(scoring_status="invalid_output", note=str(error))
        elif aggregate:
            pre_path = saved_pre_aggregate(folder, summary)
            if pre_path is None:
                record.update(
                    scoring_status="aggregate_input_missing",
                    note="系统未保存聚合前输入",
                )
            elif references[query]["pre_aggregate"] is None:
                record.update(
                    scoring_status="reference_input_missing",
                    note="参考未保存聚合前输入",
                )
            else:
                try:
                    actual = typed_result(
                        read(pre_path), references[query]["pre_aggregate"]["columns"]
                    )
                    record.update(
                        pre_aggregate_path=str(pre_path),
                        pre_aggregate_sha256=digest(pre_path),
                    )
                    results[task["task_id"]] = (
                        actual,
                        references[query]["pre_aggregate"],
                    )
                except (ValueError, KeyError, TypeError) as error:
                    record.update(scoring_status="invalid_output", note=str(error))
        else:
            try:
                actual = typed_result(
                    read(path), references[query]["result"]["columns"]
                )
                results[task["task_id"]] = (actual, references[query]["result"])
            except (ValueError, KeyError, TypeError) as error:
                record.update(scoring_status="invalid_output", note=str(error))
        records.append(record)
    return campaign, records, results
