"""Use the system-comparison row-F1 implementation for all measured feedback."""

import csv
import json
import time
from pathlib import Path

from experiments.core.io import digest
from experiments.system_comparison.accuracy.artifacts import typed_result, unique_schema
from experiments.system_comparison.accuracy.matching import (
    VERSION,
    finish_match,
    prepare_match,
    score_non_free_text,
)
from experiments.system_comparison.accuracy.teacher import judge_pairs
from planning.state import digest as content_digest

from training.artifacts import save
from training.collection.rewards import execution_reward

RESULT_ACCURACY_METRIC = VERSION
PRE_AGGREGATE_ACCURACY_METRIC = VERSION
OUTPUT_COMPARISON_REVISION = "typed-generated-columns-v1"
NON_FREE_TEXT_METRIC = "non-free-text-row-f1-v1"


def candidate_result(artifact, schema, *, aggregate=False, persist=True):
    folder = Path(artifact)
    if aggregate:
        path = folder / "aggregate-input.jsonl"
        with path.open() as stream:
            columns = json.loads(next(stream))["columns"]
            rows = [row for line in stream if line.strip() for row in json.loads(line)]
        result = typed_result({"columns": columns, "rows": rows}, schema)
        if persist:
            save(folder / "pre-aggregate-result.json", result)
        return result
    null = json.loads((folder / "result-contract.json").read_text())["null_value"]
    with (folder / "result.csv").open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is not None:
            if len(header) != len(schema):
                raise ValueError("Candidate result column count differs from reference")
            # DuckDB CLI preserves duplicate SELECT * labels. Reference JSON
            # applies positional suffixes; normalize with that same contract.
            columns = unique_schema(
                [
                    (name, declared)
                    for name, (_, declared) in zip(header, schema, strict=True)
                ]
            )
            if columns != unique_schema(schema):
                raise ValueError("Candidate result column names differ from reference")
        rows = []
        for row in reader:
            if not row and len(schema) == 1:
                row = [""]
            if len(row) != len(schema):
                raise ValueError("Candidate result row width differs from reference")
            rows.append([None if value == null else value for value in row])
    return typed_result({"columns": schema, "rows": rows}, schema)


def score_results(
    actual, expected, query_id, scope, *, output, teacher_config, text_fields=None
):
    case, jobs = prepare_match(
        actual, expected, query_id, scope, text_fields=text_fields
    )
    judgments, audit = (
        judge_pairs(jobs, teacher_config, Path(output), workers=8, batch_size=8)
        if jobs
        else ({}, {})
    )
    return {
        **finish_match(case, judgments),
        "teacher_pairs": len(jobs),
        "teacher_audit": audit,
    }


def finalize_measured_accuracy(
    entries,
    *,
    reference_answers,
    teacher_config,
    wait=False,
    update=None,
    progress=print,
):
    system = getattr(reference_answers, "system", "sema")
    pending = list(entries)
    while pending:
        remaining = []
        for workload, destination, payload in pending:
            destination = Path(destination)
            if len(payload["records"]) != 1:
                raise ValueError("Measured feedback requires one rollout per query")
            record = payload["records"][0]
            metadata = record["metadata"]
            metric = workload.get("accuracy_metric", VERSION)
            if metric not in (VERSION, NON_FREE_TEXT_METRIC):
                raise ValueError("Unsupported measured accuracy metric")
            previous = metadata.get("quality", {})
            if previous.get("metric") == metric and (
                previous.get("status") == "completed"
                and previous.get("output_comparison_revision")
                == OUTPUT_COMPARISON_REVISION
                or record["status"] == "timeout"
                and previous.get("status") == "execution_timeout"
                and previous.get("penalty_policy") == "timeout_accuracy_zero_v1"
                and record.get("accuracy") == 0.0
            ):
                continue
            started = time.perf_counter()
            quality = {
                "status": "execution_timeout"
                if record["status"] == "timeout"
                else "candidate_unavailable",
                "method": f"frozen_{system}_teacher_answer",
                "metric": metric,
            }
            if record["status"] == "timeout":
                record["accuracy"] = 0.0
                quality.update(
                    value=0.0, accuracy=0.0, penalty_policy="timeout_accuracy_zero_v1",
                    accuracy_is_penalty=True,
                    basis="timeout penalty; not a measured result F1",
                )
            if record["status"] == "completed":
                try:
                    reference = reference_answers.load(workload)
                except FileNotFoundError:
                    quality["status"] = "reference_pending"
                    record["accuracy"] = None
                    remaining.append((workload, destination, payload))
                else:
                    aggregate = reference["basis"] == "pre_aggregate_rows"
                    expected = (
                        reference["pre_aggregate"] if aggregate else reference["result"]
                    )
                    if expected is None:
                        raise ValueError("Reference lacks required aggregate input")
                    artifact = metadata["timing_artifacts"][0]
                    actual = candidate_result(
                        artifact, expected["columns"], aggregate=aggregate
                    )
                    generated = workload.get("output_contract", {}).get(
                        "generated_columns", {}
                    )
                    text_columns = {
                        name
                        for name, declared in expected["columns"]
                        if declared.upper() in {"VARCHAR", "TEXT"}
                    }
                    fields = [
                        name
                        for name, description in generated.items()
                        if name in text_columns
                        and not description.get("values")
                        and description.get("comparison") != "exact"
                    ]
                    scope = {
                        "reference_sha256": reference["answer_sha256"],
                        "teacher_config_sha256": digest(teacher_config),
                        "protocol": metric,
                    }
                    if metric == NON_FREE_TEXT_METRIC:
                        metrics = score_non_free_text(
                            actual,
                            expected,
                            workload["public_query_id"],
                        )
                    else:
                        metrics = score_results(
                            actual,
                            expected,
                            workload["id"],
                            scope,
                            output=destination / "accuracy-judgments",
                            teacher_config=teacher_config,
                            text_fields=fields,
                        )
                    if metrics["scoring_status"] != "scored":
                        raise RuntimeError(
                            f"Unresolved teacher judgments for {workload['id']}; accuracy remains unknown"
                        )
                    record["accuracy"] = metrics["accuracy"]
                    record["raw_accuracy"] = metrics["accuracy"]
                    quality.update(
                        **metrics,
                        value=metrics["accuracy"],
                        status="completed",
                        output_comparison_revision=OUTPUT_COMPARISON_REVISION,
                        artifact=artifact,
                        reference_answer=reference["answer_path"],
                        reference_artifact=reference["artifact"],
                        reference_sha256=reference["answer_sha256"],
                        result_sha256=content_digest(actual),
                        accuracy_basis=reference["basis"],
                    )
            quality["comparison_seconds"] = time.perf_counter() - started
            metadata.update(quality=quality, accuracy_status=quality["status"])
            record["reward"] = execution_reward(record)
            save(destination / "reference.json", quality)
            save(destination / "qualities.json", {record["plan_key"]: quality})
            save(destination / "reward.json", record["reward"])
            save(destination / "experience.json", payload)
            progress(
                f"{workload['id']}: row F1 {quality['status']} value={record['accuracy']}; "
                f"reward={record['reward']['vector']}",
                flush=True,
            )
        pending = remaining
        if not wait or not pending:
            return
        generation = reference_answers.check_generator()
        if generation["status"] == "completed":
            raise RuntimeError(
                "Reference generation ended with missing training answers"
            )
        if update:
            update(
                phase="waiting_reference_answers",
                pending_reference_answers=len(pending),
                reference_generation_completed=generation.get("completed"),
                reference_generation_total=generation.get("total"),
                reference_generation_status=generation["status"],
                pending_reference_queries=[
                    workload["id"] for workload, _, _ in pending
                ],
            )
        time.sleep(20)


def score_execution_pair(actual, reference, query_id, *, output, teacher_config):
    """Heldout agreement excludes generated prose and never requests judgments."""
    folder = Path(reference["artifact"])
    pre = reference.get("pre_aggregate_path")
    aggregate = pre is not None or (folder / "aggregate-input.jsonl").exists()
    if pre is not None:
        expected = typed_result(json.loads(Path(pre).read_text()))
        schema = expected["columns"]
    elif aggregate:
        with (folder / "aggregate-input.jsonl").open() as stream:
            schema = json.loads(next(stream))["columns"]
    else:
        nodes = reference["query_ir"]["operators"]
        children = {c for node in nodes for c in node["children"]}
        root = next(node for node in nodes if node["node_id"] not in children)
        with (folder / "result.csv").open(newline="") as stream:
            names = next(csv.reader(stream), [])
        schema = [
            [name, col["type"]]
            for name, col in zip(names, root["output_schema"], strict=True)
        ]
    if pre is None:
        expected = candidate_result(folder, schema, aggregate=aggregate, persist=False)
    observed = candidate_result(actual["artifact"], schema, aggregate=aggregate)
    metrics = score_non_free_text(observed, expected, query_id)
    save(Path(output) / "non-free-text.json", metrics)
    return metrics["accuracy"]
