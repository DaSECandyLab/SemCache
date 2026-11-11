#!/usr/bin/env python3
"""Execute user SQL in DuckDB with the native QueryIR/PlanSpec extension bridge."""

import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from planning.settings.configuration import (
    load_config,
    response_cache_settings,
    runtime_path,
)
from planning.settings.parallelism import parallelism_argument
from runtime_audit import effective_actions
from semantic_events import logical_rows

ROOT = Path(__file__).resolve().parents[1]


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def verify_row(row, spec, action, logical):
    if row["query_id"] != spec["query_id"] or row["parse_status"] != "validated":
        raise ValueError("Row identity/parse status invalid")
    operator = logical[row["operator_id"]]
    inputs = (
        operator["input_fields"]
        or operator["left_input_fields"] + operator["right_input_fields"]
    )
    layout = action["layout"]
    fields = layout.get(
        "field_order",
        layout.get("left_field_order", []) + layout.get("right_field_order", []),
    )
    # display_to_input proves that dense display indices still reference the
    # original bound expressions after an optimizer field permutation.
    mapping = [inputs.index(name) for name in fields]
    if row["prompt_serialization_version"] != 4 or row["display_to_input"] != mapping:
        raise ValueError("Prompt display mapping differs from bound logical inputs")
    indices = [
        int(i) for i in re.findall(r"^\{(\d+)\}:", row["user_text"], re.MULTILINE)
    ]
    if indices != list(range(len(inputs))):
        raise ValueError("Data Block display placeholders must increase from zero")
    task = re.sub(
        r"\{(\d+)\}",
        lambda match: "{" + str(mapping.index(int(match[1]))) + "}",
        operator["task"],
    )
    if "<TASK_DESCRIPTION>\n" + task + "\n</TASK_DESCRIPTION>" not in row["user_text"]:
        raise ValueError(
            "Rendered task placeholders no longer refer to the corresponding input values"
        )
    if (
        row["model_id"] != action["model_id"]
        or row["block_order"] != action["layout"]["block_order"]
    ):
        raise ValueError("Observed model/layout differs from PlanSpec")
    for key, value in action["layout"].items():
        if key != "block_order" and row[key] != value:
            raise ValueError("Observed field order differs from PlanSpec")


def verify_pair(row, previous):
    """Audit left-major identities while the first row has no final N_R yet."""
    if row["operator_kind"] != "Semantic Join" or row.get("pair_input"):
        return
    left, right, pair, size = (
        row.get(key) for key in ("left_index", "right_index", "pair_id", "right_count")
    )
    if any(type(value) is not int or value < 0 for value in (left, right, pair)):
        raise ValueError("Invalid semantic pair identity")
    old_pair, old_size = previous.get(row["operator_id"], (None, None))
    if size is None:
        valid = left == 0 and pair == right and old_size is None
    else:
        valid = type(size) is int and size > right and pair == left * size + right
        valid = valid and (old_size is None or size == old_size)
    if not valid or old_pair is not None and (left, right) <= old_pair:
        raise ValueError(
            "Semantic pair identities violate left-major order or right cardinality"
        )
    previous[row["operator_id"]] = ((left, right), size)


def verify(events):
    """Cross-check planning, dispatch, prompt bindings and terminal accounting."""

    decisions = [event for event in events if event["event"] == "planning_decision"]
    reports = [event for event in events if event["event"] == "execution_report"]
    if len(decisions) != 1 or len(reports) != 1:
        raise ValueError(
            "Expected exactly one semantic query decision and execution report"
        )
    report = reports[0]
    spec = decisions[0]["plan_spec"]
    pipeline = decisions[0]["query_ir"].get("execution_mode") == "data_chunk_pipeline"
    if pipeline != (report.get("execution_mode") == "data_chunk_pipeline"):
        raise ValueError("Execution mode differs from QueryIR")
    if decisions[0]["query_ir"].get("semantic_join_mode") != report.get(
        "semantic_join_mode"
    ):
        raise ValueError("Semantic Join execution mode differs from QueryIR")
    response_order = (
        decisions[0]["query_ir"]
        .get("environment", {})
        .get("semantic_response_order", "input_order")
    )
    if report.get("semantic_response_order", "input_order") != response_order:
        raise ValueError("Semantic response order differs from QueryIR")
    ready_first = response_order == "ready_first"
    if report["status"] != "success" or report["plan_spec"] != spec:
        raise ValueError("Execution report does not match the successful PlanSpec")
    actions = {action["operator_id"]: action for action in spec["actions"]}
    runtime_actions = effective_actions(events, decisions[0], report)
    if len(actions) != len(spec["actions"]):
        raise ValueError("Duplicate PlanSpec action in audit")
    rank = {operator_id: i for i, operator_id in enumerate(actions)}
    starts = [
        event["operator_id"] for event in events if event["event"] == "operator_start"
    ]
    if (
        starts != report["actual_start_order"]
        or len(starts) != len(set(starts))
        or any(operator_id not in actions for operator_id in starts)
        or not pipeline
        and [rank[operator_id] for operator_id in starts]
        != sorted(rank[operator_id] for operator_id in starts)
    ):
        raise ValueError("Operator starts differ from PlanSpec/report")
    if [summary["operator_id"] for summary in report["operators"]] != list(actions):
        raise ValueError("Execution report is missing/duplicating semantic stages")
    rows = [event for event in events if event["event"] == "semantic_row"]
    discarded = [
        event for event in events if event["event"] == "semantic_request_discarded"
    ]
    logical = {
        op["operator_id"]: op for op in decisions[0]["query_ir"]["semantic_operators"]
    }
    physical_requests = rows + discarded
    request_sequences = sorted(row["request_seq"] for row in physical_requests)
    if any(
        type(sequence) is not int or sequence < 1 for sequence in request_sequences
    ) or len(set(request_sequences)) != len(request_sequences):
        raise ValueError("Invalid or duplicate semantic request sequence")
    if any(row["operator_id"] not in starts for row in physical_requests):
        raise ValueError("Request from an operator that has not started")
    if any(
        event.get("query_id") != spec["query_id"]
        or event.get("reason") != "query_terminated_before_consumption"
        or not isinstance(event.get("model_id"), str)
        or not event["model_id"]
        for event in discarded
    ):
        raise ValueError("Discarded request identity/reason invalid")
    if not pipeline and [rank[row["operator_id"]] for row in rows] != sorted(
        rank[row["operator_id"]] for row in rows
    ):
        raise ValueError("Semantic requests do not follow PlanSpec stage order")
    processed = list(logical_rows(events))
    pairs, grouped = {}, defaultdict(list)
    for row in processed:
        grouped[row["operator_id"]].append(row)
        verify_row(
            row,
            spec,
            runtime_actions.get(
                (row["operator_id"], row["input_index"]), actions[row["operator_id"]]
            ),
            logical,
        )
    # Validate pair identities in admission order, independently of when their
    # responses arrived. Cached answers keep their original logical identity.
    for row in (
        sorted(processed, key=lambda row: (row["operator_id"], row["input_index"]))
        if ready_first
        else processed
    ):
        verify_pair(row, pairs)
    for summary in report["operators"]:
        selected = grouped[summary["operator_id"]]
        discarded_count = sum(
            event["operator_id"] == summary["operator_id"] for event in discarded
        )
        indices = [row["input_index"] for row in selected]
        if ready_first:
            # LIMIT may stop after a later, faster row, leaving earlier inputs
            # unconsumed. Require unique identities without inventing a prefix.
            if any(type(index) is not int or index < 0 for index in indices) or len(
                set(indices)
            ) != len(indices):
                raise ValueError("Invalid or duplicate per-operator input identity")
        elif indices != list(range(len(selected))):
            # A downstream reuser can publish a source response before its
            # owner drains that row. Physical output still follows input order;
            # only explicitly marked dependency publication may reorder logs.
            if not any(
                row.get("result_publication") == "shared_dependency" for row in selected
            ) or sorted(indices) != list(range(len(selected))):
                raise ValueError("Noncontiguous per-operator input sequence")
        if (
            summary["llm_calls"]
            != sum(row["event"] == "semantic_row" for row in selected) + discarded_count
            or summary.get("discarded_requests", 0) != discarded_count
            or summary.get("result_reuses", 0)
            != sum(
                row["event"] in ("semantic_result_reuse", "semantic_result_cache_hit")
                for row in selected
            )
            or summary["input_rows_or_pairs"] != len(selected)
            or summary["output_rows"] != sum(row["emitted"] for row in selected)
            or summary["started"] != (summary["operator_id"] in starts)
        ):
            raise ValueError("Execution report cardinalities differ from row telemetry")
    for event_type in ("submit", "complete"):
        audit = sorted(
            (event for event in events if event["event"] == event_type),
            key=lambda event: event["request_seq"],
        )
        if [event["request_seq"] for event in audit] != request_sequences:
            raise ValueError(f"Incomplete {event_type} audit")
        for event, row in zip(
            audit, sorted(physical_requests, key=lambda row: row["request_seq"])
        ):
            if (
                any(
                    event[key] != row[key]
                    for key in ("query_id", "operator_id", "model_id")
                )
                or event["retry_count"] != 0
            ):
                raise ValueError("Scheduler audit identity/retry mismatch")
    return decisions[0], report


def main():
    settings = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument(
        "--db", type=Path, required=True, help="Existing database, opened read-only"
    )
    parser.add_argument(
        "--input-mode",
        choices=("memory", "disk"),
        default="memory",
        help="Preload input into native memory tables before query timing (default: memory)",
    )
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--optimizer-socket", type=Path, required=True)
    parser.add_argument(
        "--runtime", type=Path, help="Optional runtime admission/policy JSON"
    )
    parser.add_argument(
        "--history",
        type=Path,
        help="Persist actual static-plan prefix history for future planning",
    )
    parser.add_argument("--optimizer-timeout-ms", type=int, default=120000)
    parser.add_argument(
        "--parallelism",
        type=parallelism_argument,
        default=settings["execution"]["cpu_parallelism"],
    )
    parser.add_argument(
        "--max-concurrent-requests-per-model",
        type=parallelism_argument,
        default=settings["execution"]["max_concurrent_requests_per_model"],
        help="Maximum in-flight normal execution requests for one physical model.",
    )
    parser.add_argument(
        "--enforce-model-operator-order",
        action=argparse.BooleanOptionalAction,
        default=settings["execution"]["enforce_model_operator_order"],
        help="Prioritize earlier operators in each model's pending request queue.",
    )
    parser.add_argument(
        "--build-dir", type=Path, default=runtime_path(load_config(), "build")
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New artifact directory"
    )
    args = parser.parse_args()
    if args.history and args.runtime:
        parser.error(
            "--history currently requires static execution; mixed runtime generations need separate profiles"
        )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    build = args.build_dir.resolve()
    sql = args.query.read_text(encoding="utf-8")
    (output / "query.sql").write_text(sql, encoding="utf-8")
    settings = "\n".join(
        [
            f"LOAD {quote(build / 'extension/semantic/semantic.duckdb_extension')};",
            f"SET threads={args.parallelism};",
            f"SET semantic_parallelism={args.parallelism};",
            f"SET semantic_max_concurrent_requests_per_model={args.max_concurrent_requests_per_model};",
            "SET semantic_enforce_model_operator_order="
            f"{str(args.enforce_model_operator_order).lower()};",
            "SET preserve_insertion_order=true;",
            f"SET semantic_models={quote(args.models.read_text())};",
            f"SET semantic_response_cache={quote(json.dumps(response_cache_settings(settings)))};",
            f"SET semantic_optimizer_socket={quote(args.optimizer_socket.resolve())};",
            f"SET semantic_optimizer_timeout_ms={args.optimizer_timeout_ms};",
            f"SET semantic_telemetry={quote(output / 'telemetry.jsonl')};",
            f"SET semantic_runtime={quote(args.runtime.read_text() if args.runtime else '')};",
        ]
    )
    profiling = (
        "SET enable_profiling='json';\n"
        + f"SET profiling_output={quote(output / 'profile.json')};"
    )
    # Write run.json even on failure so an artifact directory is never mistaken
    # for a successful but empty execution.
    run = {
        "status": "error",
        "user_sql_executed": True,
        "relational_optimizer": "DuckDB",
        "scheduler_concurrency": None,
        "scheduler_policy": "ordered_model_lanes"
        if args.enforce_model_operator_order
        else "fifo_model_lanes",
        "scheduler_enforce_model_operator_order": args.enforce_model_operator_order,
        "scheduler_max_inflight_per_model": args.max_concurrent_requests_per_model,
        "observed_http_concurrency": None,
        "parallelism": args.parallelism,
    }
    try:
        if args.input_mode == "memory":
            # Use the same process/phase boundary as the comparison workers.
            # Loading source data never invokes the semantic optimizer.
            sys.path.insert(0, str(ROOT.parent))
            from types import SimpleNamespace

            from experiments.system_comparison.data.memory import preload_sql
            from experiments.system_comparison.execution.sql import Session

            folder = output / "native-session"
            folder.mkdir()
            runtime_started = time.monotonic()
            session = Session(build / "duckdb", folder, None, unsigned=True)
            try:
                configured = session.execute(settings, 600)
                if configured["status"] != "completed":
                    raise ValueError("Native runtime setup failed")
                setup_seconds = time.monotonic() - runtime_started
                load_started = time.monotonic()
                input_sql = preload_sql(args.db)
                (output / "input-preload.sql").write_text(input_sql + "\n")
                loaded = session.execute(input_sql, 600)
                if loaded["status"] != "completed":
                    raise ValueError("Native memory preload failed")
                load_seconds = time.monotonic() - load_started
                executed = session.execute(
                    profiling
                    + "\n.mode csv\n.headers on\n.once "
                    + quote(output / "result.csv")
                    + "\n"
                    + sql
                    + "\n;",
                    600,
                )
                run.update(
                    input_mode="memory",
                    source_database=str(args.db.resolve()),
                    source_detached=True,
                    runtime_setup_seconds=setup_seconds,
                    load_seconds=load_seconds,
                    query_seconds=executed["seconds"],
                )
            finally:
                session.close(kill=True)
            result = SimpleNamespace(
                returncode=0 if executed["status"] == "completed" else 1,
                stdout=(output / "result.csv").read_text()
                if (output / "result.csv").exists()
                else "",
                stderr=(folder / "stderr.log").read_text(),
            )
        else:
            result = subprocess.run(
                [
                    str(build / "duckdb"),
                    "-no-init",
                    "-unsigned",
                    "-readonly",
                    "-batch",
                    "-bail",
                    "-csv",
                    "-header",
                    str(args.db.resolve()),
                ],
                input=settings + "\n" + profiling + "\n" + sql + "\n;",
                text=True,
                capture_output=True,
                timeout=600,
                check=False,
            )
        (output / "stderr.log").write_text(result.stderr)
        if result.returncode:
            raise ValueError(result.stderr or result.stdout)
        events = [
            json.loads(line)
            for line in (output / "telemetry.jsonl").read_text().splitlines()
        ]
        decision, report = verify(events)
        run["observed_http_concurrency"] = max(
            (e["concurrency"] for e in events if e["event"] == "submit"), default=0
        )
        from planning.observations.metrics import cache_metrics

        (output / "cache_metrics.json").write_text(
            json.dumps(cache_metrics(events), indent=2)
        )
        if args.history:
            from planning.observations.cache import CacheHistory
            from planning.observations.statistics import from_execution, ingest_history
            from planning.state import SearchProblem

            history = CacheHistory(args.history)
            try:
                problem = SearchProblem(decision["query_ir"])
                inputs = from_execution(problem, events)
                ingest_history(events, history)
                profile = {
                    "query_key": problem.key,
                    "execution_observations": inputs,
                    "history": history.snapshot(),
                }
                (output / "profiles.json").write_text(
                    json.dumps({problem.key: profile}, indent=2, ensure_ascii=False)
                )
            finally:
                history.close()
        for name, value in [
            ("query_ir.json", decision["query_ir"]),
            ("plan_spec.json", decision["plan_spec"]),
            ("execution_report.json", report),
        ]:
            (output / name).write_text(
                json.dumps(value, indent=2, ensure_ascii=False) + "\n"
            )
        run["status"] = "passed"
        (output / "result.csv").write_text(result.stdout)
        sys.stdout.write(result.stdout)
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        subprocess.TimeoutExpired,
    ) as error:
        run["error"] = str(error)
        print(error, file=sys.stderr)
        return 1
    finally:
        (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
