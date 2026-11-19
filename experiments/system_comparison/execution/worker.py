"""Execute one frozen v4 query and retain native errors, stats and final rows."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

from experiments.core.io import digest, save
from experiments.core.paths import ROOT, experiment_sources
from experiments.core.resources import BATCH_SIZE, CONCURRENCY, resource_snapshot
from experiments.system_comparison.reporting.results import normalize

from .worker_sql import (
    abacus_config as abacus_config,
)
from .worker_sql import (
    configure_abacus as configure_abacus,
)
from .worker_sql import (
    configure_lotus as configure_lotus,
)
from .worker_sql import (
    configure_nirvana as configure_nirvana,
)
from .worker_sql import (
    load_module as load_module,
)
from .worker_sql import (
    load_tables as load_tables,
)
from .worker_sql import (
    sql_query as sql_query,
)
from .worker_sql import (
    stopped_system as stopped_system,
)


def python_query(job):
    runtime_started = time.monotonic()
    import pandas as pd

    module = load_module("frozen_v4_query", Path(job["query_file"]))
    model = None
    llm_stats = None
    if job["system"] == "abacus":
        configure_abacus(job)
    elif job["system"] == "nirvana":
        llm_stats = configure_nirvana(job)
    else:
        model = configure_lotus(job)
    runtime_setup_seconds = time.monotonic() - runtime_started
    load_started = time.monotonic()
    tables = load_tables(job)
    load_seconds = time.monotonic() - load_started
    save(
        Path(job["output"]) / "input-loading.json",
        {
            "policy": "preload_before_query",
            "storage": "materialized_dataframes",
            "source_database": job["input_database"],
            "source_connection_closed": True,
            "tables": {
                name: {
                    "rows": len(frame),
                    "bytes": int(frame.memory_usage(index=True, deep=True).sum()),
                }
                for name, frame in tables.items()
            },
            "load_seconds": load_seconds,
            "runtime_setup_seconds": runtime_setup_seconds,
        },
    )
    from contextlib import nullcontext
    from unittest.mock import patch

    from experiments.system_comparison.reporting.timing import QueryTimer
    from experiments.validation import faithful_runtime

    captured = []
    passthrough_aggregate = bool(
        job.get("skip_final_aggregate")
        and job["system"] in {"lotus", "abacus"}
    )
    original_aggregate = faithful_runtime.aggregate_frame

    def observe_aggregate(frame, spec):
        if job["system"] == "lotus" and job.get("pre_aggregate_schema"):
            from experiments.system_comparison.backends.lotus.compat import (
                normalize_numeric_columns,
            )

            frame, conversion = normalize_numeric_columns(frame, job["pre_aggregate_schema"])
            save(Path(job["output"]) / "numeric-output-conversion.json", conversion)
        captured.append(frame)
        return frame if passthrough_aggregate else original_aggregate(frame, spec)

    observation = (
        patch.object(faithful_runtime, "aggregate_frame", observe_aggregate)
        if job.get("capture_aggregate_inputs") or passthrough_aggregate
        else nullcontext()
    )
    with (
        observation,
        QueryTimer(Path(job["output"])) as query_timer,
        query_timer.stage(
            "execution", "query execution including native semantic operators"
        ),
    ):
        result = module.run(
            tables, job, **({"model": model} if job["system"] == "lotus" else {})
        )
    if captured:
        if len(captured) != 1:
            raise ValueError("Expected exactly one final aggregate input")
        from experiments.system_comparison.references.worker import snapshot_frame

        frame = captured[0]
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame([r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in frame], dtype=object)
        if frame.empty and not len(frame.columns):
            frame = pd.DataFrame(columns=[c[0] for c in job["pre_aggregate_schema"]])
        if job["system"] == "lotus" and job.get("pre_aggregate_schema"):
            # LOTUS maps return text, but the frozen contract declares the
            # numeric outputs. Keep that schema even for all-NULL/empty input.
            schema = job["pre_aggregate_schema"]
            if list(frame.columns) != [column[0] for column in schema]:
                raise ValueError("LOTUS aggregate input differs from its declared schema")
            clean = frame.astype(object).where(pd.notna(frame), None)
            save(Path(job["output"]) / "pre-aggregate-result.json", normalize({
                "columns": schema, "rows": list(clean.itertuples(index=False, name=None))
            }))
        else:
            snapshot_frame(frame, Path(job["output"]))
    seconds = query_timer.result["query_seconds"]
    native_metrics = None
    if isinstance(result, tuple):
        result, cost, native_seconds = result
        native_metrics = {
            "reported_native_cost": cost,
            "reported_native_seconds": native_seconds,
        }
    if not isinstance(result, pd.DataFrame):
        if getattr(result, "execution_stats", None):
            stats = result.execution_stats
            save(
                Path(job["output"]) / "native-stats.json",
                stats.model_dump(mode="json")
                if hasattr(stats, "model_dump")
                else str(stats),
            )
        result = pd.DataFrame(
            [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in result],
            dtype=object,
        )
    expected = (
        job["pre_aggregate_schema"]
        if passthrough_aggregate
        else job["output_schema"]
    )
    if len(result.columns) == 0 and result.empty:
        result = pd.DataFrame(columns=[c[0] for c in expected])
    actual_names = list(result.columns)
    expected_names = [c[0] for c in expected]
    if actual_names != expected_names:
        used = set()
        unique = []
        for name in actual_names:
            candidate = str(name)
            suffix = 1
            while candidate.casefold() in used:
                candidate = str(name) + "_" + str(suffix)
                suffix += 1
            unique.append(candidate)
            used.add(candidate.casefold())
        if unique == expected_names:
            result.columns = unique
            save(
                Path(job["output"]) / "schema-mapping.json",
                {
                    "native": actual_names,
                    "canonical": unique,
                    "rule": "position-preserving duplicate-column name normalization",
                },
            )
        elif len(set(actual_names)) == len(actual_names) and set(actual_names) == set(
            expected_names
        ):
            result = result[expected_names]
            save(
                Path(job["output"]) / "schema-mapping.json",
                {
                    "native": actual_names,
                    "canonical": expected_names,
                    "rule": "frozen unique field-name order",
                },
            )
    if list(result.columns) != [c[0] for c in expected]:
        raise ValueError(
            f"Output schema differs: {list(result.columns)} != {[c[0] for c in expected]}"
        )
    clean = result.astype(object).where(pd.notna(result), None)
    payload = normalize(
        {"columns": expected, "rows": list(clean.itertuples(index=False, name=None))}
    )
    save(Path(job["output"]) / "result.json", payload)
    if llm_stats:
        llm_stats = {k: v for k, v in llm_stats.items() if k != "client"}
        save(Path(job["output"]) / "native-client-stats.json", llm_stats)
    if job["system"] == "lotus":
        from experiments.system_comparison.backends.lotus.cascade import save_plan

        save_plan(job)
    timing = query_timer.result
    timing["load_seconds"] = load_seconds
    timing["query_seconds"] = seconds
    timing.setdefault("schema", "native-phase-timing-v1")
    timing.setdefault("details", {})["worker_query_seconds"] = seconds
    timing["details"].update(
        runtime_setup_seconds=runtime_setup_seconds, input_policy="preload_before_query"
    )
    save(Path(job["output"]) / "timing.json", timing)
    plan_files = [
        str(p.relative_to(Path(job["output"])))
        for p in Path(job["output"]).rglob("plan*.json")
    ]
    aggregate_path = Path(job["output"]) / "pre-aggregate-result.json"
    return {
        "seconds": seconds,
        "execution_mode": module.__dict__.get("SYSTEM")
        + "_native_semantics_with_disclosed_host",
        "native_metrics": native_metrics,
        "llm": llm_stats,
        "result_rows": len(payload["rows"]),
        "timing": timing,
        "native_plan_files": plan_files,
        "native_optimizer_scope": "native graph, bounded only by explicit source SQL context barriers",
        **({"pre_aggregate": {"path": str(aggregate_path), "sha256": digest(aggregate_path)}} if aggregate_path.exists() else {}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    from experiments.system_comparison.data.memory import bind_database

    job = bind_database(job)
    folder = Path(job["output"])
    if (
        job.get("input_database_sha256")
        and digest(job["input_database"]) != job["input_database_sha256"]
    ):
        raise ValueError("Prepared input database changed")
    hashes = folder / "execution-source-hashes.json"
    if hashes.exists():
        sources = json.loads(hashes.read_text())
        for path in experiment_sources(ROOT):
            sources[str(path.relative_to(ROOT))] = digest(path)
        for parent in args.job.resolve().parents:
            revision = parent / "source-revisions.json"
            if not revision.is_file():
                continue
            for event in json.loads(revision.read_text()).get("revisions", []):
                for name in event.get("files", {}):
                    sources[name] = digest(ROOT / name)
            break
        save(hashes, sources)
    reason = stopped_system(args.job, job)
    if reason:
        save(
            folder / "summary.json",
            {
                "status": "cancelled",
                "reason": reason,
                "seconds": 0,
                "execution_suppressed": True,
                "accuracy": None,
                "query_id": job["id"],
                "system": job["system"],
                "native_mechanism_profile": job.get("native_mechanism_profile"),
            },
        )
        print("STOPPED", job["system"], job["id"], reason, flush=True)
        return
    import os

    save(
        folder / "cpu-affinity.json",
        {
            "worker_pid": os.getpid(),
            "requested_cpu_set": job.get("cpu_set"),
            "actual_cpu_ids": sorted(os.sched_getaffinity(0)),
        },
    )
    if digest(job["query_file"]) != job["query_sha256"]:
        raise ValueError("Frozen query hash mismatch")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler(folder / "native.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    class Errors(logging.Handler):
        messages = []

        def emit(self, record):
            self.messages.append(record.getMessage())

    errors = Errors(level=logging.ERROR)
    logging.getLogger().addHandler(errors)
    started = time.monotonic()
    summary = {
        "status": "failed",
        "accuracy": None,
        "query_id": job["id"],
        "system": job["system"],
    }
    try:
        if job["system"] == "moar":
            from experiments.system_comparison.backends.moar.worker import run

            stats = run(job)
        else:
            stats = sql_query(job) if job["system"] in {"sema", "plop"} else python_query(job)
        summary.update(stats, status="completed")
        timing = stats.get("timing") or {}
        for name in (
            "sampling_seconds",
            "planning_seconds",
            "execution_seconds",
            "host_seconds",
            "load_seconds",
        ):
            if name in timing:
                summary[name] = timing[name]
        if errors.messages:
            summary.update(
                status="partial",
                reason="Native framework logged errors; inspect native-errors.json",
            )
        if stats.get("llm", {}):
            native = stats["llm"]
            # After bounded protocol retries, Nirvana records native False/NULL fallbacks; unrecovered requests remain partial.
            if native.get("failed_requests") or native.get("missing_tags"):
                summary.update(
                    status="partial",
                    reason="Unrecovered native model/format failures; inspect client-events.jsonl",
                )
    except Exception as exc:
        summary.update(reason=f"{type(exc).__name__}: {exc}")
        if job["system"] == "abacus":
            from experiments.system_comparison.backends.abacus.optimizer import (
                AbacusSearchTimeout,
            )

            if isinstance(exc, AbacusSearchTimeout):
                summary.update(status="timeout", timed_out=True, timeout_scope="plan_search")
        (folder / "error.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        summary.setdefault("seconds", time.monotonic() - started)
        summary["worker_wall_seconds"] = time.monotonic() - started
        save(folder / "native-errors.json", errors.messages)
        summary["execution_resources"] = resource_snapshot(
            job["system"], batch_size=job.get("batch_size", BATCH_SIZE),
            gateway_request_limit=job.get("gateway_concurrency_limit", 32),
            request_concurrency=job.get("native_concurrency", CONCURRENCY),
            abacus_queue_batch_size=job.get("abacus_queue_batch_size"),
        )
        summary["native_mechanism_profile"] = job.get(
            "native_mechanism_profile", "legacy"
        )
        save(folder / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, default=str), flush=True)
    raise SystemExit(0 if summary["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
