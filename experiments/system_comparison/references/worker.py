"""Teacher-only instrumentation, confined to this new worker process.

Reuse native query execution without changing any backend's source or prompts.
SQL aggregate inputs are materialized once and consumed by the native aggregate;
Python aggregates are observed at their existing deterministic host boundary.
"""

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

from experiments.core.io import digest, save


def wait_for_restart(job):
    """Pause a fresh worker at the requested controller's query boundary."""
    run = Path(job["output"]).parents[2]
    request_path = run / "restart-request.json"
    try:
        request = json.loads(request_path.read_text())
    except FileNotFoundError:
        return
    controller_pid = request.get("controller_pid")
    if request.get("phase") != "pending" or os.getppid() != controller_pid:
        return
    if job["system"] in request.get("continue_systems", []):
        return
    save(
        run / "restart-ready.json",
        {
            "nonce": request["nonce"],
            "controller_pid": controller_pid,
            "worker_pid": os.getpid(),
            "system": job["system"],
            "query": job.get("id") or f"{job['dataset']}/{job['query']}",
            "phase": "ready",
            "execution_started": False,
        },
    )
    while True:
        if os.getppid() != controller_pid:
            raise SystemExit("Restart controller exited before query execution")
        try:
            current = json.loads(request_path.read_text())
        except FileNotFoundError:
            return
        if current.get("phase") != "pending":
            return
        time.sleep(0.1)


def snapshot_frame(frame, folder):
    import duckdb
    import pandas as pd

    from experiments.system_comparison.reporting.results import normalize

    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(
            [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in frame],
            dtype=object,
        )
    if len(frame.columns) == 0:
        raise ValueError("Aggregate input has no native schema")
    with duckdb.connect(config={"threads": 1}) as connection:
        connection.register("snapshot", frame)
        schema = [
            [c[0], str(c[1])]
            for c in connection.execute("SELECT * FROM snapshot LIMIT 0").description
        ]
        rows = connection.execute("SELECT * FROM snapshot").fetchall()
    save(
        folder / "pre-aggregate-result.json",
        normalize({"columns": schema, "rows": rows}),
    )
    return frame


def single_lotus(job):
    import lotus
    from lotus.models import LM

    model = LM(
        model="openai/semantic-large",
        api_base=job["endpoint"],
        api_key="EMPTY",
        max_ctx_len=job["models"][0]["context_length"],
        max_tokens=job["max_tokens"],
        max_batch_size=16,
        rate_limit=None,
        timeout=job["request_timeout"],
        num_retries=0,
    )
    lotus.settings.configure(
        lm=model, helper_lm=None, enable_cache=False, parallel_groupby_max_threads=16
    )
    save(
        Path(job["output"]) / "teacher-lotus.json",
        {
            "mode": "native_single_teacher",
            "four_model_extension": False,
            "filter_helper": None,
            "native_join_retrieval_optimizer": True,
            "response_cache": False,
        },
    )
    return model


def main():
    from experiments.system_comparison.execution import worker
    from experiments.system_comparison.execution.sql import Session, parse_result
    from experiments.validation import faithful_runtime

    job = json.loads(Path(sys.argv[sys.argv.index("--job") + 1]).read_text())
    folder = Path(job["output"])
    wait_for_restart(job)
    aggregate = job.get("aggregate_capture")
    original_aggregate = faithful_runtime.aggregate_frame
    original_execute = Session.execute

    from experiments.system_comparison.backends import sema as sema_backend

    original_transport_binary = sema_backend.transport_binary

    def transport_binary(source, target, _query_timeout):
        # The shared worker passes the query deadline; this campaign configures
        # the existing transport copy with its separate HTTP request timeout.
        return original_transport_binary(source, target, job["request_timeout"])

    def observed_aggregate(frame, spec):
        # An empty native collection may have lost column metadata. Recover the
        # frozen pre-aggregate schema, retaining zero rows rather than inventing data.
        import pandas as pd

        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(
                [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in frame],
                dtype=object,
            )
        if frame.empty and len(frame.columns) == 0:
            frame = pd.DataFrame(columns=[c[0] for c in job["pre_aggregate_schema"]])
        frame = snapshot_frame(frame, folder)
        return original_aggregate(frame, spec)

    def execute(session, text, *args, **kwargs):
        if (
            aggregate
            and "EXECUTE __v4_query;" in text
            and "EXPLAIN EXECUTE" not in text
        ):
            native = folder / "pre-aggregate-native.json"
            schema_file = folder / "pre-aggregate-schema.json"
            replacement = (
                "CREATE TEMP TABLE __reference_aggregate_input AS "
                + aggregate["input_sql"]
                + ";\nPRAGMA disable_profiling;\n"
                + f".mode json\n.once {schema_file}\nDESCRIBE __reference_aggregate_input;\n"
                + f".once {native}\nSELECT * FROM __reference_aggregate_input;\n"
                + f".once {folder / 'native-result.json'}\n"
                + aggregate["aggregate_sql"]
                + ";"
            )
            text = text.replace("EXECUTE __v4_query;", replacement)
            (folder / "aggregate-capture.sql").write_text(text)
            result = original_execute(session, text, *args, **kwargs)
            if result["status"] == "completed":
                schema = [
                    [r["column_name"], r["column_type"]]
                    for r in json.loads(schema_file.read_text())
                ]
                save(folder / "pre-aggregate-result.json", parse_result(native, schema))
            return result
        return original_execute(session, text, *args, **kwargs)

    from experiments.system_comparison.backends.lotus import cascade

    original_plan = cascade.save_plan

    def lotus_plan(job):
        events = Path(job["output"]) / "lotus-join-events.jsonl"
        if events.exists():
            records = [
                json.loads(line)
                for line in events.read_text().splitlines()
                if line.strip()
            ]
            for event in records:
                event["join_proxy"] = (
                    "native local embedding search; oracle/main is the configured Teacher"
                )
            events.write_text(
                "".join(
                    json.dumps(event, ensure_ascii=False) + "\n" for event in records
                )
            )
        original_plan(job)
        path = Path(job["output"]) / "plan.json"
        plan = json.loads(path.read_text())
        plan["four_model_filter_extension"] = False
        plan["teacher_single_model"] = True
        save(path, plan)

    with (
        patch.object(worker, "configure_lotus", single_lotus),
        patch.object(sema_backend, "transport_binary", transport_binary),
        patch.object(faithful_runtime, "aggregate_frame", observed_aggregate),
        patch.object(Session, "execute", execute),
        patch.object(cascade, "save_plan", lotus_plan),
    ):
        try:
            worker.main()
        finally:
            pre = folder / "pre-aggregate-result.json"
            summary_path = folder / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                summary.setdefault("execution_resources", {}).update(
                    gateway_concurrency_limit=16,
                    teacher_requests_per_active_system=16,
                    concurrency_scope="16 native workers and at most 16 upstream requests per active system",
                )
                if "batch_size" in job:
                    summary["execution_resources"]["batch_size"] = job["batch_size"]
                if job.get("plop_runtime"):
                    summary["plop_runtime"] = job["plop_runtime"]
                if pre.exists():
                    summary["pre_aggregate"] = {"path": str(pre), "sha256": digest(pre)}
                save(summary_path, summary)


if __name__ == "__main__":
    main()
