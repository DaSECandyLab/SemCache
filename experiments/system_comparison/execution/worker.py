"""Execute one frozen v4 query and retain native errors, stats and final rows."""

from __future__ import annotations

import argparse
import importlib.util
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


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stopped_system(job_path, job):
    """Honor an explicit per-run stop without stopping another model lane."""
    for parent in job_path.resolve().parents:
        control = parent / "control.json"
        if not (parent / "campaign.json").is_file() or not control.is_file():
            continue
        settings = json.loads(control.read_text())
        if job["system"] in settings.get("disabled_systems", []) or job.get(
            "variant", job.get("name")
        ) in settings.get("disabled_variants", []):
            return settings.get("reason", "System stopped by user")
    return None


def load_tables(job):
    import duckdb
    import pandas as pd

    tables = {}
    with duckdb.connect(
        job["input_database"], read_only=True, config={"threads": CONCURRENCY}
    ) as con:
        for name in job["inputs"]:
            cursor = con.execute('SELECT * FROM "' + name + '"')
            schema = [(c[0], str(c[1])) for c in cursor.description]
            frame = pd.DataFrame(
                cursor.fetchall(), columns=[c[0] for c in schema], dtype=object
            )
            if job["system"] == "nirvana":
                for column, kind in schema:
                    if "INT" in kind.upper():
                        frame[column] = frame[column].astype(
                            "Int64" if frame[column].isna().any() else "int64"
                        )
                    elif kind.upper() in {"DOUBLE", "FLOAT", "REAL"}:
                        frame[column] = frame[column].astype("float64")
            frame.attrs["duckdb_schema"] = schema
            tables[name] = frame
    return tables


def abacus_config(job):
    formal = load_module("v4_abacus_formal", ROOT / "systems/abacus/formal.py")
    config = formal.native_config(job)
    config.max_workers = job.get("native_concurrency", CONCURRENCY)
    config.join_parallelism = job.get("native_concurrency", CONCURRENCY)
    config.batch_size = job.get("abacus_queue_batch_size", job.get("batch_size", BATCH_SIZE))
    # Limit process concurrency explicitly, without replacing the native policy.
    save(
        Path(job["output"]) / "abacus-model-pool.json",
        {
            "policy": job["policy"],
            "models": [
                {"alias": alias, "real_id": entry["id"]}
                for alias, entry in zip(
                    formal.model_pool(job), job["models"], strict=True
                )
            ],
        },
    )
    return config


def configure_nirvana_resources(job=None):
    """Make the native semaphore explicit, including optimizer-created ops."""
    import inspect

    from nirvana.ops.base import BaseOperation
    from nirvana.ops.join import JoinOperation

    batch_size = (job or {}).get("batch_size", BATCH_SIZE)
    concurrency = (job or {}).get("native_concurrency", CONCURRENCY)
    for cls, settings in (
        (BaseOperation, {"rate_limit": concurrency}),
        (JoinOperation, {"rate_limit": concurrency, "batch_size": batch_size}),
    ):
        # Reconfigure from the native constructor so a previous job's batch
        # does not survive a second configuration in the same process.
        original = getattr(cls.__init__, "_comparison_original_init", cls.__init__)
        signature = inspect.signature(original)

        def configured(
            self,
            *args,
            _original=original,
            _signature=signature,
            _settings=settings,
            **kwargs,
        ):
            bound = _signature.bind(self, *args, **kwargs)
            for name, value in _settings.items():
                if name in _signature.parameters:
                    bound.arguments[name] = value
            return _original(*bound.args, **bound.kwargs)

        configured._comparison_resources = True
        configured._comparison_original_init = original
        cls.__init__ = configured


def configure_abacus(job):
    # Local models occasionally repeat a valid JSON object in a markdown block;
    # accept the first balanced object while retaining the native Abacus
    # generator, retries, and output type validation.
    from experiments.system_comparison.backends.abacus.compat import (
        install_tolerant_json_parser,
    )

    install_tolerant_json_parser()
    from experiments.system_comparison.backends.abacus.compat import install_short_output_prompts

    install_short_output_prompts()
    from experiments.validation import native_api

    adapter = native_api.load_adapter("abacus")
    adapter.native_config = abacus_config
    original = native_api.load_adapter
    native_api.load_adapter = lambda system: (
        adapter if system == "abacus" else original(system)
    )
    sys.modules["systems.abacus.validate"] = adapter


def configure_nirvana(job):
    configure_nirvana_resources(job)
    formal = load_module("v4_nirvana_formal", ROOT / "systems/nirvana/formal.py")
    stats = formal.configure_client(job)
    formal.configure_embedding(job, stats["client"])
    # Frozen SQL text columns remain text even when they contain image/PDF URLs.
    # This adapter only supplies the declared input types, not optimizer rules.
    import nirvana.dataframe.frame as frame_module

    original_infer = frame_module.infer_and_convert_dtype
    text_fields = {
        name
        for item in job["inputs"].values()
        for name, kind in item["columns"]
        if kind.upper() in {"VARCHAR", "TEXT"}
    }

    def declared_type(column):
        if (
            column.name in text_fields
            or column.name in getattr(frame_module, "_comparison_text_columns", set())
            or str(column.name).startswith("arg")
        ):
            return column, str
        return original_infer(column)

    frame_module.infer_and_convert_dtype = declared_type
    from experiments.system_comparison.backends.nirvana.compat import install

    install()
    save(
        Path(job["output"]) / "input-type-adapter.json",
        {
            "text_fields": sorted(text_fields),
            "policy": "Respect frozen SQL types; avoid object-integer string inference and implicit text-URL media conversion",
        },
    )
    return stats


def configure_lotus(job):
    import lotus
    import pandas as pd
    from lotus.models import LM

    from experiments.system_comparison.backends.lotus.cascade import FourModelCascade
    from experiments.system_comparison.backends.lotus.compat import (
        install_boolean_protocol,
    )

    install_boolean_protocol()

    models = [
        LM(
            model="openai/" + m["alias"],
            api_base=m["endpoint"],
            api_key="EMPTY",
            max_ctx_len=m["context_length"],
            max_tokens=job["max_tokens"],
            # LOTUS chunks independent one-input HTTP requests with this
            # setting; it is request concurrency, not rows in a prompt.
            max_batch_size=job.get("native_concurrency", CONCURRENCY),
            rate_limit=None,
            timeout=job["request_timeout"],
            num_retries=0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        for m in job["models"]
    ]
    lotus.settings.configure(
        lm=models[-1], helper_lm=models[0], parallel_groupby_max_threads=CONCURRENCY
    )
    cascade = FourModelCascade(models, job["output"])

    def filter(frame, task, *args, **kwargs):
        return cascade.filter(frame, task, *args, **kwargs)

    pd.DataFrame.sem_filter = filter
    save(
        Path(job["output"]) / "lotus-cascade-config.json",
        {
            "extension": len(models) > 2,
            "cascade": "helper_main" if len(models) == 2 else "multi_model_extension",
            "models": [m["id"] for m in job["models"]],
            "map_model": job["models"][-1]["id"],
            "precision_target_per_level": 0.9,
            "recall_target_per_level": 0.9,
            "failure_probability_per_level": 0.05 / (len(models) - 1),
            "model_count": len(models),
            "boolean_protocol": "whitespace-normalized-logprobs-and-yes-no-v1",
            "components": "native sem_filter, log-probability scoring, importance sampling and threshold learning",
            "join_optimizer": "native search-filter/map-search-filter cascade; local embedding proxy and 12B oracle",
            "join_input_adapter": "lossless per-side JSON packing of all semantic arguments",
            "join_cascade_min_pairs": 100,
            "global_reference_agreement_guaranteed": False,
        },
    )
    return models[-1]


def sql_query(job):
    from experiments.system_comparison.execution.sql import (
        Session,
        literal,
        parse_result,
        sql_setup,
    )

    folder = Path(job["output"])
    system = job["system"]
    text = Path(job["query_file"]).read_text()
    load_preparation_started = time.monotonic()
    setup = sql_setup(job)
    load_preparation_seconds = time.monotonic() - load_preparation_started
    import os

    env = os.environ.copy()
    sql_folder = folder / "native-sql"
    sql_folder.mkdir()
    if system == "sema":
        from experiments.system_comparison.backends.sema import (
            settings as optimizer_settings,
        )
        from experiments.system_comparison.backends.sema import (
            transport_binary,
        )

        optimization, selected = optimizer_settings(text)
        save(folder / "sema-optimizer.json", selected)
        exe = transport_binary(
            ROOT / "systems/sema/vendor/sema-paper/Sema",
            sql_folder / "Sema",
            job["timeout"],
        )
        settings = [
            f"SET llm_url={literal(job['endpoint'] + '/chat/completions')};",
            f"SET llm_model={literal(job['model'])};",
            "SET llm_api_key='EMPTY';",
            "SET llm_rate_limit=0;",
            f"SET semantic_batch_size={job.get('batch_size', BATCH_SIZE)};",
            *optimization,
            "SET semantic_filter_accuracy_threshold=0.9;",
            "SET semantic_filter_latency_first=true;",
            f"SET semantic_filter_batch_size={job.get('batch_size', BATCH_SIZE)};",
            f"SET threads={CONCURRENCY};",
        ]
    else:
        port = job.get("gateway_port", 8899)
        exe = (
            ROOT
            / "systems/plop"
            / ("duckdb-local" if port == 8899 else f"duckdb-local-{port}")
        )
        if runtime := job.get("plop_runtime"):
            exe = Path(runtime["binary"])
            if digest(exe) != runtime["binary_sha256"]:
                raise ValueError("Configured PLOP binary changed")
        from experiments.system_comparison.scheduling.campaign import (
            native_gateway_port,
        )

        if native_gateway_port(exe) != port:
            raise ValueError(
                "PLOP binary transport does not match its model lane gateway"
            )
        env.update(
            SEMANTIC_OPTIMIZER="true",
            DUCKDB_SEMANTIC_MODE="costmodel",
            API_KEY="EMPTY",
            NUM_LLM_WORKERS=str(CONCURRENCY),
        )
        settings = [f"SET threads={CONCURRENCY};"]
    runtime_setup = "\n".join(settings)
    raw = folder / "native-result.json"
    explain = folder / "prepared-explain.csv"
    prepare_program = f"PREPARE __v4_query AS {text.rstrip().rstrip(';')};\n.mode csv\n.once {explain}\nEXPLAIN EXECUTE __v4_query;\n"
    execute_program = f"PRAGMA enable_profiling='json';\nPRAGMA profiling_output={literal(folder / 'profile.json')};\n.mode json\n.once {raw}\nEXECUTE __v4_query;"
    program = prepare_program + execute_program
    save(
        folder / "native-config.json",
        {
            "settings": settings,
            "environment": {
                k: env[k]
                for k in (
                    "SEMANTIC_OPTIMIZER",
                    "DUCKDB_SEMANTIC_MODE",
                    "NUM_LLM_WORKERS",
                )
                if k in env
            },
            "binary": str(exe),
            "binary_sha256": digest(exe),
            **(
                {"plop_runtime": job["plop_runtime"]} if job.get("plop_runtime") else {}
            ),
        },
    )
    (folder / "program.sql").write_text(runtime_setup + "\n" + setup + "\n" + program)
    # Separate logs prevent the native SQL process and Python from writing the
    # same file through independently truncated file descriptors.
    runtime_started = time.monotonic()
    session = Session(exe, sql_folder, env)
    try:
        configured = session.execute(runtime_setup, job["timeout"])
        if configured["status"] != "completed":
            raise RuntimeError("Native runtime configuration failed")
        runtime_setup_seconds = time.monotonic() - runtime_started
        load = session.execute(setup, job["timeout"])
        if load["status"] != "completed":
            raise RuntimeError("SQL input setup failed; inspect native-sql/stderr.log")
        load["seconds"] += load_preparation_seconds
        save(
            folder / "input-loading.json",
            {
                "policy": "preload_before_query",
                "storage": "native_memory_tables",
                "source_database": job.get("input_database"),
                "tables": list(job["inputs"]),
                "source_detached": job.get("input_loading") == "database_memory",
                "load_seconds": load["seconds"],
                "runtime_setup_seconds": runtime_setup_seconds,
            },
        )
        prepared = session.execute(prepare_program, job["timeout"])
        if prepared["status"] != "completed":
            raise RuntimeError(
                "Native plan preparation failed; inspect native-sql/stderr.log"
            )
        capture = None
        if system in {"sema", "plop"} and job.get("sql_capture_aggregate_inputs"):
            from SemCache.optimizer.reference_answers.sql.aggregate import capture_plan

            capture = capture_plan(text, table="__comparison_aggregate_input")
        aggregate_input = None
        if capture:
            # Materialize the semantic result once, then consume that same
            # relation for both the final aggregate and the saved snapshot.
            # This native release does not profile CTAS through enable_profiling.
            # EXPLAIN ANALYZE executes CTAS once and emits its actual operator
            # counters, including the native semantic AQE counters.
            materialize = (
                ".mode json\n"
                + f".once {folder / 'profile.json'}\n"
                + "EXPLAIN (ANALYZE, FORMAT JSON) CREATE TEMP TABLE __comparison_aggregate_input AS "
                + capture["input_sql"] + ";"
            )
            (folder / "aggregate-capture.sql").write_text(materialize + "\n" + capture["aggregate_sql"] + ";\n")
            result = session.execute(materialize, job["timeout"])
            if result["status"] == "completed":
                final = session.execute(
                    "PRAGMA disable_profiling;\n.mode json\n"
                    + f".once {raw}\n" + capture["aggregate_sql"] + ";", job["timeout"]
                )
                result = {**final, "seconds": result["seconds"] + final["seconds"]}
            if result["status"] == "completed":
                # Export time is outside the measured query execution span.
                schema_path = folder / "pre-aggregate-schema.json"
                native_path = folder / "pre-aggregate-native.json"
                exported = session.execute(
                    f".mode json\n.once {schema_path}\nDESCRIBE __comparison_aggregate_input;\n"
                    + f".once {native_path}\nSELECT * FROM __comparison_aggregate_input;", job["timeout"]
                )
                if exported["status"] != "completed":
                    raise RuntimeError("Aggregate input export failed")
                schema = [[row["column_name"], row["column_type"]] for row in json.loads(schema_path.read_text())]
                target = folder / "pre-aggregate-result.json"
                save(target, parse_result(native_path, schema))
                aggregate_input = {"path": str(target), "sha256": digest(target)}
        else:
            result = session.execute(execute_program, job["timeout"])
        if result["status"] != "completed":
            raise RuntimeError(
                "Native SQL failed: " + (sql_folder / "stderr.log").read_text()[-1500:]
            )
        if system == "sema":
            from experiments.system_comparison.backends.sema import profile_evidence

            save(
                folder / "sema-mechanism.json",
                profile_evidence(folder / "profile.json", selected),
            )
        save(folder / "result.json", parse_result(raw, job["output_schema"]))
        from experiments.system_comparison.reporting.sql_timing import phases
        from experiments.system_comparison.reporting.timing import write_phase_timing

        native_phases = phases(
            folder / "profile.json",
            system,
            execution_wall_seconds=result["seconds"],
            planning_seconds=prepared["seconds"],
        )
        timing = write_phase_timing(
            folder,
            query_seconds=prepared["seconds"] + result["seconds"],
            load_seconds=load["seconds"],
            execution_seconds=native_phases["execution_seconds"],
            sampling_seconds=native_phases["sampling_seconds"],
            planning_seconds=prepared["seconds"],
            unavailable=native_phases["unavailable"],
            method=native_phases["method"],
            details={
                **native_phases["details"],
                "runtime_setup_seconds": runtime_setup_seconds,
                "input_policy": "preload_before_query",
            },
        )
        return {
            "seconds": prepared["seconds"] + result["seconds"],
            "load_seconds": load["seconds"],
            "execution_mode": "native_sql",
            "native_format_fallbacks": native_phases["native_format_fallbacks"],
            "optimizer": "native AQE/fusion/DP settings"
            if system == "sema"
            else "native costmodel",
            "timing": timing,
            **({"pre_aggregate": aggregate_input, "aggregate_capture": "same-execution materialized native SQL input; export excluded from query clock"} if aggregate_input else {}),
        }
    finally:
        session.close(kill=True)


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
            from experiments.system_comparison.backends.lotus.compat import normalize_numeric_columns

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
            from experiments.system_comparison.backends.abacus.optimizer import AbacusSearchTimeout

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
