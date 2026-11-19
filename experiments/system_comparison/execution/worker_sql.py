"""Execute one frozen v4 query and retain native errors, stats and final rows."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

from experiments.core.io import digest, save
from experiments.core.paths import ROOT
from experiments.core.resources import BATCH_SIZE, CONCURRENCY


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
    from experiments.system_comparison.backends.abacus.compat import (
        install_short_output_prompts,
    )

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
