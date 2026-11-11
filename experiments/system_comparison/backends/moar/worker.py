"""Run genuine upstream MOAR search, then its selected full-data pipeline."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

from experiments.core.io import digest, save
from experiments.system_comparison.backends.moar.pipeline import Builder, AMENITIES_COMPACT_INSTRUCTION
from experiments.system_comparison.backends.moar.runtime import install_relational_join, install_nullable_numbers
from experiments.system_comparison.reporting.results import normalize, value_for_type


COMPACT_OUTPUT_INSTRUCTION = (
    "Output only the required result, with no explanation or unrelated details. "
    "Use concise phrases for text while preserving all information and exact wording "
    "required by the task. Preserve numeric values, missing-value rules and the "
    "required output format. Do not invent or omit task-relevant facts."
)


def concise_execution_messages(model, messages, aliases):
    """Apply to candidate execution models, not MOAR's rewrite agent."""
    if (not isinstance(model, str) or model.removeprefix("openai/") not in aliases
            or not isinstance(messages, list)):
        return messages
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, dict) and message.get("role") == "user" and isinstance(message.get("content"), str):
            if COMPACT_OUTPUT_INSTRUCTION in message["content"]:
                return messages
            copied = list(messages)
            copied[index] = {**message, "content": message["content"] + "\n\n" + COMPACT_OUTPUT_INSTRUCTION}
            return copied
    return messages


def install_client(job):
    """Route both DocETL and rewrite-agent calls through the logged gateway."""
    import litellm

    original = litellm.completion
    aliases = {model["alias"] for model in job["models"]}
    request_stage = {"value": "planning"}

    def set_request_stage(stage):
        if stage not in {"planning", "execution"}:
            raise ValueError(f"Unknown MOAR request stage: {stage}")
        request_stage["value"] = stage

    def completion(*args, **kwargs):
        # A gateway retry must be able to finish before the native client
        # times out; the upstream timeout remains independently enforced.
        kwargs.update(api_base=job["endpoint"], api_key="EMPTY", timeout=job.get("client_request_timeout", job["request_timeout"]), num_retries=0)
        kwargs["extra_headers"] = {
            **(kwargs.get("extra_headers") or {}),
            "X-Experiment-Request-Stage": request_stage["value"],
        }
        if "messages" in kwargs:
            kwargs["messages"] = concise_execution_messages(kwargs.get("model"), kwargs["messages"], aliases)
        # Operator requests carry their own 1024-token limit. Rewriting a
        # complete pipeline needs room for executable code/configuration.
        kwargs.setdefault("max_tokens", 4096)
        kwargs.setdefault("extra_body", {})["chat_template_kwargs"] = {"enable_thinking": False}
        return original(*args, **kwargs)

    litellm.completion = completion
    import docetl.operations.utils.api as api

    api.completion = completion
    for model in [*job["models"], *([job["moar_agent_model"]] if job.get("moar_agent_model") else [])]:
        litellm.register_model({"openai/" + model["alias"]: {
            "max_tokens": model["context_length"], "max_input_tokens": model["context_length"],
            "max_output_tokens": 4096, "input_cost_per_token": 0.0, "output_cost_per_token": 0.0,
            "litellm_provider": "openai", "mode": "chat", "supports_response_schema": True,
        }})
    return set_request_stage


def read_rows(builder, rows):
    schema = builder.output_schema
    # Canonical SQL duplicate labels are normalized positionally by the
    # existing worker. Map directly by ordered internal identifiers here.
    return normalize({"columns": schema, "rows": [
        [row[internal] for _, internal in builder.output_mapping] for row in rows
    ]})


def evaluation(builder, job):
    root = Path(job["reference_directory"]) / "nirvana" / job["id"]
    path = root / ("pre-aggregate-result.json" if builder.aggregate else "answer.json")
    source = json.loads(path.read_text())
    reference = source if builder.aggregate else source["result"]
    columns = [column[0] for column in reference["columns"]]
    keys = {
        "estate": ["Unnamed: 0"], "movies": ["id"], "movies_sema": ["movie_id"], "appstore": ["id"],
    }.get(job["dataset"], [name for name, _ in builder.output_schema])
    if builder.aggregate and builder.aggregate["column"] in {"Price", "RuntimeMinutes"}:
        keys = keys + [builder.aggregate["column"]]
    key_indices = [columns.index(name) for name in keys]
    expected = reference["rows"]
    if builder.sample_ids is not None:
        name, allowed = builder.sample_ids
        index = columns.index(name)
        expected = [row for row in expected if row[index] in allowed]
    types = dict(reference["columns"])

    def encode(values):
        return json.dumps([value_for_type(value, types[name]) for name, value in zip(keys, values, strict=True)], ensure_ascii=False, separators=(",", ":"))

    gold = Counter(encode([row[i] for i in key_indices]) for row in expected)
    save(Path(job["output"]) / "optimizer-evaluation.json", {
        "metric": "row_f1", "fields": keys, "reference": str(path), "reference_sha256": digest(path),
        "reference_rows_in_scope": len(expected), "sample_size": 40 if builder.sample_ids else None,
        "sample_policy": "uniform seed42; independent of reference labels", "free_text_judgment": False,
    })

    def evaluate(path):
        payload = read_rows(builder, json.loads(Path(path).read_text()))
        candidate_columns = [name for name, _ in payload["columns"]]
        indices = [candidate_columns.index(name) for name in keys]
        actual = Counter(encode([row[i] for i in indices]) for row in payload["rows"])
        matched = sum((actual & gold).values())
        total = sum(actual.values()) + sum(gold.values())
        return {"row_f1": 2 * matched / total if total else 1.0}

    return evaluate


def reuse_selected_plan(job, builder, folder):
    """Rebind a verified selected MOAR plan to this run's full input files."""
    import yaml

    source = json.loads((folder / "plan-reuse-input.json").read_text())
    text = source["source_pipeline_yaml"]
    if (source.get("query_id") != job["id"] or source.get("query_sha256") != job["query_sha256"]
            or source.get("input_database_sha256") != job.get("input_database_sha256")
            or source.get("source_plan_sha256") != hashlib.sha256(text.encode()).hexdigest()):
        raise ValueError("Saved MOAR plan does not match the current query or input database")
    chosen = yaml.safe_load(text)
    if set(chosen.get("datasets", {})) != set(builder.full_paths):
        raise ValueError("Saved MOAR plan has different source datasets")
    chosen["default_lm_api_base"] = job["endpoint"]
    for name, path in builder.full_paths.items():
        chosen["datasets"][name]["path"] = path
    for operation in chosen.get("operations", []):
        schema = operation.get("output", {}).get("schema", {})
        if (operation.get("type") == "map" and isinstance(schema, dict)
                and "Amenities" in schema and isinstance(operation.get("prompt"), str)
                and AMENITIES_COMPACT_INSTRUCTION.strip() not in operation["prompt"]):
            operation["prompt"] += "\n" + AMENITIES_COMPACT_INSTRUCTION.strip()
    return chosen, source


def run(job):
    import pandas as pd

    folder = Path(job["output"])
    request_concurrency = job.get("native_concurrency", 16)
    os.environ["DOCETL_HOME_DIR"] = str(folder / "docetl-cache")
    os.environ["OPENAI_API_KEY"] = "EMPTY"
    os.environ["OPENAI_API_BASE"] = job["endpoint"]
    os.environ["OPENAI_BASE_URL"] = job["endpoint"]
    set_request_stage = install_client(job)
    install_relational_join()
    install_nullable_numbers()
    from experiments.system_comparison.execution.worker import load_tables, load_module
    from experiments.core.paths import ROOT
    from docetl.moar.optimizer import run_moar
    from docetl.runner import DSLRunner

    audit_module = load_module("moar_validation_audit", ROOT / "systems/moar/validate.py")
    load_started = time.monotonic()
    tables = load_tables(job)
    load_seconds = time.monotonic() - load_started
    started = time.monotonic()
    builder = Builder(job, tables)
    full_config = builder.build()
    config = deepcopy(full_config)
    for name, path in builder.sample_paths.items():
        config["datasets"][name]["path"] = path
    import yaml

    (folder / "pipeline.yaml").write_text(yaml.safe_dump(full_config, sort_keys=False, allow_unicode=True))
    reused = (folder / "plan-reuse-input.json").is_file()
    save(folder / "moar-settings.json", {
        "entrypoint": "docetl.runner.DSLRunner with saved selected plan" if reused else "docetl.moar.optimizer.run_moar",
        "max_iterations": 0 if reused else 20, "max_threads": request_concurrency,
        "max_concurrent_agents": 3, "models": [m["alias"] for m in job["models"]],
        "agent_model": job.get("moar_agent_model", {}).get("id", "semantic-large"),
        "agent_config": job.get("moar_agent_config"), "batch_size": 1, "reference_directory": job["reference_directory"],
        "gateway_upstream_timeout_seconds": job["request_timeout"],
        "client_request_timeout_seconds": job.get("client_request_timeout", job["request_timeout"]),
        "ordinary_join_adapter": "SQL hash join at native DocETL binary DAG boundary",
        "cost_unit": "candidate local execution monetary cost is zero; rewrite agent provider billing is not inferred from alias-priced cost",
    })
    if reused:
        chosen, reuse = reuse_selected_plan(job, builder, folder)
        planning_seconds, iterations = 0.0, 0
    else:
        evaluate = evaluation(builder, job)
        first_name = next(iter(config["datasets"]))
        search_started = time.monotonic()
        with audit_module.LLMAudit(folder / "llm-validation.jsonl") as audit:
            search = run_moar(
                config, eval_fn=evaluate, metric_key="row_f1", models=["openai/" + m["alias"] for m in job["models"]],
                agent_model="openai/" + job.get("moar_agent_model", {}).get("alias", "semantic-large"),
                max_iterations=20, max_threads=request_concurrency, max_concurrent_agents=3,
                dataset_name=first_name, dataset_path=config["datasets"][first_name]["path"], save_dir=str(folder / "search"),
            )
        planning_seconds = time.monotonic() - search_started
        iterations = search.iterations
        save(folder / "moar-search-summary.json", {
            "iterations": iterations, "plans": len(search.all_plans), "frontier": len(search.frontier),
            "seconds": planning_seconds, "llm_validation": audit.summary(),
        })
        if iterations == 0:
            gateway_log = folder / "gateway.jsonl"
            if gateway_log.exists():
                import re

                match = re.search(r"input.{0,30}?(\d+) tokens.{0,120}?context length.{0,30}?(\d+)", gateway_log.read_text(errors="replace"), flags=re.I)
                if match:
                    raise RuntimeError(f"MOAR rewrite input needs {match[1]} tokens, but served context is {match[2]}; no rewrite completed")
            raise RuntimeError("MOAR completed no rewrite iteration; baseline candidates alone do not constitute completed MOAR optimization")
        selected = search.best()
        if selected is None:
            raise RuntimeError("MOAR has no executable frontier plan")
        chosen = yaml.safe_load(Path(selected.yaml_path).read_text())
    # Restore full inputs only after selecting a plan on optimization data.
    for name, path in builder.full_paths.items():
        if name not in chosen["datasets"]:
            raise ValueError("MOAR removed a required source dataset")
        chosen["datasets"][name]["path"] = path
    chosen["pipeline"]["output"]["path"] = str(folder / "native-output.json")
    (folder / "selected-pipeline.yaml").write_text(yaml.safe_dump(chosen, sort_keys=False, allow_unicode=True))
    execute_started = time.monotonic()
    set_request_stage("execution")
    with audit_module.LLMAudit(folder / "execution-llm-validation.jsonl") as execution_audit:
        runner = DSLRunner(chosen, max_threads=request_concurrency)
        rows, _ = runner.run()
    validation = execution_audit.summary()
    if validation["unvalidated_calls"] or validation["exceptions"]:
        raise RuntimeError("Selected MOAR pipeline has unvalidated model outputs")
    payload = read_rows(builder, rows)
    pre = None
    if builder.aggregate:
        # Capture exactly the selected full-data pipeline output, not samples
        # and not a second semantic execution.
        save(folder / "pre-aggregate-result.json", payload)
        pre = {"path": str(folder / "pre-aggregate-result.json"), "sha256": digest(folder / "pre-aggregate-result.json")}
        from experiments.validation.faithful_runtime import aggregate_frame

        frame = pd.DataFrame(payload["rows"], columns=[name for name, _ in payload["columns"]], dtype=object)
        result = aggregate_frame(frame, builder.aggregate)
        payload = normalize({"columns": job["output_schema"], "rows": list(result.itertuples(index=False, name=None))})
    save(folder / "result.json", payload)
    execution_seconds = time.monotonic() - execute_started
    seconds = time.monotonic() - started
    timing = {"query_seconds": seconds, "load_seconds": load_seconds, "planning_seconds": planning_seconds,
              "execution_seconds": execution_seconds, "sampling_seconds": 0.0 if reused else None,
              "method": ("saved MOAR selected plan reused; no new search or sampling; selected full-data execution"
                         if reused else "native MOAR optimization including candidate evaluation; selected full-data execution; separate input loading")}
    save(folder / "timing.json", timing)
    return {"seconds": seconds, "load_seconds": load_seconds, "timing": timing, "result_rows": len(payload["rows"]),
            "optimizer": "saved MOAR selected plan" if reused else "native MOAR MCTS",
            "optimizer_exercised": not reused, "plan_reused": reused, "moar_iterations": iterations,
            "native_optimizer_scope": "complete DocETL pipeline with explicit ordinary SQL tools",
            "native_plan_files": ["selected-pipeline.yaml", "plan-reuse-input.json" if reused else "moar-search-summary.json"],
            **({"pre_aggregate": pre} if pre else {})}
