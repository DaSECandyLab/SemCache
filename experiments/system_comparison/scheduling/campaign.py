"""Execute frozen comparison tasks with isolated workers and live reports."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import signal
import threading
import time
import traceback
from pathlib import Path

from experiments.core.io import digest, save
from experiments.core.models import model_snapshot
from experiments.core.paths import ROOT, experiment_sources
from experiments.core.process import process
from experiments.core.resources import (
    BATCH_SIZE,
    CONCURRENCY,
    configure_open_file_limit,
    job_resources,
    resource_snapshot,
)
from experiments.system_comparison.data.inputs import (
    TABLES,
    freeze_tables,
    output_schema,
)
from experiments.system_comparison.reporting.monitor import render
from experiments.system_comparison.transport.gateway import (
    PoolGateway,
    flush_model_caches,
)

CONTRACT = "selected19-common-numeric-v4"
REPORT_LOCK = threading.Lock()
MODEL_PARAMETER_WEIGHTS = {"semantic-small": 1.7, "semantic-e2b": 2.0, "semantic-e4b": 4.0, "semantic-large": 12.0}


def delivery_errors_invalidate_result(system, result):
    """A failed MOAR candidate need not invalidate a verified final plan."""
    planning = result.get("planning_response_delivery_errors", 0) if system == "moar" else 0
    return result.get("response_delivery_errors", 0) > planning


def render_progress(run):
    # Concurrent model lanes serialize shared report writes.
    with REPORT_LOCK:
        return render(run)


def execution_fingerprints():
    sources = experiment_sources(ROOT)
    sources += [
        ROOT / name
        for name in (
            "systems/abacus/validate.py",
            "systems/abacus/formal.py",
            "systems/abacus/model-metrics.json",
            "systems/nirvana/formal.py",
            "systems/plop/duckdb-local",
            "systems/sema/vendor/sema-paper/Sema",
        )
    ]
    for relative in (
        "systems/abacus/vendor/abacus-upstream/src/palimpzest",
        "systems/nirvana/vendor/nirvana/nirvana",
        "systems/lotus/lotus-upstream/lotus",
        "systems/moar/vendor/docetl-upstream/docetl",
    ):
        sources.extend((ROOT / relative).rglob("*.py"))
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(set(sources))}


def native_gateway_port(binary=None):
    values = set(
        re.findall(
            rb"http://127\.0\.0\.1:(\d+)/v1/responses",
            Path(binary or ROOT / "systems/plop/duckdb-local").read_bytes(),
        )
    )
    if len(values) != 1:
        raise ValueError("PLOP must have one explicit local transport URL")
    return int(next(iter(values)))


def cache_models_for_variant(variant, models):
    if variant["system"] in {"plop", "sema"}:
        allowed = variant.get("allowed", [])
        selected = [model for model in models if model["alias"] in allowed]
        if len(allowed) != 1 or len(selected) != 1:
            raise ValueError(
                "A single-model configuration must own exactly one model cache"
            )
        return selected
    if "allowed" in variant:
        selected = [model for model in models if model["alias"] in variant["allowed"]]
        if not selected or len(selected) != len(set(variant["allowed"])):
            raise ValueError("Configuration references an unavailable execution model")
        return selected
    return models


def cleanup_owned_processes(folder):
    """Clean only native descendants whose cwd remains inside this attempt."""
    folder = Path(folder).resolve()
    owned = []
    for path in Path("/proc").glob("[0-9]*/cwd"):
        try:
            if path.resolve(strict=True).is_relative_to(folder):
                owned.append(int(path.parent.name))
        except (OSError, RuntimeError):
            pass
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in owned:
            try:
                if Path(f"/proc/{pid}/cwd").resolve(strict=True).is_relative_to(folder):
                    os.kill(pid, sig)
            except (OSError, RuntimeError):
                pass
        if owned and sig == signal.SIGTERM:
            time.sleep(0.2)
    return owned


def variants(preset="legacy"):
    if preset == "three-model-quality-grid":
        result = [
            {"name": f"{system}-{label}", "system": system, "allowed": [alias]}
            for system in ("plop", "sema")
            for label, alias in (("e2b", "semantic-e2b"), ("e4b", "semantic-e4b"), ("12b", "semantic-large"))
        ]
        return result + [
            {"name": "nirvana-pool", "system": "nirvana"},
            {"name": "lotus-two-stage", "system": "lotus", "cascade": "helper_main",
             "allowed": ["semantic-e2b", "semantic-large"]},
            *[{"name": f"abacus-MinTime-q{percent}", "system": "abacus", "policy": "MinTime", "quality_threshold": percent / 100}
              for percent in (20, 50, 80)],
        ]
    if preset != "legacy":
        raise ValueError("Unknown experiment preset")
    result = []
    for system in ("plop", "sema"):
        for name in ("small", "e2b", "e4b", "large"):
            result.append(
                {
                    "name": system + "-" + name,
                    "system": system,
                    "allowed": ["semantic-" + name],
                }
            )
    result += [
        {"name": "abacus-MaxQuality", "system": "abacus", "policy": "MaxQuality"},
        {"name": "abacus-MinTime", "system": "abacus", "policy": "MinTime"},
        {"name": "nirvana-pool", "system": "nirvana"},
        {
            "name": "lotus-four-stage",
            "system": "lotus",
            "cascade": "experimental_four_model_extension",
        },
    ]
    return result


def initialize(
    run,
    selected_variant=None,
    *,
    workload="baseline19",
    exclude_variants=(),
    batch_size=None,
    abacus_last=False,
    preset="legacy",
    system_batch_sizes=None,
    gateway_request_limit=None,
):
    if system_batch_sizes is not None:
        if set(system_batch_sizes) - {"plop", "sema", "nirvana", "lotus", "abacus"}:
            raise ValueError("Unknown system batch-size override")
        for size in system_batch_sizes.values():
            job_resources(size)
    if gateway_request_limit is not None and (
        type(gateway_request_limit) is not int or not 1 <= gateway_request_limit <= 1024
    ):
        raise ValueError("Gateway request limit must be an integer in [1,1024]")
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / "campaign.json").exists():
        previous = json.loads((run / "campaign.json").read_text())
        if previous.get("preset", "legacy") != preset:
            raise ValueError("Preset differs from the frozen campaign")
        frozen_batch = previous.get("execution_resources", {}).get(
            "batch_size", BATCH_SIZE
        )
        if batch_size is not None and batch_size != frozen_batch:
            raise ValueError(
                "Batch size differs from the frozen campaign; use a new directory"
            )
        if abacus_last and not previous.get("abacus_last"):
            raise ValueError(
                "Pool ordering differs from the frozen campaign; use a new directory"
            )
        if system_batch_sizes is not None and previous.get("system_batch_sizes", {}) != system_batch_sizes:
            raise ValueError("System batch sizes differ from the frozen campaign")
        if gateway_request_limit is not None and previous.get("gateway_request_limit", 32) != gateway_request_limit:
            raise ValueError("Gateway request limit differs from the frozen campaign")
        return previous
    resources = job_resources(BATCH_SIZE if batch_size is None else batch_size)
    manifest = ROOT / "benchmarks/queries/translated/manifest.json"
    queries = json.loads(manifest.read_text())
    if queries["contract"] != CONTRACT:
        raise ValueError("Only the current v4 common workload is supported")
    order = (
        [f"estate/{q}" for q in ("Q5", "Q9", "Q10", "Q11", "Q12")]
        + [f"movies/{q}" for q in ("Q9", "Q10", "Q11", "Q12")]
        + [f"appstore/{q}" for q in ("Q7", "Q8", "Q9")]
        + [f"tpch/{q}" for q in ("Q16", "Q17", "Q25", "Q26", "Q27", "Q28", "Q30")]
    )
    if workload not in {"baseline19", "all", "all20", "all28"}:
        raise ValueError(f"Unknown query workload: {workload}")
    if workload != "baseline19":
        available = [
            record["id"]
            for record in queries["records"]
            if record["system"] == "semcache"
        ]
        expected = queries["query_count_per_system"]["semcache"]
        if len(available) != expected or len(set(available)) != expected:
            raise ValueError("The complete workload differs from its manifest query count")
        datasets = ("estate", "movies", "movies_sema", "appstore", "tpch")
        order = sorted(
            available,
            key=lambda key: (
                datasets.index(key.split("/")[0]),
                int(key.rsplit("/Q", 1)[1]),
            ),
        )
    excluded = set(exclude_variants)
    unknown = excluded - {item["name"] for item in variants(preset)}
    if unknown:
        raise ValueError(f"Unknown excluded configurations: {sorted(unknown)}")
    selected = [item for item in variants(preset) if item["name"] not in excluded]
    if selected_variant is not None:
        selected = [item for item in selected if item["name"] == selected_variant]
    if not selected:
        raise ValueError("No experiment configurations selected")
    if abacus_last:
        selected.sort(key=lambda item: item["system"] == "abacus")
    tasks = []
    for variant in selected:
        for query_id in order:
            tasks.append(
                {
                    "task_id": variant["name"] + "/" + query_id,
                    "variant": variant["name"],
                    "system": variant["system"],
                    "query_id": query_id,
                    "folder": "tasks/"
                    + variant["name"]
                    + "/"
                    + query_id
                    + "/attempt-1",
                }
            )
    config = {
        "contract": CONTRACT,
        "preset": preset,
        "variants": selected,
        "query_ids": order,
        "query_workload": workload,
        "excluded_variants": sorted(excluded),
        "query_contracts": sorted(
            {
                record["contract"]
                for record in queries["records"]
                if record["system"] == "semcache" and record["id"] in order
            }
        ),
        "tasks": tasks,
        "scheduled_attempts": len(tasks),
        "query_manifest_sha256": digest(manifest),
        "created_unix": time.time(),
        "query_timeout_seconds": 1800,
        "request_timeout_seconds": 120,
        "sema_request_timeout_seconds": 600,
        "gateway_request_limit": gateway_request_limit if gateway_request_limit is not None else 32,
        "progress_interval_seconds": 10,
        "semcache_excluded": True,
        "execution_resources": resources,
        "system_batch_sizes": dict(system_batch_sizes or {}),
        "abacus_last": abacus_last,
        "native_mechanism_profile": "paper-mechanisms-v1",
        "timing_schema": "native-phase-timing-v1",
        "lotus_cascade": "User-authorized multi-model extension; native LOTUS filtering and threshold learning; maps use 12B",
        "input_mode": "full frozen tables, no truncation or positive-row selection",
        "input_loading": "database_memory",
        "reference_api_calls": 0,
        "accuracy_scoring": "Not performed in this run; outputs preserved for the independent common protocol",
    }
    save(run / "campaign.json", config)
    save(run / "status.json", {"phase": "preparing", "updated_unix": time.time()})
    render(run)
    return config


def prepare(run, config):
    import duckdb

    folder = run / "inputs"
    if (folder / "manifest.json").exists():
        frozen = json.loads((folder / "manifest.json").read_text())
        if frozen.get("input_loading") == "database_memory":
            for datasets in frozen["databases"].values():
                for item in datasets.values():
                    if digest(item["path"]) != item["sha256"]:
                        raise ValueError("Prepared database changed")
            return frozen
        for table in frozen["tables"].values():
            if (
                digest(table["path"]) != table["sha256"]
                or digest(table["csv"]) != table["csv_sha256"]
            ):
                raise ValueError("Frozen inputs were modified")
        return frozen
    source_manifest = ROOT / "benchmarks/queries/translated/manifest.json"
    if digest(source_manifest) != config["query_manifest_sha256"]:
        raise ValueError("Query manifest changed during preparation")
    manifest = json.loads(source_manifest.read_text())
    save(run / "query-manifest.json", manifest)
    memory = config.get("input_loading") == "database_memory"
    if not memory:
        tables = freeze_tables(ROOT / "experiments/inputs/inputs.duckdb", folder)
    jobs = {}
    query_files = {}
    for record in manifest["records"]:
        if record["system"] == "semcache":
            continue
        path = ROOT / "benchmarks/queries" / record["query_file"]
        if digest(path) != record["sha256"]:
            raise ValueError("Generated query hash mismatch: " + str(path))
        relative = (
            Path("queries")
            / record["system"]
            / Path(record["id"]).parent
            / (Path(record["id"]).name + path.suffix)
        )
        target = run / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        shutil.copyfile(
            ROOT / "benchmarks/queries" / record["metadata_file"],
            target.with_suffix(".json"),
        )
        query_files[record["system"] + "/" + record["id"]] = {
            "path": str(target),
            "sha256": record["sha256"],
            "source": record["query_file"],
        }
    if memory:
        from experiments.system_comparison.data.memory import prepare_campaign

        return prepare_campaign(run, config, manifest, query_files, root=ROOT)
    with duckdb.connect(str(folder / "inputs.duckdb"), read_only=True) as con:
        for query_id in config["query_ids"]:
            generic = next(
                r
                for r in manifest["records"]
                if r["system"] == "semcache" and r["id"] == query_id
            )
            # Read the generic declaration only to bind a model-free result schema.
            # The SemCache engine/optimizer is never imported or executed.
            sql = (ROOT / "benchmarks/queries" / generic["query_file"]).read_text()
            dataset, query = query_id.split("/")
            jobs[query_id] = {
                "id": query_id,
                "dataset": dataset,
                "query": query,
                "input_database": str(folder / "inputs.duckdb"),
                "inputs": {name: tables[name] for name in TABLES[dataset]},
                "output_schema": output_schema(con, sql),
                "workload_contract": CONTRACT,
            }
    frozen = {
        "tables": tables,
        "jobs": jobs,
        "query_files": query_files,
        "source_database_sha256": digest(ROOT / "experiments/inputs/inputs.duckdb"),
        "frozen_database_sha256": digest(folder / "inputs.duckdb"),
    }
    save(folder / "manifest.json", frozen)
    return frozen


def worker_command(system, job_path, *, cpu_set=None):
    interpreter = ROOT / (
        "systems/abacus/vendor/abacus-upstream/.venv/bin/python"
        if system == "abacus"
        else f"systems/{system}/.venv/bin/python"
        if system in {"nirvana", "lotus", "moar"}
        else "SemCache/.venv-optimizer/bin/python"
    )
    paths = [ROOT, ROOT / f"systems/{system}"]
    if system == "abacus":
        paths.append(ROOT / "systems/abacus/vendor/abacus-upstream/src")
    if system == "nirvana":
        paths.append(ROOT / "systems/nirvana/vendor/nirvana")
    if system == "lotus":
        paths.append(ROOT / "systems/lotus/lotus-upstream")
    if system == "moar":
        paths.insert(1, ROOT / "systems/moar/vendor/docetl-upstream")
    if system != "lotus":
        paths += [
            ROOT / "SemCache/.venv-optimizer/lib/python3.12/site-packages",
            ROOT / "sglang/.venv/lib/python3.12/site-packages",
        ]
    env = {
        **os.environ,
        "PYTHONPATH": ":".join(map(str, paths)),
        "OPENAI_API_KEY": "EMPTY",
        "VALIDATION_API_KEY": "EMPTY",
        "FORMAL_API_KEY": "EMPTY",
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": str(CONCURRENCY),
        "MKL_NUM_THREADS": str(CONCURRENCY),
        "OPENBLAS_NUM_THREADS": str(CONCURRENCY),
        "NUMEXPR_NUM_THREADS": str(CONCURRENCY),
        "PZ_MODEL_METRICS_FILE": str(ROOT / "systems/abacus/model-metrics.json"),
        "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        "LITELLM_LOG": "ERROR",
    }
    command = [
        str(interpreter),
        "-u",
        "-m",
        "experiments.system_comparison.execution.worker",
        "--job",
        str(job_path),
    ]
    if cpu_set is not None:
        command = ["taskset", "-c", cpu_set, *command]
    return command, env


def snapshot_environment(run, models):
    save(run / "models.json", models)
    # Keep only public runtime settings, not credentials or full server args.
    import urllib.request

    server_settings = []
    for model in models:
        base = model["endpoint"].removesuffix("/v1")
        try:
            with urllib.request.urlopen(
                base + "/get_server_info", timeout=10
            ) as response:
                info = json.load(response)
            selected = {
                key: info.get(key)
                for key in (
                    "served_model_name",
                    "max_running_requests",
                    "max_queued_requests",
                    "context_length",
                    "max_total_tokens",
                )
            }
            server_settings.append(
                {"alias": model["alias"], "endpoint": base, **selected}
            )
        except (OSError, ValueError) as error:
            server_settings.append(
                {
                    "alias": model["alias"],
                    "endpoint": base,
                    "snapshot_error": str(error),
                }
            )
    save(run / "server-settings.json", server_settings)


def execute(
    run,
    config,
    frozen,
    *,
    persist_campaign=True,
    final_phase="finished",
    cpu_set=None,
    gateway_port=None,
    status_file="status.json",
    models=None,
    write_shared_metadata=True,
    source_hashes=None,
    reset_variant_caches=True,
    abacus_sampling=True,
):
    from contextlib import ExitStack
    import shutil
    runtime_limits = configure_open_file_limit()
    fingerprints = execution_fingerprints() if source_hashes is None else source_hashes
    models = model_snapshot() if models is None else models
    if write_shared_metadata:
        snapshot_environment(run, models)
    variants_by_name = {v["name"]: v for v in config["variants"]}
    batch_size = config.get("execution_resources", {}).get("batch_size", BATCH_SIZE)
    current_variant = None
    config["gateway_port"] = (
        native_gateway_port() if gateway_port is None else gateway_port
    )
    # Allow queued Sema exploration requests to wait without causing duplicate upstream retries.
    task_systems = {task.get("system") for task in config.get("tasks", [])}
    rewrite_agent = None
    if "moar" in task_systems and config.get("moar_agent_config"):
        from experiments.system_comparison.backends.moar.agent import agent_model

        rewrite_agent = agent_model(config["moar_agent_config"])
    gateway_timeout = (
        config.get("sema_request_timeout_seconds", 600)
        if task_systems == {"sema"}
        else config["request_timeout_seconds"]
    )
    if persist_campaign:
        save(run / "campaign.json", config)
    gateway_limit = config.get("gateway_request_limit", 32)
    if len(task_systems) == 1:
        gateway_limit = config.get("system_gateway_limits", {}).get(next(iter(task_systems)), gateway_limit)
    with ExitStack() as cache_stack, PoolGateway(
        [*models, rewrite_agent] if rewrite_agent else models,
        port=config["gateway_port"],
        request_timeout=gateway_timeout,
        upstream_concurrency_limit=gateway_limit,
    ) as gateway:
        for task in config["tasks"]:
            control = {}
            control_path = run / "control.json"
            if control_path.exists():
                try:
                    control = json.loads(control_path.read_text())
                except (OSError, ValueError):
                    control = {}
            skipped_tasks = set(control.get("skipped_tasks", []))
            if task["task_id"] in skipped_tasks:
                folder = run / task["folder"]
                folder.mkdir(parents=True, exist_ok=True)
                save(
                    folder / "summary.json",
                    {
                        **task,
                        "status": "skipped",
                        "reason": control.get(
                            "skip_reason", "Skipped by active campaign control"
                        ),
                        "seconds": 0.0,
                        "parent_finalized": True,
                        "accuracy": None,
                        "model_requests": 0,
                        "execution_resources": {
                            **job_resources(
                                config.get("system_batch_sizes", {}).get(
                                    task["system"], batch_size
                                )
                            ),
                            "gateway_concurrency_limit": config.get(
                                "gateway_request_limit", 32
                            ),
                        },
                        "cache_policy": {
                            "reset_at_variant_start": False,
                            "preserved_by_control": True,
                        },
                    },
                )
                render_progress(run)
                continue
            folder = run / task["folder"]
            if (folder / "summary.json").exists() and json.loads(
                (folder / "summary.json").read_text()
            ).get("parent_finalized"):
                continue
            folder.mkdir(parents=True, exist_ok=True)
            variant = variants_by_name[task["variant"]]
            task_batch_size = config.get("system_batch_sizes", {}).get(
                task["system"], batch_size
            )
            if current_variant != task["variant"]:
                preserve_cache = bool(control.get("preserve_cache", False))
                if reset_variant_caches and not preserve_cache:
                    save(
                        run / status_file,
                        {
                            "phase": "clearing_variant_cache",
                            "variant": task["variant"],
                            "updated_unix": time.time(),
                        },
                    )
                    cache_models = cache_models_for_variant(variant, models)
                    save(
                        run / "cache-resets" / (task["variant"] + ".json"),
                        {
                            "variant": task["variant"],
                            "models": [m["alias"] for m in cache_models],
                            "services": flush_model_caches(
                                cache_models, idle_timeout=20
                            ),
                            "reset_unix": time.time(),
                        },
                    )
                else:
                    save(
                        run / "cache-reuse" / (task["variant"] + ".json"),
                        {
                            "variant": task["variant"],
                            "reset_performed": False,
                            "reason": "User explicitly requested retaining model caches for this rerun",
                            "tasks": [
                                t["task_id"]
                                for t in config["tasks"]
                                if t["variant"] == task["variant"]
                            ],
                            "recorded_unix": time.time(),
                        },
                    )
                prompt_cache = None
                if config.get("system_prompt_cache", {}).get(task["system"], config.get("prompt_cache_enabled", False)):
                    from experiments.system_comparison.transport.prompt_cache import PromptCache

                    cache_directory = run / "prompt-cache" / task["variant"]
                    if reset_variant_caches and not preserve_cache and cache_directory.exists():
                        shutil.rmtree(cache_directory)
                    prompt_cache = PromptCache(cache_directory, task["variant"])
                    cache_stack.callback(prompt_cache.close)
                current_variant = task["variant"]
            base = frozen["jobs"][task["query_id"]]
            query = frozen["query_files"][task["system"] + "/" + task["query_id"]]
            if digest(query["path"]) != query["sha256"]:
                raise ValueError("Frozen query was modified")
            allowed = variant.get("allowed", [m["alias"] for m in models])
            if task["system"] == "moar" and rewrite_agent:
                allowed = [*allowed, rewrite_agent["alias"]]
            token = task["task_id"].replace("/", "-")
            aliases = (
                {"gpt-5-mini": allowed[0], "gpt-5-mini-2025-08-07": allowed[0]}
                if task["system"] == "plop"
                else {}
            )
            prompt_protocol = (
                config.get("plop_prompt_protocol") if task["system"] == "plop" else None
            )
            task_gateway_limit = config.get("system_gateway_limits", {}).get(
                task["system"], gateway_limit
            )
            endpoint = gateway.register(
                token,
                folder,
                allowed=allowed,
                aliases=aliases,
                default=task["system"] == "plop",
                prompt_protocol=prompt_protocol,
                response_cache=prompt_cache,
                compact_boolean_aliases=[m["alias"] for m in models] if task["system"] == "moar" else None,
                record_request_stage=task["system"] == "moar",
                upstream_concurrency_limit=task_gateway_limit,
            )
            routed = [
                {**m, "upstream_endpoint": m["endpoint"], "endpoint": endpoint}
                for m in models if m["alias"] in allowed
            ]
            job = {
                **base,
                **variant,
                **task,
                "output": str(folder),
                "query_file": query["path"],
                "query_sha256": query["sha256"],
                "models": routed,
                "model": allowed[0],
                "capture_aggregate_inputs": config.get("capture_aggregate_inputs", False),
                "sql_capture_aggregate_inputs": task["system"] in {"sema", "plop"} and config.get("sql_capture_aggregate_inputs", False),
                "pre_aggregate_schema": config.get("aggregate_input_schemas", {}).get(task["system"] + "/" + task["query_id"]),
                "skip_final_aggregate": task["system"] in {"lotus", "abacus"}
                and config.get("aggregate_input_schemas", {}).get(
                    task["system"] + "/" + task["query_id"]
                )
                is not None,
                **({"quality_threshold": config["abacus_min_quality"]} if task["system"] == "abacus" and "abacus_min_quality" in config else {}),
                "endpoint": endpoint,
                "teacher_model": "semantic-large",
                **({"reference_directory": config["reference_directory"]} if task["system"] == "moar" else {}),
                **({"moar_agent_model": {key: rewrite_agent[key] for key in ("alias", "id", "context_length")},
                    "moar_agent_config": config["moar_agent_config"]} if task["system"] == "moar" and rewrite_agent else {}),
                "timeout": control.get(
                    "query_timeout_seconds", config["query_timeout_seconds"]
                ),
                "request_timeout": gateway_timeout,
                **({"client_request_timeout": min(
                    control.get("query_timeout_seconds", config["query_timeout_seconds"]),
                    gateway_timeout * (gateway.upstream_retries + 1) + 30,
                )} if task["system"] == "moar" else {}),
                **job_resources(task_batch_size, request_concurrency=config.get("system_concurrency", {}).get(task["system"], CONCURRENCY)),
                **({"abacus_queue_batch_size": config.get("abacus_queue_batch_size", 32)} if task["system"] == "abacus" else {}),
                **({"abacus_prompt_protocol": "short-output-v1"} if task["system"] == "abacus" else {}),
                "gateway_concurrency_limit": task_gateway_limit,
                # Reserve additional output tokens for Abacus reasoning so the required structured answer is not truncated.
                "max_tokens": 2048 if task["system"] == "abacus" else 1024,
                "optimizer": {
                    "do_logical_optimization": True,
                    "do_physical_optimization": True,
                    "available_models": [m["alias"] for m in models if m["alias"] in allowed],
                    "max_rounds": 5,
                    "sample_size": 5,
                },
                "native_mechanisms": task["system"] in {"abacus", "nirvana", "lotus"},
                "native_mechanism_profile": (
                    "native-no-mab-no-sampling-v1"
                    if task["system"] == "abacus" and not abacus_sampling
                    else "paper-mechanisms-v1"
                ),
                "optimization_cost_unit": "parameter_weighted_tokens_per_1000; not USD",
                "model_costs": {
                    m["alias"]: dict.fromkeys(("Input", "Output", "Cache"), MODEL_PARAMETER_WEIGHTS[m["alias"]])
                    for m in models
                },
                "abacus_optimizer": {
                    "sample_budget": 32,
                    "k": 4,
                    "j": 4,
                    "judge_model": "semantic-large",
                    "seed": 42,
                    "sampling_enabled": abacus_sampling,
                    "max_search_seconds": 100,
                },
                "embedding": {
                    "backend": "sentence_transformers",
                    "model": str(ROOT / "sglang/models/all-MiniLM-L6-v2"),
                    "device": "cpu",
                },
                "cache_policy": {
                    "prompt_cache_enabled": prompt_cache is not None,
                    "prompt_cache_scope": "same configuration across queries; exact model and full request",
                    "reset_at_variant_start": reset_variant_caches
                    and not bool(control.get("preserve_cache", False)),
                    "preserved_by_control": bool(control.get("preserve_cache", False)),
                },
                "cpu_set": cpu_set,
                "gateway_port": config["gateway_port"],
            }
            if base.get("input_loading") == "database_memory":
                from experiments.system_comparison.data.memory import bind_database

                job = bind_database(job)
            else:
                job["input_database_sha256"] = frozen["frozen_database_sha256"]
            if prompt_protocol:
                job["prompt_protocol"] = prompt_protocol
            save(folder / "job.json", job)
            resources = {
                **resource_snapshot(
                    task["system"], task_batch_size,
                    gateway_request_limit=task_gateway_limit,
                    request_concurrency=job["native_concurrency"],
                    abacus_queue_batch_size=job.get("abacus_queue_batch_size"),
                ),
                **runtime_limits,
                "cpu_set": cpu_set,
                "gateway_port": config["gateway_port"],
            }
            if prompt_protocol:
                resources["prompt_protocol"] = prompt_protocol
            save(folder / "resources.json", resources)
            save(folder / "execution-source-hashes.json", fingerprints)
            command, env = worker_command(
                task["system"], folder / "job.json", cpu_set=cpu_set
            )
            save(
                folder / "launch.json",
                {"command": command, "started_unix": time.time()},
            )
            print("START", task["task_id"], flush=True)

            def heartbeat(elapsed, pid):
                save(
                    run / status_file,
                    {
                        "phase": "running",
                        **task,
                        "elapsed_seconds": elapsed,
                        "child_pid": pid,
                        "controller_pid": os.getpid(),
                        "cpu_set": cpu_set,
                        "gateway_port": config["gateway_port"],
                        "updated_unix": time.time(),
                        **gateway.summary(folder),
                    },
                )

            try:
                stats = process(
                    command,
                    folder,
                    job["timeout"],
                    env=env,
                    heartbeat=heartbeat,
                )
                native = (
                    json.loads((folder / "summary.json").read_text())
                    if (folder / "summary.json").exists()
                    else {}
                )
                result = {**stats, **native, "wall_seconds": stats["seconds"]}
                if stats["status"] == "timeout":
                    result.update(
                        status="timeout",
                        reason=f"Whole-query {job['timeout']} second deadline; worker process group terminated",
                    )
                elif not native:
                    result.update(
                        status="failed",
                        reason=(folder / "stderr.log").read_text()[-5000:],
                    )
            except Exception as exc:
                (folder / "controller-error.txt").write_text(traceback.format_exc())
                result = {
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "seconds": 0,
                }
            finally:
                gateway.close_run(token)
                save(
                    folder / "process-cleanup.json",
                    {"terminated_attempt_owned_pids": cleanup_owned_processes(folder)},
                )
            deadline = time.monotonic() + config["request_timeout_seconds"] + 10
            while gateway.pending(token) and time.monotonic() < deadline:
                heartbeat(result.get("wall_seconds", 0), None)
                time.sleep(1)
            result.update(gateway.summary(folder))
            if result.get("status") == "completed" and result["model_errors"]:
                result.update(
                    status="partial",
                    reason="Transport errors occurred; inspect complete gateway request log",
                )
            # Failed planning candidates do not invalidate a subsequently
            # selected and fully validated MOAR execution. Preserve their
            # error counts; execution and unknown-stage failures remain fatal.
            if result.get("status") == "completed" and delivery_errors_invalidate_result(
                task["system"], result
            ):
                result.update(
                    status="partial",
                    reason="Native client disconnected during response delivery; inspect gateway delivery_error events",
                )
            if (
                result.get("status") == "completed"
                and result["model_requests"] == 0
                and base["dataset"] != "tpch"
            ):
                result.update(
                    status="failed",
                    reason="Nonempty text workload returned without any observed model request; inference was not verified",
                )
            result.update(
                **task,
                parent_finalized=True,
                workload_contract=job.get("workload_contract", CONTRACT),
                accuracy=None,
                execution_resources=resources,
                cache_policy=job["cache_policy"],
            )
            save(folder / "summary.json", result)
            render_progress(run)
            print(
                "DONE",
                task["task_id"],
                result["status"],
                result.get("seconds"),
                flush=True,
            )
    save(
        run / status_file,
        {
            "phase": final_phase,
            "updated_unix": time.time(),
            "controller_pid": os.getpid(),
        },
    )
    render_progress(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=[item["name"] for item in variants()],
        help="Run only one configuration in a fresh result directory",
    )
    parser.add_argument(
        "--workload", choices=["baseline19", "all", "all20", "all28"], default="baseline19"
    )
    parser.add_argument(
        "--exclude-variants",
        nargs="+",
        choices=[item["name"] for item in variants()],
        default=[],
    )
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--abacus-last", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    config = initialize(
        run,
        args.variant,
        workload=args.workload,
        exclude_variants=args.exclude_variants,
        batch_size=args.batch_size,
        abacus_last=args.abacus_last,
    )
    if args.initialize_only:
        return
    with (run / "campaign.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(
            run / "controller-launch.json",
            {"pid": os.getpid(), "started_unix": time.time()},
        )
        try:
            if not args.prepare_only:
                fingerprints = execution_fingerprints()
                previous = run / "execution-source-hashes.json"
                if (
                    previous.exists()
                    and json.loads(previous.read_text()) != fingerprints
                ):
                    raise ValueError(
                        "Execution code changed after campaign start; preserve this campaign and use a new run directory"
                    )
                save(previous, fingerprints)
            frozen = prepare(run, config)
            if not args.prepare_only:
                execute(run, config, frozen)
        except BaseException as exc:
            status = (
                json.loads((run / "status.json").read_text())
                if (run / "status.json").exists()
                else {}
            )
            if status.get("folder"):
                cleanup_owned_processes(run / status["folder"])
            (run / "controller-error.txt").write_text(traceback.format_exc())
            save(
                run / "status.json",
                {
                    "phase": "controller_failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "updated_unix": time.time(),
                },
            )
            render(run)
            raise


if __name__ == "__main__":
    main()
