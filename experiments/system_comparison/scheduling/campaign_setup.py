"""Execute frozen comparison tasks with isolated workers and live reports."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import threading
import time
from pathlib import Path

from experiments.core.io import digest, save
from experiments.core.paths import (
    ROOT,
    experiment_sources,
    load_translated_manifest,
    translated_manifest_digest,
)
from experiments.core.resources import (
    BATCH_SIZE,
    CONCURRENCY,
    job_resources,
)
from experiments.system_comparison.data.inputs import (
    TABLES,
    freeze_tables,
    output_schema,
)
from experiments.system_comparison.reporting.monitor import render

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
    queries = load_translated_manifest()
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
        "query_manifest_sha256": translated_manifest_digest(),
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
    if translated_manifest_digest() != config["query_manifest_sha256"]:
        raise ValueError("Query manifest changed during preparation")
    manifest = load_translated_manifest()
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
        meta_rel = record.get("metadata_file")
        if meta_rel:
            meta_path = ROOT / "benchmarks/queries" / meta_rel
            if meta_path.exists():
                shutil.copyfile(meta_path, target.with_suffix(".json"))
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
