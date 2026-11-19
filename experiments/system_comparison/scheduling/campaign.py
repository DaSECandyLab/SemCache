"""Execute frozen comparison tasks with isolated workers and live reports."""

from __future__ import annotations

import json
import os
import time
import traceback

from experiments.core.io import digest, save
from experiments.core.models import model_snapshot as model_snapshot
from experiments.core.paths import ROOT
from experiments.core.process import process as process
from experiments.core.resources import (
    configure_open_file_limit as configure_open_file_limit,
)
from experiments.core.resources import (
    resource_snapshot as resource_snapshot,
)
from experiments.system_comparison.transport.gateway import (
    PoolGateway,
    flush_model_caches,
)

from .campaign_setup import (
    BATCH_SIZE as BATCH_SIZE,
)
from .campaign_setup import (
    CONCURRENCY as CONCURRENCY,
)
from .campaign_setup import (
    CONTRACT as CONTRACT,
)
from .campaign_setup import (
    MODEL_PARAMETER_WEIGHTS as MODEL_PARAMETER_WEIGHTS,
)
from .campaign_setup import (
    REPORT_LOCK as REPORT_LOCK,
)
from .campaign_setup import (
    cache_models_for_variant as cache_models_for_variant,
)
from .campaign_setup import (
    cleanup_owned_processes as cleanup_owned_processes,
)
from .campaign_setup import (
    delivery_errors_invalidate_result as delivery_errors_invalidate_result,
)
from .campaign_setup import (
    execution_fingerprints as execution_fingerprints,
)
from .campaign_setup import (
    freeze_tables as freeze_tables,
)
from .campaign_setup import (
    initialize as initialize,
)
from .campaign_setup import (
    job_resources as job_resources,
)
from .campaign_setup import (
    load_translated_manifest as load_translated_manifest,
)
from .campaign_setup import (
    native_gateway_port as native_gateway_port,
)
from .campaign_setup import (
    output_schema as output_schema,
)
from .campaign_setup import (
    prepare as prepare,
)
from .campaign_setup import (
    render_progress as render_progress,
)
from .campaign_setup import (
    snapshot_environment as snapshot_environment,
)
from .campaign_setup import (
    translated_manifest_digest as translated_manifest_digest,
)
from .campaign_setup import (
    variants as variants,
)
from .campaign_setup import (
    worker_command as worker_command,
)


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
    import shutil
    from contextlib import ExitStack
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
                    from experiments.system_comparison.transport.prompt_cache import (
                        PromptCache,
                    )

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
    from .campaign_cli import main as _main

    return _main()


if __name__ == "__main__":
    main()
