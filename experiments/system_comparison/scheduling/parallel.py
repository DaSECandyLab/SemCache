"""Four independent single-model lanes, then the four model-pool configurations.

A lane runs all selected PLOP queries before all selected Sema queries for its model. It owns
one gateway port and 16 logical CPUs. No other lane can flush that model cache.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments.core.io import digest, save
from experiments.core.models import model_snapshot
from experiments.core.paths import ROOT
from experiments.core.resources import CONCURRENCY
from experiments.system_comparison.scheduling.campaign import (
    execute,
    execution_fingerprints,
    initialize,
    native_gateway_port,
    prepare,
    render_progress,
    snapshot_environment,
    variants,
)


def format_cpu_set(cpus):
    values = sorted(set(cpus))
    parts = []
    for value in values:
        if parts and parts[-1][-1] + 1 == value:
            parts[-1].append(value)
        else:
            parts.append([value])
    return ",".join(
        str(part[0]) if len(part) == 1 else f"{part[0]}-{part[-1]}" for part in parts
    )


def allocate_cpu_sets(count=4, width=CONCURRENCY, topology=None):
    """Keep SMT siblings within a lane; do not share a physical core across lanes."""
    if topology is None:
        topology = {}
        for cpu in sorted(os.sched_getaffinity(0)):
            path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            topology[cpu] = (
                int((path / "physical_package_id").read_text()),
                int((path / "core_id").read_text()),
            )
    cores = defaultdict(list)
    for cpu, key in sorted(topology.items()):
        cores[key].append(cpu)
    packages = defaultdict(list)
    for (package, core), cpus in sorted(cores.items()):
        packages[package].append(cpus)
    sockets = sorted(packages)
    if len(topology) < count * width:
        raise ValueError(
            f"Need {count * width} available logical CPUs for disjoint tasksets"
        )
    # Leave headroom at both ends of each socket where capacity allows it.
    for index, socket in enumerate(sockets):
        assigned = sum(i % len(sockets) == index for i in range(count))
        spare = max(0, sum(map(len, packages[socket])) - assigned * width)
        reserve = spare // 2
        used = 0
        while packages[socket] and used + len(packages[socket][0]) <= reserve:
            used += len(packages[socket].pop(0))
    result = []
    for index in range(count):
        selected = []
        order = sockets[index % len(sockets) :] + sockets[: index % len(sockets)]
        for socket in order:
            while packages[socket] and len(selected) < width:
                group = packages[socket].pop(0)
                if len(selected) + len(group) > width:
                    raise ValueError(
                        "CPU width must fit complete available SMT core groups"
                    )
                selected.extend(group)
            if len(selected) == width:
                break
        if len(selected) != width:
            raise ValueError(
                "Insufficient complete physical cores for independent tasksets"
            )
        result.append(sorted(selected))
    return result


def build_lanes(config, models, cpu_sets):
    lanes = []
    for index, model in enumerate(models):
        selected = [
            v
            for v in config["variants"]
            if v["system"] in {"plop", "sema"} and v.get("allowed") == [model["alias"]]
        ]
        selected.sort(key=lambda v: 0 if v["system"] == "plop" else 1)
        if not selected:
            continue
        lanes.append(
            {
                "model": model["alias"],
                "model_id": model["id"],
                "variants": [v["name"] for v in selected],
                "cpu_ids": cpu_sets[index],
                "cpu_set": format_cpu_set(cpu_sets[index]),
                "gateway_port": 8899 + index,
                "status_file": f"lanes/{model['alias']}.json",
            }
        )
    return lanes


def verify_transports(lanes):
    original = (ROOT / "systems/plop/duckdb-local").read_bytes()
    base = b"http://127.0.0.1:8899/v1/responses"
    records = []
    for lane in lanes:
        port = lane["gateway_port"]
        path = (
            ROOT
            / "systems/plop"
            / ("duckdb-local" if port == 8899 else f"duckdb-local-{port}")
        )
        raw = path.read_bytes()
        url = f"http://127.0.0.1:{port}/v1/responses".encode()
        if native_gateway_port(path) != port or raw.replace(url, base) != original:
            raise ValueError(f"PLOP lane binary differs beyond its HTTP URL: {path}")
        records.append(
            {
                "port": port,
                "binary": str(path),
                "sha256": digest(path),
                "same_optimizer_code": True,
                "transport_url": url.decode(),
            }
        )
    return records


def run_model_lanes(lanes, run_variant):
    """Serial within each lane; parallel only across distinct model identities."""

    def lane_work(lane):
        for variant in lane["variants"]:
            run_variant(lane, variant)

    with ThreadPoolExecutor(
        max_workers=len(lanes), thread_name_prefix="model-lane"
    ) as pool:
        futures = [pool.submit(lane_work, lane) for lane in lanes]
        for future in as_completed(futures):
            future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--preset", choices=["legacy", "three-model-quality-grid"], default="legacy")
    parser.add_argument("--capture-aggregate-inputs", action="store_true", help="Capture Python host aggregate inputs without changing SQL optimizer boundaries")
    parser.add_argument("--references", type=Path, help="Reference schemas for empty aggregate inputs")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--sema-batch-size", type=int,
                        help="Maximum native Sema batch size, overriding the common batch size")
    parser.add_argument("--gateway-request-limit", type=int,
                        help="Total concurrent upstream HTTP requests per configuration across all models")
    parser.add_argument("--abacus-last", action="store_true")
    parser.add_argument(
        "--semcache-followup", action="store_true",
        help="Document that a separately scheduled SemCache evaluation follows this campaign",
    )
    parser.add_argument(
        "--settings-from", type=Path,
        help="Reuse comparison cache, concurrency and aggregate-capture settings from a prior campaign",
    )
    parser.add_argument(
        "--workload", choices=["baseline19", "all", "all20", "all28"], default="baseline19"
    )
    parser.add_argument(
        "--exclude-variants",
        nargs="+",
        choices=sorted({item["name"] for preset in ("legacy", "three-model-quality-grid") for item in variants(preset)}),
        default=[],
    )
    args = parser.parse_args()
    run = args.run.resolve()
    config = initialize(
        run,
        workload=args.workload,
        exclude_variants=args.exclude_variants,
        batch_size=args.batch_size,
        abacus_last=args.abacus_last,
        preset=args.preset,
        system_batch_sizes={"sema": args.sema_batch_size} if args.sema_batch_size is not None else None,
        gateway_request_limit=args.gateway_request_limit,
    )
    if args.settings_from:
        source = json.loads(args.settings_from.resolve().read_text())
        if source.get("query_ids") != config.get("query_ids"):
            raise ValueError("Reference campaign has a different query workload")
        if source.get("query_manifest_sha256") != config.get("query_manifest_sha256"):
            raise ValueError("Reference campaign has a different query manifest")
        if source.get("execution_resources", {}).get("batch_size") != config["execution_resources"]["batch_size"]:
            raise ValueError("Batch size differs from the reference campaign")
        if source.get("system_batch_sizes", {}) != config.get("system_batch_sizes", {}):
            raise ValueError("Per-system batch size differs from the reference campaign")
        for key in (
            "system_concurrency", "system_gateway_limits", "prompt_cache_enabled",
            "system_prompt_cache", "capture_aggregate_inputs", "sql_capture_aggregate_inputs",
            "aggregate_input_schemas", "aggregate_capture_scope", "abacus_queue_batch_size",
            "query_timeout_seconds", "request_timeout_seconds", "sema_request_timeout_seconds",
        ):
            if key in source:
                config[key] = source[key]
        # The server-wide pool cap must not be lower than any route-specific cap.
        limits = config.get("system_gateway_limits", {})
        if limits:
            config["gateway_request_limit"] = max(limits.values())
        config["rerun_variants"] = [v["name"] for v in config["variants"]]
        config["settings_source"] = str(args.settings_from.resolve())
        save(run / "campaign.json", config)
    if args.semcache_followup:
        config["semcache_followup"] = True
        save(run / "campaign.json", config)
    with (run / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(
            run / "controller-launch.json",
            {
                "pid": os.getpid(),
                "started_unix": time.time(),
                "entrypoint": "parallel_campaign",
            },
        )
        try:
            fingerprints = execution_fingerprints()
            previous = run / "execution-source-hashes.json"
            if previous.exists() and json.loads(previous.read_text()) != fingerprints:
                raise ValueError("Execution sources changed; start a fresh full run")
            save(previous, fingerprints)
            frozen = prepare(run, config)
            if args.capture_aggregate_inputs:
                if args.references is None:
                    raise ValueError("Aggregate capture requires --references for empty-input schemas")
                schemas = {}
                for variant in config["variants"]:
                    if variant["system"] in {"plop", "sema"}:
                        continue
                    for query in config["query_ids"]:
                        folder = args.references.resolve() / variant["system"] / query
                        pre = folder / "pre-aggregate-result.json"
                        if pre.exists():
                            answer = json.loads((folder / "answer.json").read_text())
                            if frozen["query_files"][variant["system"] + "/" + query]["sha256"] != answer["query_sha256"]:
                                raise ValueError("Reference aggregate schema belongs to a different query")
                            schemas[variant["system"] + "/" + query] = json.loads(pre.read_text())["columns"]
                config.update(capture_aggregate_inputs=True, aggregate_input_schemas=schemas,
                              aggregate_capture_scope="Python host aggregates; SQL optimizer and timing boundaries preserved")
            models = model_snapshot()
            snapshot_environment(run, models)
            cpu_sets = allocate_cpu_sets(count=len(models))
            lanes = build_lanes(config, models, cpu_sets)
            if len({lane["model_id"] for lane in lanes}) != len(lanes):
                raise ValueError("Parallel lanes require distinct actual model IDs")
            transports = verify_transports(lanes)
            config.update(
                scheduling="per_model_parallel_then_pool_serial",
                model_lanes=lanes,
                single_model_cache_reset="only the owning model before each configuration",
                pool_cache_reset="all configured models before each pool configuration",
            )
            save(run / "campaign.json", config)
            save(
                run / "cpu-layout.json",
                {
                    "lanes": lanes,
                    "logical_cpus_per_lane": CONCURRENCY,
                    "physical_core_sharing_between_lanes": False,
                    "unassigned_cpu_ids": sorted(
                        set(os.sched_getaffinity(0))
                        - {cpu for group in cpu_sets for cpu in group}
                    ),
                },
            )
            save(run / "plop-lane-transports.json", transports)
            save(
                run / "status.json",
                {
                    "phase": "running_parallel",
                    "controller_pid": os.getpid(),
                    "lane_status_files": [lane["status_file"] for lane in lanes],
                    "updated_unix": time.time(),
                },
            )
            render_progress(run)

            def run_variant(lane, name):
                selected = [task for task in config["tasks"] if task["variant"] == name]
                execute(
                    run,
                    {**config, "tasks": selected},
                    frozen,
                    persist_campaign=False,
                    final_phase="variant_finished",
                    cpu_set=lane["cpu_set"],
                    gateway_port=lane["gateway_port"],
                    status_file=lane["status_file"],
                    models=models,
                    write_shared_metadata=False,
                    source_hashes=fingerprints,
                )

            run_model_lanes(lanes, run_variant)
            # Multi-model candidates share endpoints, so start only after all
            # eight single-model configurations have finished and drained.
            pool_tasks = [
                task
                for task in config["tasks"]
                if task["system"] not in {"plop", "sema"}
            ]
            execute(
                run,
                {**config, "tasks": pool_tasks},
                frozen,
                persist_campaign=False,
                cpu_set=format_cpu_set(cpu_sets[0]),
                models=models,
                write_shared_metadata=False,
                source_hashes=fingerprints,
            )
        except BaseException as error:
            (run / "controller-error.txt").write_text(traceback.format_exc())
            save(
                run / "status.json",
                {
                    "phase": "controller_failed",
                    "reason": f"{type(error).__name__}: {error}",
                    "updated_unix": time.time(),
                },
            )
            render_progress(run)
            raise


if __name__ == "__main__":
    main()
