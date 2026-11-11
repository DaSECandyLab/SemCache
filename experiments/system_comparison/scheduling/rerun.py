"""Overwrite explicitly selected tasks inside an existing comparison directory."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import time
import traceback
from pathlib import Path

from experiments.core.io import digest, save
from experiments.core.models import model_snapshot
from experiments.core.resources import CONCURRENCY
from experiments.system_comparison.reporting.monitor import render
from experiments.system_comparison.scheduling.campaign import (
    execute,
    execution_fingerprints,
    prepare,
    snapshot_environment,
)
from experiments.system_comparison.scheduling.parallel import (
    run_model_lanes,
    verify_transports,
)
from experiments.system_comparison.transport.plop_protocol import PLOP_PROMPT_PROTOCOLS


def select_tasks(config, variant, dataset=None, queries=None):
    selected = [
        task
        for task in config["tasks"]
        if task["variant"] == variant
        and (dataset is None or task["query_id"].split("/")[0] == dataset)
    ]
    if queries:
        if dataset is None and any("/" not in query for query in queries):
            raise ValueError(
                "Query selection without --dataset requires full dataset/query IDs"
            )
        wanted = {q if "/" in q else f"{dataset}/{q}" for q in queries}
        selected = [task for task in selected if task["query_id"] in wanted]
    if not selected:
        raise ValueError("No tasks match the requested variant and dataset: " + variant)
    return selected


def snapshot_moar_selected_plans(run, frozen, selected):
    """Validate and hold selected plan inputs before an in-place overwrite."""
    import hashlib
    import yaml

    plans = {}
    for task in selected:
        folder = run / task["folder"]
        old_job = json.loads((folder / "job.json").read_text())
        old_summary = json.loads((folder / "summary.json").read_text())
        source = (folder / "selected-pipeline.yaml").read_text()
        parsed = yaml.safe_load(source)
        expected = frozen["jobs"][task["query_id"]]
        expected_hash = frozen["query_files"]["moar/" + task["query_id"]]["sha256"]
        expected_db = expected["input_databases"]["moar"]["sha256"]
        if (old_job.get("id") != task["query_id"] or old_job.get("query_sha256") != expected_hash
                or old_job.get("input_database_sha256") != expected_db
                or old_job.get("output_schema") != expected.get("output_schema")
                or not isinstance(parsed, dict) or not parsed.get("operations") or not parsed.get("pipeline")):
            raise ValueError("Saved MOAR plan does not match the frozen task: " + task["task_id"])
        search_path = folder / "moar-search-summary.json"
        search = json.loads(search_path.read_text()) if search_path.exists() else {}
        plans[task["task_id"]] = {
            "query_id": task["query_id"], "query_sha256": expected_hash,
            "input_database_sha256": expected_db,
            "source_plan_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "source_pipeline_yaml": source,
            "previous_status": old_summary.get("status"),
            "previous_search_seconds": search.get("seconds"),
            "previous_search_iterations": search.get("iterations"),
        }
    return plans


def select_model_lanes(config, names, models):
    """Reuse the original resources, restoring membership from model ownership."""
    variants = {item["name"]: item for item in config["variants"]}
    models_by_alias = {item["alias"]: item for item in models}
    chosen = {}
    for name in names:
        variant = variants[name]
        allowed = variant.get("allowed", [])
        if variant["system"] not in {"plop", "sema"} or len(allowed) != 1:
            raise ValueError(
                "Parallel reruns require single-model PLOP or Sema configurations"
            )
        matches = [
            lane
            for lane in config.get("model_lanes", [])
            if lane["model"] == allowed[0]
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one original model lane for {name}")
        lane = matches[0]
        model = models_by_alias.get(lane["model"])
        if model is None or model["id"] != lane["model_id"]:
            raise ValueError(f"Actual model identity changed for {name}")
        if lane["model"] not in chosen:
            chosen[lane["model"]] = {**lane, "variants": []}
        chosen[lane["model"]]["variants"].append(name)
    lanes = list(chosen.values())
    if len({lane["model_id"] for lane in lanes}) != len(lanes):
        raise ValueError("Parallel lanes require distinct actual model IDs")
    if len({lane["gateway_port"] for lane in lanes}) != len(lanes):
        raise ValueError("Parallel lanes require distinct gateway ports")
    if len({lane["status_file"] for lane in lanes}) != len(lanes):
        raise ValueError("Parallel lanes require distinct status files")
    cpus = set()
    for lane in lanes:
        owned = set(lane["cpu_ids"])
        if len(owned) != CONCURRENCY or cpus & owned:
            raise ValueError(
                "Parallel lanes require disjoint original 16-CPU allocations"
            )
        cpus.update(owned)
        status = Path(lane["status_file"])
        if status.is_absolute() or ".." in status.parts or status.parts[0] != "lanes":
            raise ValueError(
                "Parallel status files must stay inside the lane directory"
            )
    return lanes


def preserved_output_hashes(run, selected_folders):
    """Hash every other task output, including tasks outside the campaign list."""
    return {
        str(path.relative_to(run)): digest(path)
        for path in sorted((run / "tasks").rglob("*"))
        if path.is_file()
        and not any(path.is_relative_to(folder) for folder in selected_folders)
    }


def concurrent_variant_roots(run, config, names, concurrent_names, models, lanes):
    """Exclude only declared external workers with independent model resources."""
    if not concurrent_names:
        return []
    if len(concurrent_names) != len(set(concurrent_names)):
        raise ValueError("Concurrent configurations cannot be repeated")
    if set(names) & set(concurrent_names):
        raise ValueError("Concurrent configurations cannot also be selected for rerun")
    for name in concurrent_names:
        select_tasks(config, name)
    concurrent_lanes = select_model_lanes(config, concurrent_names, models)
    if {lane["model"] for lane in lanes} & {lane["model"] for lane in concurrent_lanes}:
        raise ValueError("Concurrent configurations cannot share a selected model lane")
    # Validate model IDs, CPU allocations, ports and status files before scheduling selected configurations.
    select_model_lanes(config, [*names, *concurrent_names], models)
    roots = [run / "tasks" / name for name in concurrent_names]
    if any(
        not root.resolve().is_relative_to((run / "tasks").resolve())
        or root.resolve() == (run / "tasks").resolve()
        for root in roots
    ):
        raise ValueError("Concurrent task root escaped the campaign tasks directory")
    return roots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--abacus-min-quality", type=float)
    parser.add_argument("--capture-aggregate-inputs", action="store_true")
    parser.add_argument("--references", type=Path, help="Frozen reference schemas for empty aggregate inputs")
    variants = parser.add_mutually_exclusive_group(required=True)
    variants.add_argument("--variant")
    variants.add_argument(
        "--variants", nargs="+", help="Complete configurations in the listed order"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run distinct original model lanes concurrently, serial within each model",
    )
    parser.add_argument(
        "--concurrent-variants",
        nargs="+",
        default=[],
        help="Independent single-model configurations already running externally; exclude only their task roots from static-output verification",
    )
    parser.add_argument(
        "--dataset",
        choices=["estate", "movies", "appstore", "tpch"],
        help="Omit to rerun every query in each selected configuration",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        help="Optional query IDs (Q5 or estate/Q5); keep the same attempt-1 path",
    )
    parser.add_argument(
        "--skip-cache-reset",
        action="store_true",
        help="Retain model server caches for this explicitly requested rerun",
    )
    parser.add_argument(
        "--plop-prompt-protocol",
        choices=PLOP_PROMPT_PROTOCOLS,
        help="Opt into the PLOP task-output adapter; teacher answers remain unchanged",
    )
    parser.add_argument(
        "--abacus-no-sampling",
        action="store_true",
        help="Use native Dataset.run, without MAB, Validator or training samples",
    )
    parser.add_argument(
        "--reuse-moar-selected-plans",
        action="store_true",
        help="Reuse each selected MOAR task's saved selected plan; do not search again",
    )
    parser.add_argument(
        "--final-phase",
        choices=["selection_finished", "repeat_pending"],
        default="selection_finished",
        help="Keep the monitor alive between repeat measurements",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Preserve finalized tasks and continue the selected sequence",
    )
    args = parser.parse_args()
    if args.concurrent_variants and not args.parallel:
        raise ValueError("--concurrent-variants requires --parallel")
    run = args.run.resolve()
    config = json.loads((run / "campaign.json").read_text())
    names = args.variants or [args.variant]
    if len(names) != len(set(names)):
        raise ValueError("A rerun sequence cannot repeat a configuration")
    selected = [
        task
        for name in names
        for task in select_tasks(config, name, args.dataset, args.queries)
    ]
    if args.abacus_min_quality is not None:
        if not 0 <= args.abacus_min_quality <= 1 or not any(t["system"] == "abacus" for t in selected):
            raise ValueError("A quality threshold in [0, 1] requires an Abacus selection")
        config["abacus_min_quality"] = args.abacus_min_quality
    if args.capture_aggregate_inputs:
        if args.references is None:
            raise ValueError("--capture-aggregate-inputs requires --references for frozen schemas")
        schemas = {}
        for task in selected:
            folder = args.references.resolve() / task["system"] / task["query_id"]
            pre = folder / "pre-aggregate-result.json"
            if pre.exists():
                answer = json.loads((folder / "answer.json").read_text())
                query_path = run / "queries" / task["system"] / (task["query_id"] + ".py")
                if digest(query_path) != answer["query_sha256"]:
                    raise ValueError("Aggregate reference schema belongs to a different query")
                schemas[task["system"] + "/" + task["query_id"]] = json.loads(pre.read_text())["columns"]
        config.update(capture_aggregate_inputs=True, aggregate_input_schemas=schemas)
    if args.abacus_min_quality is not None or args.capture_aggregate_inputs:
        save(run / "pool-rerun-settings.json", {
            "selected_variants": names,
            "abacus_min_quality": config.get("abacus_min_quality"),
            "capture_aggregate_inputs": config.get("capture_aggregate_inputs", False),
            "references": str(args.references.resolve()) if args.references else None,
            "updated_unix": time.time(),
        })
    if args.plop_prompt_protocol:
        if any(task["system"] != "plop" for task in selected):
            raise ValueError("--plop-prompt-protocol applies only to PLOP reruns")
        config["plop_prompt_protocol"] = args.plop_prompt_protocol
    if args.abacus_no_sampling and any(task["system"] != "abacus" for task in selected):
        raise ValueError("--abacus-no-sampling applies only to Abacus tasks")
    if args.reuse_moar_selected_plans and (not args.overwrite or any(task["system"] != "moar" for task in selected)):
        raise ValueError("--reuse-moar-selected-plans requires --overwrite and MOAR tasks only")
    # Use the same lock as the full controller. Never interrupt another worker.
    with (run / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = prepare(run, config)
        reuse_plans = snapshot_moar_selected_plans(run, frozen, selected) if args.reuse_moar_selected_plans else {}
        lane = next(
            (
                item
                for item in config.get("model_lanes", [])
                if len(names) == 1 and names[0] in item.get("variants", [])
            ),
            None,
        )
        if (
            not args.parallel
            and len(names) > 1
            and any(task["system"] in {"plop", "sema"} for task in selected)
        ):
            raise ValueError(
                "Multi-configuration rerun sequences support pool systems only"
            )
        if lane is None and config.get("model_lanes"):
            # Pool variants run after all single-model lanes. Reuse the
            # declared 16-CPU pool allocation for an in-place rerun as well.
            first_lane = config["model_lanes"][0]
            lane = {
                "cpu_set": first_lane["cpu_set"],
                "gateway_port": max(
                    item["gateway_port"] for item in config["model_lanes"]
                )
                + 1,
                "status_file": "status.json",
            }
        folders = [run / task["folder"] for task in selected]
        if any(
            not folder.resolve().is_relative_to(
                (run / "tasks" / task["variant"]).resolve()
            )
            or folder.resolve() == (run / "tasks" / task["variant"]).resolve()
            or not (run / "tasks" / task["variant"])
            .resolve()
            .is_relative_to((run / "tasks").resolve())
            for task, folder in zip(selected, folders, strict=True)
        ):
            raise ValueError("Task output folder escaped the selected variant")
        if args.parallel:
            models = model_snapshot()
            lanes = select_model_lanes(config, names, models)
            concurrent_roots = concurrent_variant_roots(
                run, config, names, args.concurrent_variants, models, lanes
            )
            hash_exclusions = [*folders, *concurrent_roots]
            transports = verify_transports(lanes)
            fingerprints = execution_fingerprints()
            preserved = preserved_output_hashes(run, hash_exclusions)
            # Shared snapshots are written once, before any overwrite, never by lanes.
            snapshot_environment(run, models)
            save(run / "execution-source-hashes.json", fingerprints)
            save(run / "plop-lane-transports.json", transports)
            save(
                run / "controller-launch.json",
                {
                    "pid": os.getpid(),
                    "started_unix": time.time(),
                    "entrypoint": "parallel_rerun",
                    "selected_variants": names,
                    "concurrent_variants": args.concurrent_variants,
                },
            )
        for task, folder in zip(selected, folders, strict=True):
            if task["task_id"] in reuse_plans:
                folder.mkdir(parents=True, exist_ok=True)
                # Persist all source plans before removing any old run data.
                # The saved file remains in the existing query directory so a
                # controller interruption cannot erase the selected plan.
                save(folder / "plan-reuse-input.json", reuse_plans[task["task_id"]])
        for task, folder in zip(selected, folders, strict=True):
            if args.overwrite and folder.exists():
                if task["task_id"] in reuse_plans:
                    for child in folder.iterdir():
                        if child.name == "plan-reuse-input.json":
                            continue
                        if child.is_dir() and not child.is_symlink():
                            shutil.rmtree(child)
                        else:
                            child.unlink()
                else:
                    shutil.rmtree(folder)
        save(
            run / "status.json",
            {
                "phase": "preparing",
                "selected_variants": names,
                "concurrent_variants": args.concurrent_variants,
                "selected_tasks": [t["task_id"] for t in selected],
                "updated_unix": time.time(),
            },
        )
        render(run)
        try:
            # campaign.json continues to describe the full campaign. Only the
            # in-memory schedule is narrowed to the explicitly selected set.
            if args.parallel:
                reason = "Rerunning only selected configurations: " + ", ".join(names)
                if args.concurrent_variants:
                    reason += "; independently running configurations: " + ", ".join(
                        args.concurrent_variants
                    )
                for model_lane in lanes:
                    save(
                        run / model_lane["status_file"],
                        {
                            "phase": "preparing",
                            "selected_variants": model_lane["variants"],
                            "updated_unix": time.time(),
                        },
                    )
                save(
                    run / "status.json",
                    {
                        "phase": "running_parallel",
                        "controller_pid": os.getpid(),
                        "selected_variants": names,
                        "concurrent_variants": args.concurrent_variants,
                        "selected_tasks": [task["task_id"] for task in selected],
                        "lane_status_files": [item["status_file"] for item in lanes],
                        "reason": reason,
                        "updated_unix": time.time(),
                    },
                )
                render(run)

                def run_variant(model_lane, name):
                    execute(
                        run,
                        {
                            **config,
                            "tasks": [
                                task for task in selected if task["variant"] == name
                            ],
                        },
                        frozen,
                        persist_campaign=False,
                        final_phase="variant_finished",
                        cpu_set=model_lane["cpu_set"],
                        gateway_port=model_lane["gateway_port"],
                        status_file=model_lane["status_file"],
                        reset_variant_caches=not args.skip_cache_reset,
                        models=models,
                        write_shared_metadata=False,
                        source_hashes=fingerprints,
                    )

                run_model_lanes(lanes, run_variant)
                if preserved_output_hashes(run, hash_exclusions) != preserved:
                    raise RuntimeError(
                        "Unselected task outputs changed during parallel rerun"
                    )
                save(
                    run / "status.json",
                    {
                        "phase": args.final_phase,
                        "controller_pid": os.getpid(),
                        "selected_variants": names,
                        "concurrent_variants": args.concurrent_variants,
                        "selected_tasks": [task["task_id"] for task in selected],
                        "preserved_outputs_verified": True,
                        "reason": reason,
                        "updated_unix": time.time(),
                    },
                )
                render(run)
            else:
                execute(
                    run,
                    {**config, "tasks": selected},
                    frozen,
                    persist_campaign=False,
                    final_phase=args.final_phase,
                    cpu_set=lane.get("cpu_set") if lane else None,
                    gateway_port=lane.get("gateway_port") if lane else None,
                    reset_variant_caches=not args.skip_cache_reset,
                    abacus_sampling=not args.abacus_no_sampling,
                    status_file=lane.get("status_file", "status.json")
                    if lane
                    else "status.json",
                )
        except BaseException as error:
            (run / "controller-error.txt").write_text(traceback.format_exc())
            save(
                run / "status.json",
                {
                    "phase": "controller_failed",
                    "selected_variants": names,
                    "concurrent_variants": args.concurrent_variants,
                    "reason": f"{type(error).__name__}: {error}",
                    "updated_unix": time.time(),
                },
            )
            render(run)
            raise


if __name__ == "__main__":
    main()
