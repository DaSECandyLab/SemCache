"""Overwrite and rerun every configuration except preserved PLOP tasks."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
import traceback

from experiments.core.io import digest, save
from experiments.core.models import model_snapshot
from experiments.core.paths import ROOT
from experiments.system_comparison.reporting.monitor import render
from experiments.system_comparison.scheduling.campaign import execute, execution_fingerprints, prepare, snapshot_environment
from experiments.system_comparison.scheduling.moar import prepare as prepare_moar
from experiments.system_comparison.scheduling.parallel import build_lanes, run_model_lanes
from experiments.system_comparison.scheduling.rerun import preserved_output_hashes


def plop_hashes(run):
    return {str(path.relative_to(run)): digest(path)
            for folder in sorted((run / "tasks").glob("plop-*"))
            for path in sorted(folder.rglob("*")) if path.is_file()}


def refresh_queries(run, config, frozen, tasks):
    source = ROOT / "benchmarks/queries/translated/manifest.json"
    current = json.loads(source.read_text())
    records = {(item["system"], item["id"]): item for item in current["records"]}
    revised = []
    for task in tasks:
        system, query = task["system"], task["query_id"]
        record = records[("nirvana" if system == "moar" else system, query)]
        original = ROOT / "benchmarks/queries" / record["query_file"]
        key = system + "/" + query
        target = Path(frozen["query_files"][key]["path"])
        old = digest(target) if target.exists() else None
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        if system != "moar":
            shutil.copyfile(ROOT / "benchmarks/queries" / record["metadata_file"], target.with_suffix(".json"))
        frozen["query_files"][key] = {"path": str(target), "sha256": digest(target), "source": record["query_file"]}
        if old != digest(target):
            revised.append({"system": system, "query": query, "before_sha256": old, "after_sha256": digest(target)})
    selected_systems = {task["system"] for task in tasks}
    snapshot_path = run / "query-manifest.json"
    snapshot = json.loads(snapshot_path.read_text())
    snapshot["records"] = [records.get((item["system"], item["id"]), item)
                           if item["system"] in selected_systems else item for item in snapshot["records"]]
    save(snapshot_path, snapshot)
    config.update(query_manifest_sha256=digest(snapshot_path), source_query_manifest_sha256=digest(source))
    save(run / "query-revisions.json", {"policy": "Refresh only selected translations; original queries unchanged", "revisions": revised})
    save(run / "inputs/manifest.json", frozen)


def initialize_rerun(run, references, agent_config, names=None):
    config, frozen, _ = prepare_moar(run, references, agent_config)
    prepare(run, config)   
    names = names or [v["name"] for v in config["variants"] if v["system"] != "plop"]
    if len(names) != len(set(names)) or set(names) - {v["name"] for v in config["variants"]}:
        raise ValueError("Unknown or duplicate selected configurations")
    selected = [task for name in names for task in config["tasks"] if task["variant"] == name]
    refresh_queries(run, config, frozen, selected)
    schemas = dict(config.get("aggregate_input_schemas", {}))
    for task in selected:
        system = "nirvana" if task["system"] == "moar" else task["system"]
        path = references / system / task["query_id"] / "pre-aggregate-result.json"
        if path.exists():
            schemas[task["system"] + "/" + task["query_id"]] = json.loads(path.read_text())["columns"]
    preserved = [v["name"] for v in config["variants"] if v["name"] not in names]
    for lane in config["model_lanes"]:
        lane["variants"] = [name for name in lane["variants"] if name not in preserved]
    config.update(
        preserved_variants=preserved, query_timeout_seconds=1800, gateway_request_limit=16,
        system_concurrency={"plop": 16, "sema": 16, "nirvana": 32, "lotus": 32, "abacus": 32, "moar": 32},
        system_gateway_limits={"plop": 16, "sema": 16, "nirvana": 32, "lotus": 32, "abacus": 32, "moar": 32},
        capture_aggregate_inputs=True, sql_capture_aggregate_inputs=True, aggregate_input_schemas=schemas,
        aggregate_capture_scope="same full execution; SQL input materialized once, Python input observed at final aggregate",
        rerun_variants=names,
        scheduled_attempts=len(config["tasks"]), scheduling="selected_model_lanes_then_pool_serial",
        next_run_timeout_seconds=1800,
        timeout_policy="1800-second whole-query deadline; clear execution KV cache at configuration start, retain across queries",
    )
    config["moar"].update(max_threads=32, max_requests_in_flight=32, preserve_model_cache=False)
    save(run / "campaign.json", config)
    control = json.loads((run / "control.json").read_text())
    control.update(query_timeout_seconds=1800, preserve_cache=False, disabled_variants=preserved,
                   disabled_systems=[], skipped_tasks=[], reason="Rerun selected configurations: " + ", ".join(names))
    control.pop("skip_reason", None)
    save(run / "control.json", control)
    return config, frozen, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--agent-config", type=Path, default=ROOT / "SemCache/config/teacher/teacher.opencode.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--variants", nargs="+", help="Override the default non-PLOP selection")
    parser.add_argument("--prompt-cache", action="store_true", help="Reuse exact completed model responses within each configuration")
    parser.add_argument("--no-prompt-cache-systems", nargs="+", default=[], choices=["plop", "sema", "nirvana", "lotus", "abacus", "moar"])
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run, references = args.run.resolve(), args.references.resolve()
    with (run / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config, frozen, selected = initialize_rerun(run, references, args.agent_config, args.variants)
        config.update(prompt_cache_enabled=args.prompt_cache, next_run_status="prepared_only" if args.prepare_only else "running")
        config["system_prompt_cache"] = {system: args.prompt_cache and system not in args.no_prompt_cache_systems
                                         for system in ("plop", "sema", "nirvana", "lotus", "abacus", "moar")}
        save(run / "campaign.json", config)
        control = json.loads((run / "control.json").read_text())
        control["next_run_status"] = config["next_run_status"]
        save(run / "control.json", control)
        selected_folders = [run / task["folder"] for task in selected]
        baseline_path = run / "preserved-outputs.json"
        preserved = preserved_output_hashes(run, selected_folders)
        if args.resume and baseline_path.exists() and json.loads(baseline_path.read_text())["files"] != preserved:
            raise ValueError("Unselected outputs changed since rerun started")
        save(baseline_path, {"files": preserved, "verified_unix": time.time()})
        if args.prepare_only:
            print(f"Prepared {len(selected)} selected tasks; unselected files preserved: {len(preserved)}")
            render(run)
            return
        if not args.resume:
            for task in selected:
                folder = run / task["folder"]
                if not folder.resolve().is_relative_to((run / "tasks" / task["variant"]).resolve()):
                    raise ValueError("Task folder escaped variant")
                if folder.exists():
                    shutil.rmtree(folder)
        fingerprints = execution_fingerprints()
        save(run / "execution-source-hashes.json", fingerprints)
        models = model_snapshot()
        snapshot_environment(run, models)
        source_lanes = config["model_lanes"]
        lanes = build_lanes(config, models, [lane["cpu_ids"] for lane in source_lanes])
        for lane in lanes:
            lane["variants"] = [name for name in lane["variants"] if name not in config["preserved_variants"]]
        lanes = [lane for lane in lanes if lane["variants"]]
        config["model_lanes"] = lanes
        save(run / "campaign.json", config)
        record = {"pid": os.getpid(), "started_unix": time.time(), "entrypoint": __name__,
                  "selected_tasks": len(selected), "preserved_tasks": len(config["tasks"]) - len(selected),
                  "selected_variants": config["rerun_variants"], "cache_reset": "each configuration start"}
        save(run / "controller-launch.json", record)
        save(run / "selected-rerun.json", record)
        for lane in lanes:
            save(run / lane["status_file"], {"phase": "preparing", "variant": lane["variants"][0]})
        save(run / "status.json", {"phase": "running_parallel", "controller_pid": os.getpid(),
                                  "lane_status_files": [lane["status_file"] for lane in lanes],
                                  "reason": "Rerun selected configurations: " + ", ".join(config["rerun_variants"]), "updated_unix": time.time()})
        render(run)
        try:
            def run_variant(lane, variant):
                execute(run, {**config, "tasks": [task for task in selected if task["variant"] == variant]}, frozen,
                        persist_campaign=False, final_phase="variant_finished", cpu_set=lane["cpu_set"],
                        gateway_port=lane["gateway_port"], status_file=lane["status_file"], models=models,
                        write_shared_metadata=False, source_hashes=fingerprints, reset_variant_caches=True)

            run_model_lanes(lanes, run_variant)
            pool_order = {system: i for i, system in enumerate(("nirvana", "lotus", "abacus", "moar"))}
            pool = sorted((task for task in selected if task["system"] not in {"plop", "sema"}), key=lambda task: pool_order[task["system"]])
            execute(run, {**config, "tasks": pool, "gateway_request_limit": 32}, frozen, persist_campaign=False, cpu_set=source_lanes[0]["cpu_set"],
                    models=models, write_shared_metadata=False, source_hashes=fingerprints, reset_variant_caches=True)
            if preserved_output_hashes(run, selected_folders) != preserved:
                raise RuntimeError("Unselected task output changed")
            save(baseline_path, {"files": preserved, "verified_unix": time.time(), "unchanged": True})
        except BaseException as error:
            (run / "controller-error.txt").write_text(traceback.format_exc())
            save(run / "status.json", {"phase": "controller_failed", "reason": f"{type(error).__name__}: {error}", "updated_unix": time.time()})
            render(run)
            raise


if __name__ == "__main__":
    main()
