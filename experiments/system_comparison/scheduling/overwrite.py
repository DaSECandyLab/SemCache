"""Overwrite or resume a campaign, retaining PLOP and omitting MaxQuality."""

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
from experiments.core.paths import ROOT
from experiments.system_comparison.scheduling.campaign import (
    execute,
    execution_fingerprints,
    prepare,
    render_progress,
    snapshot_environment,
)
from experiments.system_comparison.scheduling.parallel import run_model_lanes

SKIPPED = "abacus-MaxQuality"


def prepare_resume(run, config):
    """Replace invalid attempts while keeping finalized successful results."""
    frozen = prepare(run, config)
    for query in frozen["query_files"].values():
        if digest(query["path"]) != query["sha256"]:
            raise ValueError("Frozen query changed: " + query["path"])
    removed, preserved = [], {}
    for task in config["tasks"]:
        folder = run / task["folder"]
        if not folder.resolve().is_relative_to((run / "tasks").resolve()):
            raise ValueError("Task output path escaped the campaign")
        summary = folder / "summary.json"
        record = json.loads(summary.read_text()) if summary.exists() else {}
        valid = (
            record.get("status") == "completed"
            and record.get("parent_finalized")
            and (folder / "result.json").is_file()
        )
        if task["system"] == "plop" or valid:
            preserved.update(
                {
                    str(p.relative_to(run)): digest(p)
                    for p in folder.rglob("*")
                    if p.is_file()
                }
            )
        elif folder.exists() and task["variant"] != SKIPPED:
            shutil.rmtree(folder)
            removed.append(task["task_id"])
    save(
        run / "repair-resume.json",
        {
            "started_unix": time.time(),
            "removed_partial_tasks": removed,
            "reason": "Resume invalid attempts; preserve successful outputs",
        },
    )
    return config, frozen, preserved


def prepare_overwrite(run, config):
    """Validate sources first, then replace only explicitly rerun systems."""
    frozen = prepare(run, config)
    source = ROOT / "benchmarks/queries/translated/manifest.json"
    manifest = json.loads(source.read_text())
    if manifest["contract"] != config["contract"]:
        raise ValueError("Cannot mix workload contracts in an existing campaign")
    copies = []
    for record in manifest["records"]:
        if record["system"] == "semcache":
            continue
        query = ROOT / "benchmarks/queries" / record["query_file"]
        metadata = ROOT / "benchmarks/queries" / record["metadata_file"]
        key = record["system"] + "/" + record["id"]
        target = Path(frozen["query_files"][key]["path"])
        if not target.resolve().is_relative_to((run / "queries").resolve()):
            raise ValueError("Frozen query path escaped the campaign")
        if digest(query) != record["sha256"] or not metadata.is_file():
            raise ValueError("Invalid generated query: " + key)
        if record["system"] == "plop":
            if digest(target) != record["sha256"]:
                raise ValueError(
                    "Retained PLOP query differs from the current workload"
                )
        else:
            copies.append((query, metadata, target, key, record))
    preserved = {
        str(path.relative_to(run)): digest(path)
        for folder in (run / "tasks").glob("plop-*")
        for path in folder.rglob("*")
        if path.is_file()
    }
    for query, metadata, target, key, record in copies:
        shutil.copyfile(query, target)
        shutil.copyfile(metadata, target.with_suffix(".json"))
        frozen["query_files"][key].update(
            sha256=record["sha256"], source=record["query_file"]
        )
    save(run / "inputs/manifest.json", frozen)
    save(run / "query-manifest.json", manifest)
    config = {
        **config,
        "variants": [v for v in config["variants"] if v["name"] != SKIPPED],
        "tasks": [t for t in config["tasks"] if t["variant"] != SKIPPED],
        "query_manifest_sha256": digest(source),
    }
    config["scheduled_attempts"] = len(config["tasks"])
    config["preserved_variants"] = [
        v["name"] for v in config["variants"] if v.get("system") == "plop"
    ]
    config["model_lanes"] = [
        {**lane, "variants": [v for v in lane["variants"] if v.startswith("sema-")]}
        for lane in config.get("model_lanes", [])
    ]
    for variant in {t["variant"] for t in config["tasks"] if t["system"] != "plop"} | {
        SKIPPED
    }:
        folder = run / "tasks" / variant
        if folder.exists():
            shutil.rmtree(folder)
        (run / "cache-resets" / (variant + ".json")).unlink(missing_ok=True)
    for path in (run / "lanes").glob("*.json"):
        path.unlink()
    save(
        run / "control.json",
        {
            "disabled_systems": [],
            "disabled_variants": [SKIPPED],
            "excluded_systems": [],
            "updated_unix": time.time(),
            "reason": "User authorized overwrite in place, retaining PLOP and skipping Abacus MaxQuality",
        },
    )
    save(run / "campaign.json", config)
    save(
        run / "overwrite.json",
        {
            "updated_unix": time.time(),
            "skipped_variants": [SKIPPED],
            "rerun_tasks": sum(t["system"] != "plop" for t in config["tasks"]),
            "preserved_plop_files": preserved,
        },
    )
    return config, frozen, preserved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument(
        "--resume",
        action="store_true",
        help="Retry invalid attempts and continue unfinished tasks",
    )
    args = parser.parse_args()
    run = args.run.resolve()
    with (run / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(
            run / "controller-launch.json",
            {
                "pid": os.getpid(),
                "started_unix": time.time(),
                "entrypoint": "overwrite_campaign",
            },
        )
        try:
            config = json.loads((run / "campaign.json").read_text())
            models = model_snapshot()
            config, frozen, preserved = (
                prepare_resume if args.resume else prepare_overwrite
            )(run, config)
            fingerprints = execution_fingerprints()
            save(run / "execution-source-hashes.json", fingerprints)
            snapshot_environment(run, models)
            lanes = [
                {
                    **lane,
                    "variants": [v for v in lane["variants"] if v.startswith("sema-")],
                }
                for lane in config["model_lanes"]
            ]
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

            def run_variant(lane, name, status_file=None):
                selected = [task for task in config["tasks"] if task["variant"] == name]
                execute(
                    run,
                    {**config, "tasks": selected},
                    frozen,
                    persist_campaign=False,
                    final_phase="variant_finished",
                    cpu_set=lane["cpu_set"],
                    gateway_port=lane["gateway_port"],
                    status_file=status_file or lane["status_file"],
                    models=models,
                    write_shared_metadata=False,
                    source_hashes=fingerprints,
                )

            run_model_lanes(lanes, run_variant)
            for name in ("abacus-MinTime", "nirvana-pool", "lotus-four-stage"):
                run_variant(lanes[0], name, "status.json")
            if any(digest(run / name) != sha for name, sha in preserved.items()):
                raise RuntimeError("Retained PLOP files changed during overwrite")
            save(
                run / "status.json",
                {
                    "phase": "finished",
                    "controller_pid": os.getpid(),
                    "updated_unix": time.time(),
                    "preserved_plop_verified": True,
                },
            )
        except BaseException as error:
            (run / "controller-error.txt").write_text(traceback.format_exc())
            save(
                run / "status.json",
                {
                    "phase": "controller_failed",
                    "controller_pid": os.getpid(),
                    "reason": f"{type(error).__name__}: {error}",
                    "updated_unix": time.time(),
                },
            )
            raise
        finally:
            render_progress(run)


if __name__ == "__main__":
    main()
