"""Repeat a selected query twice per configuration, keeping both measurements."""

from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from experiments.core.io import digest, save
from experiments.core.models import model_snapshot
from experiments.core.paths import ROOT
from experiments.system_comparison.reporting.monitor import read
from experiments.system_comparison.scheduling.campaign import (
    execute,
    execution_fingerprints,
    prepare,
    render_progress,
    snapshot_environment,
)
from experiments.system_comparison.scheduling.parallel import (
    run_model_lanes,
    verify_transports,
)
from experiments.system_comparison.scheduling.rerun import select_tasks


def refresh_query(run, config, frozen, query_id):
    """Refresh only the selected query; validate all copies before changing any."""
    source = ROOT / "benchmarks/queries/translated/manifest.json"
    manifest = read(source)
    records = [r for r in manifest["records"] if r["id"] == query_id]
    systems = {v["system"] for v in config["variants"]}
    if not systems <= {r["system"] for r in records}:
        raise ValueError("Selected query does not cover every requested system")
    copies = []
    for record in records:
        if record["system"] == "semcache":
            continue
        key = record["system"] + "/" + query_id
        query = ROOT / "benchmarks/queries" / record["query_file"]
        metadata = ROOT / "benchmarks/queries" / record["metadata_file"]
        target = Path(frozen["query_files"][key]["path"])
        if not target.resolve().is_relative_to((run / "queries").resolve()):
            raise ValueError("Frozen query path escaped the campaign")
        if digest(query) != record["sha256"] or not metadata.is_file():
            raise ValueError("Invalid translated query: " + key)
        copies.append((key, query, metadata, target, record))
    for key, query, metadata, target, record in copies:
        shutil.copyfile(query, target)
        shutil.copyfile(metadata, target.with_suffix(".json"))
        frozen["query_files"][key].update(
            sha256=record["sha256"], source=record["query_file"]
        )
    snapshot = read(run / "query-manifest.json")
    snapshot["records"] = [
        r for r in snapshot["records"] if r["id"] != query_id
    ] + records
    save(run / "query-manifest.json", snapshot)
    save(run / "inputs/manifest.json", frozen)
    return {key: record["sha256"] for key, _, _, _, record in copies}


def publish(run, task, folders, summaries):
    """Publish the latest result while retaining independently addressable runs."""
    target = run / task["folder"]
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(folders[-1], target)
    latest = {**summaries[-1], "folder": task["folder"]}
    if len(summaries) == 2:
        first, second = summaries
        complete = all(s.get("status") == "completed" for s in summaries)
        average = (first["seconds"] + second["seconds"]) / 2 if complete else None
        manifest = {
            "variant": task["variant"],
            "query": task["query_id"],
            "initial": {
                "status": first["status"],
                "seconds": first.get("seconds"),
                "artifact": str(folders[0].relative_to(run)),
            },
            "second_run": {
                "status": second["status"],
                "seconds": second.get("seconds"),
                "artifact": str(folders[1].relative_to(run)),
            },
            "average_seconds": average,
            "cache_reset_before_initial": True,
            "cache_reset_before_second": False,
            "recorded_unix": time.time(),
        }
        save(target / "repeat-manifest.json", manifest)
        latest.update(
            repeat_runs=2,
            initial_seconds=first.get("seconds"),
            second_seconds=second.get("seconds"),
            repeat_manifest=str(target / "repeat-manifest.json"),
            repeat_results=[str(folder) for folder in folders],
            time_basis="two_run_mean" if complete else "single_latest_run",
        )
        if average is not None:
            latest["average_seconds"] = average
    save(target / "summary.json", latest)


def run_twice(
    run, config, frozen, task, lane, models, fingerprints, *, status_file=None
):
    folders, summaries = [], []
    for index in (1, 2):
        folder = (
            run / "repetitions" / task["variant"] / task["query_id"] / f"run-{index}"
        )
        repeated = {**task, "folder": str(folder.relative_to(run))}
        print(f"REPEAT {index}/2 {task['task_id']}", flush=True)
        execute(
            run,
            {**config, "tasks": [repeated]},
            frozen,
            persist_campaign=False,
            final_phase="variant_finished",
            cpu_set=lane["cpu_set"],
            gateway_port=lane["gateway_port"],
            status_file=status_file or lane["status_file"],
            models=models,
            write_shared_metadata=False,
            source_hashes=fingerprints,
            reset_variant_caches=index == 1,
        )
        folders.append(folder)
        summaries.append(read(folder / "summary.json"))
        publish(run, task, folders, summaries)


def render_repeats(run, config, query):
    now = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# {query} 两次运行",
        "",
        f"更新时间：{now}；每 10 秒刷新。",
        "",
        "每个配置第一轮前清模型服务缓存，第二轮保留缓存。两次独立启动原生查询，分别保存输出、计划和日志；仅两次均完成时计算平均时间。",
        "",
        "| 配置 | 第一次（秒） | 第二次（秒） | 平均（秒） | 输出与日志 |",
        "|---|---:|---:|---:|---|",
    ]
    counts, errors, data = 0, [], []
    for variant in config["variants"]:
        base = run / "repetitions" / variant["name"] / query
        folders = [base / f"run-{i}" for i in (1, 2)]
        records = [read(folder / "summary.json", {}) for folder in folders]
        valid = [
            r.get("parent_finalized") and r.get("status") == "completed"
            for r in records
        ]
        counts += sum(bool(r.get("parent_finalized")) for r in records)
        mean = sum(r["seconds"] for r in records) / 2 if all(valid) else None
        cells = [
            f"{r['seconds']:.2f}" if v else "-"
            for r, v in zip(records, valid, strict=True)
        ]
        links = " · ".join(
            f"[第{i}]({folder.relative_to(run)})" for i, folder in enumerate(folders, 1)
        )
        lines.append(
            f"| {variant['name']} | {' | '.join(cells)} | {mean:.2f} | {links} |"
            if mean is not None
            else f"| {variant['name']} | {' | '.join(cells)} | - | {links} |"
        )
        for index, record in enumerate(records, 1):
            if record.get("parent_finalized") and record.get("status") != "completed":
                errors.append(
                    f"- {variant['name']} 第 {index} 次：{record.get('status')}；{record.get('reason', '')}"
                )
        data.append(
            {
                "variant": variant["name"],
                "query_id": query,
                "runs": records,
                "average_seconds": mean,
            }
        )
    lines += ["", f"已结束 {counts}/{len(config['variants']) * 2} 次。", ""]
    if errors:
        lines.extend(["异常记录：", "", *errors, ""])
    filename = "repeat-" + query.replace("/", "-")
    # This renderer has one owner; the full monitor is separately synchronized.
    path = run / (filename + ".md")
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(path)
    save(
        run / (filename + ".json"),
        {"query_id": query, "completed_runs": counts, "records": data},
    )
    render_progress(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--exclude-variants", nargs="+", default=[])
    args = parser.parse_args()
    run = args.run.resolve()
    with (run / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = read(run / "campaign.json")
        unknown = set(args.exclude_variants) - {v["name"] for v in config["variants"]}
        if unknown:
            raise ValueError(f"Unknown configurations: {sorted(unknown)}")
        config = {
            **config,
            "variants": [
                v for v in config["variants"] if v["name"] not in args.exclude_variants
            ],
        }
        if not config["variants"]:
            raise ValueError("No configurations selected")
        tasks = {
            v["name"]: select_tasks(config, v["name"], queries=[args.query])[0]
            for v in config["variants"]
        }
        frozen = prepare(run, config)
        query_hashes = refresh_query(run, config, frozen, args.query)
        models = model_snapshot()
        pool_lane = config["model_lanes"][0]
        lanes = [
            {**lane, "variants": [name for name in lane["variants"] if name in tasks]}
            for lane in config["model_lanes"]
        ]
        lanes = [lane for lane in lanes if lane["variants"]]
        expected_lanes = {
            v["name"] for v in config["variants"] if v["system"] in {"plop", "sema"}
        }
        scheduled = [name for lane in lanes for name in lane["variants"]]
        if len(scheduled) != len(set(scheduled)) or set(scheduled) != expected_lanes:
            raise ValueError(
                "Model lanes do not cover the selected single-model configurations exactly"
            )
        if len({model["id"] for model in models}) != len(models):
            raise ValueError("Parallel lanes require distinct actual model IDs")
        verify_transports(lanes)
        snapshot_environment(run, models)
        fingerprints = execution_fingerprints()
        save(
            run / "repeat-selection.json",
            {
                "query": args.query,
                "variants": list(tasks),
                "excluded_variants": args.exclude_variants,
                "runs_per_variant": 2,
                "query_hashes": query_hashes,
                "cache_reset_before_initial": True,
                "cache_reset_before_second": False,
                "started_unix": time.time(),
            },
        )
        save(
            run / "controller-launch.json",
            {
                "pid": os.getpid(),
                "started_unix": time.time(),
                "entrypoint": "repeat_selection",
            },
        )
        for task in tasks.values():
            if (
                not (run / task["folder"])
                .resolve()
                .is_relative_to((run / "tasks" / task["variant"]).resolve())
            ):
                raise ValueError(
                    "Task output folder escaped its selected configuration"
                )
            for folder in (
                run / task["folder"],
                run / "repetitions" / task["variant"] / args.query,
            ):
                if not folder.resolve().is_relative_to(run):
                    raise ValueError("Repeated output folder escaped the campaign")
                if folder.exists():
                    shutil.rmtree(folder)
        save(
            run / "status.json",
            {
                "phase": "running_parallel",
                "controller_pid": os.getpid(),
                "lane_status_files": [lane["status_file"] for lane in lanes],
                "updated_unix": time.time(),
            },
        )
        done = threading.Event()

        def ticker():
            while not done.wait(10):
                render_repeats(run, config, args.query)

        thread = threading.Thread(target=ticker, name="repeat-progress", daemon=True)
        render_repeats(run, config, args.query)
        thread.start()

        def run_variant(lane, name, status_file=None):
            run_twice(
                run,
                config,
                frozen,
                tasks[name],
                lane,
                models,
                fingerprints,
                status_file=status_file,
            )

        try:
            if lanes:
                run_model_lanes(lanes, run_variant)
            for variant in config["variants"]:
                if variant["system"] not in {"plop", "sema"}:
                    run_variant(pool_lane, variant["name"], "status.json")
            save(
                run / "status.json",
                {
                    "phase": "selection_finished",
                    "controller_pid": os.getpid(),
                    "reason": f"{args.query}: two runs per configuration finished",
                    "updated_unix": time.time(),
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
            done.set()
            thread.join()
            render_repeats(run, config, args.query)


if __name__ == "__main__":
    main()
