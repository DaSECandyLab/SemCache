"""Run one query cold, then warm once only when the cold run completes."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from experiments.core.io import save
from experiments.system_comparison.reporting.monitor import render


def run_once(root: Path, run: Path, variant: str, query: str, *, warm: bool, log):
    command = [
        sys.executable,
        "-u",
        "-m",
        "experiments.system_comparison.scheduling.rerun",
        "--run",
        str(run),
        "--variant",
        variant,
        "--queries",
        query,
        "--overwrite",
    ]
    if warm:
        command.append("--skip-cache-reset")
    else:
        command.extend(["--final-phase", "repeat_pending"])
    print("START", "warm" if warm else "initial", *command, flush=True)
    result = subprocess.run(command, check=False, cwd=root, stdout=log, stderr=log)
    if result.returncode != 0:
        raise RuntimeError(f"rerun controller exited with {result.returncode}")
    folder = run / "tasks" / variant / query / "attempt-1"
    summary = json.loads((folder / "summary.json").read_text())
    return folder, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    from experiments.core.paths import ROOT as root

    run = args.run.resolve()
    folder = run / "tasks" / args.variant / args.query / "attempt-1"
    log_path = run / "conditional-repeat.log"
    with log_path.open("a") as log:
        log.write(
            json.dumps(
                {
                    "event": "conditional_repeat_started",
                    "variant": args.variant,
                    "query": args.query,
                    "unix": time.time(),
                }
            )
            + "\n"
        )
        first_folder, first = run_once(
            root, run, args.variant, args.query, warm=False, log=log
        )
        if first.get("status") == "timeout":
            save(
                folder / "repeat-manifest.json",
                {
                    "status": "stopped_after_initial_timeout",
                    "variant": args.variant,
                    "query": args.query,
                    "initial": {"status": "timeout", "seconds": first.get("seconds")},
                    "second_run": None,
                    "average_seconds": None,
                    "cache_reset_before_initial": True,
                    "cache_reset_before_second": None,
                    "recorded_unix": time.time(),
                },
            )
            save(
                run / "status.json",
                {
                    "phase": "selection_finished",
                    "reason": "Initial query timed out; second run cancelled",
                    "updated_unix": time.time(),
                },
            )
            render(run)
            print("STOP initial timeout; no warm rerun", flush=True)
            return
        if first.get("status") != "completed":
            save(
                folder / "repeat-manifest.json",
                {
                    "status": "stopped_after_initial_noncompletion",
                    "variant": args.variant,
                    "query": args.query,
                    "initial": first,
                    "second_run": None,
                    "average_seconds": None,
                    "recorded_unix": time.time(),
                },
            )
            save(
                run / "status.json",
                {
                    "phase": "selection_finished",
                    "reason": "Initial query did not complete; second run cancelled",
                    "updated_unix": time.time(),
                },
            )
            render(run)
            print("STOP initial non-completion; no warm rerun", flush=True)
            return

        # Keep the first complete artifact outside attempt-1 because the
        # second rerun deliberately replaces that directory.
        repetitions = run / "repetitions" / args.variant / args.query / "initial"
        if repetitions.exists():
            shutil.rmtree(repetitions)
        shutil.copytree(first_folder, repetitions)
        save(
            run / "status.json",
            {
                "phase": "repeat_pending",
                "variant": args.variant,
                "query_id": args.query,
                "reason": "Initial run completed; warm rerun retains cache",
                "updated_unix": time.time(),
            },
        )
        render(run)
        second_folder, second = run_once(
            root, run, args.variant, args.query, warm=True, log=log
        )
        if second.get("status") != "completed":
            save(
                folder / "repeat-manifest.json",
                {
                    "status": "warm_rerun_noncompletion",
                    "variant": args.variant,
                    "query": args.query,
                    "initial": {
                        "status": first.get("status"),
                        "seconds": first.get("seconds"),
                    },
                    "second_run": second,
                    "average_seconds": None,
                    "cache_reset_before_initial": True,
                    "cache_reset_before_second": False,
                    "recorded_unix": time.time(),
                },
            )
            render(run)
            print("WARM RERUN DID NOT COMPLETE", flush=True)
            return
        average = (float(first["seconds"]) + float(second["seconds"])) / 2
        manifest = {
            "status": "completed_two_runs",
            "variant": args.variant,
            "query": args.query,
            "initial": {
                "status": first.get("status"),
                "seconds": first.get("seconds"),
                "artifact": str(repetitions.relative_to(run)),
            },
            "second_run": {
                "status": second.get("status"),
                "seconds": second.get("seconds"),
                "artifact": ".",
            },
            "average_seconds": average,
            "cache_reset_before_initial": True,
            "cache_reset_before_second": False,
            "recorded_unix": time.time(),
        }
        save(folder / "repeat-manifest.json", manifest)
        save(
            folder / "summary.json",
            {
                **second,
                "repeat_runs": 2,
                "initial_seconds": first.get("seconds"),
                "second_seconds": second.get("seconds"),
                "average_seconds": average,
                "repeat_manifest": str(folder / "repeat-manifest.json"),
            },
        )
        render(run)
        print("DONE two runs; average seconds", average, flush=True)


if __name__ == "__main__":
    main()
