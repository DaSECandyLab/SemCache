"""CLI entrypoint for system comparison campaign execution."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import time
import traceback
from pathlib import Path

from experiments.core.io import save
from experiments.system_comparison.reporting.monitor import render

from .campaign_setup import (
    cleanup_owned_processes,
    execution_fingerprints,
    initialize,
    prepare,
    variants,
)


def main():
    from .campaign import execute

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=[item["name"] for item in variants()],
        help="Run only one configuration in a fresh result directory",
    )
    parser.add_argument(
        "--workload",
        choices=["baseline19", "all", "all20", "all28"],
        default="baseline19",
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
