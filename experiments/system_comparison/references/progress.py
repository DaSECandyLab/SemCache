"""Live reference accounting without restarting native query execution."""

import argparse
import fcntl
import json
import os
import time
from functools import lru_cache
from pathlib import Path

from experiments.core.process import alive

COUNTERS = (
    "upstream_http_attempts",
    "cache_hits",
    "cache_verifications",
    "cache_vote_calls",
)


def read(path):
    return json.loads(path.read_text())


@lru_cache(maxsize=5)
def checkpoint_totals(run, system):
    # Checkpoints are immutable while one collector owns the directory.
    totals = dict.fromkeys(COUNTERS, 0)
    for path in (run / "checkpoints").glob(f"*/{system}/*/*/gateway.jsonl"):
        with path.open() as stream:
            for line in stream:
                if not line.endswith("\n"):
                    continue
                record = json.loads(line)
                if record["event"] not in {"completed", "error"}:
                    continue
                totals["upstream_http_attempts"] += record.get(
                    "upstream_attempts", int(record["event"] == "completed")
                )
                cache = record.get("cache", {})
                totals["cache_hits"] += int(cache.get("hit", False))
                totals["cache_verifications"] += cache.get("verification_calls", 0)
                totals["cache_vote_calls"] += cache.get("vote_calls", 0)
    return totals


def request_totals(run, system):
    totals = dict(checkpoint_totals(run, system))
    finalized = set()
    for path in (run / system).glob("*/*/summary.json"):
        record = read(path)
        if record.get("parent_finalized"):
            finalized.add(str(path.parent.relative_to(run / system)))
            for key in COUNTERS:
                totals[key] += record.get(key, 0)
    status_path = run / "status" / (system + ".json")
    status = read(status_path) if status_path.exists() else {}
    if status.get("phase") == "running" and status.get("query") not in finalized:
        for key in COUNTERS:
            totals[key] += status.get(key, 0)
    return totals


def write_progress(run, text, snapshot):
    for name, content in (
        ("progress.md", text),
        ("progress.json", json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n"),
    ):
        target = run / name
        temporary = run / f".{name}.{os.getpid()}.tmp"
        temporary.write_text(content)
        temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    run = parser.parse_args().run.resolve()
    from experiments.system_comparison.references.campaign import render

    with (run / "progress-render.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        controller_pid = read(run / "controller.json")["pid"]
        last_revision = None
        while True:
            controller = read(run / "controller.json")
            revision = (run / "progress.md").stat().st_mtime_ns
            finished = (
                controller["pid"] != controller_pid
                or controller.get("phase") == "finished"
                or not alive(controller_pid)
            )
            if revision != last_revision or finished:
                render(run, read(run / "contract.json"))
                last_revision = (run / "progress.md").stat().st_mtime_ns
            if finished:
                return
            time.sleep(0.25)


if __name__ == "__main__":
    main()
