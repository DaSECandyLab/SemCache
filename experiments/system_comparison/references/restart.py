"""Restart selected failed references at a query boundary, retaining caches."""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path

from experiments.core.io import digest, save
from experiments.core.paths import ROOT
from experiments.core.process import alive
from experiments.system_comparison.references.campaign import SYSTEMS
from experiments.system_comparison.scheduling.campaign import cleanup_owned_processes


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--retry-system", choices=SYSTEMS, action="append", required=True
    )
    parser.add_argument("--retry-query", action="append")
    parser.add_argument("--after-system", choices=SYSTEMS)
    args = parser.parse_args()
    run = args.run.resolve()
    with (run / "restart.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = read(run / "controller.json")
        pid = original["pid"]
        contract = read(run / "contract.json")
        selected = set(args.retry_query) if args.retry_query is not None else None
        if selected is not None:
            allowed = {
                system + "/" + query
                for system in args.retry_system
                for query in contract["query_ids"]
            }
            if invalid := selected - allowed:
                raise ValueError("Invalid retry queries: " + ", ".join(sorted(invalid)))
        queries = []
        retained = {}
        for system in SYSTEMS:
            for query in contract["query_ids"]:
                folder = run / system / query
                summary = (
                    read(folder / "summary.json")
                    if (folder / "summary.json").exists()
                    else {}
                )
                if (
                    system in args.retry_system
                    and (selected is None or system + "/" + query in selected)
                    and not (
                        summary.get("parent_finalized")
                        and summary.get("status") in {"completed", "skipped"}
                    )
                ):
                    queries.append(system + "/" + query)
                elif (folder / "answer.json").exists():
                    retained[str((folder / "answer.json").relative_to(run))] = digest(
                        folder / "answer.json"
                    )
        request = {
            "phase": "pending",
            "controller_pid": pid,
            "nonce": uuid.uuid4().hex,
            "scheduler_pid": os.getpid(),
            "retry_systems": args.retry_system,
            "queries": queries,
            "after_system": args.after_system,
            "continue_systems": list(SYSTEMS[: SYSTEMS.index(args.after_system) + 1])
            if args.after_system
            else [],
            "requested_unix": time.time(),
        }
        save(run / "restart-request.json", request)
        print("QUEUED", json.dumps(queries), flush=True)
        while alive(pid):
            latest_request = read(run / "restart-request.json")
            if (
                latest_request.get("nonce") != request["nonce"]
                or latest_request.get("phase") != "pending"
            ):
                return
            current = read(run / "controller.json")
            if current["pid"] != pid or current.get("phase") == "stopped_by_user":
                request.update(
                    phase="cancelled", reason="Collector changed or was stopped by user"
                )
                save(run / "restart-request.json", request)
                return
            ready_path = run / "restart-ready.json"
            if ready_path.exists():
                ready = read(ready_path)
                if (
                    ready.get("nonce") == request["nonce"]
                    and ready.get("controller_pid") == pid
                    and ready.get("execution_started") is False
                ):
                    break
            time.sleep(0.2)
        if args.after_system:
            pending = []
            for query in contract["query_ids"]:
                path = run / args.after_system / query / "summary.json"
                if not path.exists() or not read(path).get("parent_finalized"):
                    pending.append(query)
            if pending:
                request.update(
                    phase="failed",
                    reason="Requested system has not finished",
                    pending_queries=pending,
                )
                save(run / "restart-request.json", request)
                raise RuntimeError("Cannot retry before the requested system finishes")
        if alive(pid):
            command = (
                Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            )
            if (
                "experiments.system_comparison.references" not in command
                or str(run) not in command
            ):
                raise RuntimeError("Collector identity changed before handoff")
            os.kill(pid, signal.SIGTERM)
            for _ in range(50):
                if not alive(pid):
                    break
                time.sleep(0.1)
            if alive(pid):
                raise RuntimeError("Old collector did not exit")
        terminated = cleanup_owned_processes(run)
        ready_path = run / "restart-ready.json"
        if ready_path.exists():
            ready = read(ready_path)
            if ready.get("nonce") == request["nonce"]:
                status_path = run / "status" / (ready["system"] + ".json")
                if status_path.exists():
                    status = read(status_path)
                    if status.get("worker_pid") == ready["worker_pid"]:
                        status.update(
                            phase="waiting_for_retry", execution_started=False
                        )
                        save(status_path, status)
        # Check the collector lock is released before creating its successor.
        with (run / "controller.lock").open("a") as controller_lock:
            fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request.update(phase="starting", terminated_workers=terminated)
        request["source_sha256"] = {
            name: digest(ROOT / name)
            for name in (
                "experiments/system_comparison/references/campaign.py",
                "experiments/system_comparison/references/worker.py",
                "experiments/system_comparison/references/protocol.py",
                "experiments/system_comparison/transport/gateway.py",
                "experiments/system_comparison/transport/upstream.py",
            )
        }
        save(run / "restart-request.json", request)
        changed = [name for name, sha in retained.items() if digest(run / name) != sha]
        if changed:
            raise RuntimeError("Previously completed answers changed: " + str(changed))
        launch = read(run / "launch.json")
        command = launch["command"]
        for flag in ("--retry-system", "--retry-query"):
            while flag in command:
                index = command.index(flag)
                del command[index : index + 2]
        for system in args.retry_system:
            command.extend(["--retry-system", system])
        if selected is not None:
            for query in sorted(selected):
                command.extend(["--retry-query", query])
        if "--allow-runtime-update" not in command:
            command.append("--allow-runtime-update")
        with (run / "generator.log").open("ab", buffering=0) as output:
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT)},
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        launch.update(
            command=command,
            pid=child.pid,
            started_unix=time.time(),
            retry_systems=args.retry_system,
            retry_queries=queries,
            resume_from=queries[0] if queries else None,
        )
        save(run / "launch.json", launch)
        (run / "launcher.pid").write_text(str(child.pid) + "\n")
        request.update(
            phase="started",
            new_controller_pid=child.pid,
            started_unix=time.time(),
            retained_completed_answers=len(retained),
            changed_completed_answers=[],
        )
        save(run / "restart-request.json", request)
        print("STARTED", child.pid, flush=True)


if __name__ == "__main__":
    main()
