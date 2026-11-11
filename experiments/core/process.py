"""Run isolated subprocess groups and preserve timeout/error evidence."""

import os
import signal
import subprocess
import threading
import time
from pathlib import Path


def process(command, folder, timeout, *, stdin=None, env=None, heartbeat=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    input_path = folder / "stdin.sql"
    if stdin is not None:
        input_path.write_text(stdin)
    with (
        (folder / "stdout.log").open("w") as stdout,
        (folder / "stderr.log").open("w") as stderr,
        input_path.open() if stdin is not None else open(os.devnull) as input_stream,
    ):
        child = subprocess.Popen(
            command,
            cwd=folder,
            env=env,
            stdin=input_stream,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        timed_out = False
        try:
            while child.poll() is None:
                elapsed = time.monotonic() - started
                if heartbeat:
                    heartbeat(elapsed, child.pid)
                if timeout is not None and elapsed >= timeout:
                    timed_out = True
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                    break
                try:
                    child.wait(
                        timeout=5
                        if timeout is None
                        else max(0.001, min(5, timeout - elapsed))
                    )
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
    return {
        "status": "timeout"
        if timed_out
        else "completed"
        if child.returncode == 0
        else "failed",
        "returncode": child.returncode,
        "seconds": time.monotonic() - started,
        "timed_out": timed_out,
        "timeout_seconds": timeout,
    }


def alive(pid):
    try:
        return (
            Path(f"/proc/{int(pid)}/stat").read_text().split(") ", 1)[1].split()[0]
            != "Z"
        )
    except (OSError, TypeError, ValueError, IndexError):
        return False


def stop_stage(child, folder=None):
    """Stop this driver's descendants even when native executors use setsid()."""
    processes = {}
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().split(") ", 1)[1].split()
            processes[int(path.parent.name)] = (int(fields[1]), int(fields[2]))
        except (OSError, ValueError, IndexError):
            continue
    owned = {child.pid}
    if folder is not None:
        folder = Path(folder).resolve()
        for pid in processes:
            try:
                if Path(f"/proc/{pid}/cwd").resolve(strict=True).is_relative_to(folder):
                    owned.add(pid)
            except (OSError, RuntimeError):
                pass
    while True:
        descendants = {pid for pid, (parent, _) in processes.items() if parent in owned}
        if descendants <= owned:
            break
        owned.update(descendants)
    groups = (
        {child.pid} | {processes[pid][1] for pid in owned if pid in processes}
    ) - {os.getpgrp()}
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for group in groups:
            try:
                os.killpg(group, sig)
            except ProcessLookupError:
                pass
        if sig == signal.SIGTERM:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    child.wait(timeout=5)
    # Grandchildren are reaped by their owner or init, not by this controller.

    end = time.monotonic() + 5
    while any(alive(pid) for pid in owned) and time.monotonic() < end:
        threading.Event().wait(0.01)
