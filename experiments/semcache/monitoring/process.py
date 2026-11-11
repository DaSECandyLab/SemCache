"""Verified pidfd handle for supervising an existing training controller."""

import json
import os
import select
import subprocess
from pathlib import Path


class AttachedTraining:
    """The poll/wait interface used by the experiment training launcher, without becoming a child."""

    def __init__(self, run, root, python):
        self.run = Path(run).resolve()
        self.root = Path(root).resolve()
        self.python = Path(python).resolve()
        status = self._status()
        self.pid = status.get("pid")
        if (
            type(self.pid) is not int
            or self.pid <= 1
            or self.pid == os.getpid()
            or status.get("status") != "running"
        ):
            raise ValueError("Attach requires a running pipeline with a valid PID")
        self.args = ["existing planning benchmark", "--run", str(self.run)]
        self.returncode = None
        self.fd = os.pidfd_open(self.pid)
        try:
            self.started = self._stat()[19]
            self.validate()
        except BaseException:
            self.close()
            raise

    def _status(self):
        return json.loads((self.run / "pipeline-status.json").read_text())

    def _stat(self):
        return Path(f"/proc/{self.pid}/stat").read_text().rsplit(")", 1)[1].split()

    def validate(self):
        """Reject unrelated/reused PIDs before assuming supervision or signaling."""
        base = Path(f"/proc/{self.pid}")
        stat = self._stat()
        command = [
            os.fsdecode(arg)
            for arg in (base / "cmdline").read_bytes().split(b"\0")
            if arg
        ]
        try:
            module = command.index("-m")
            run_option = command.index("--run")
            matches_command = (
                command[module + 1 : module + 3] == ["planning", "benchmark"]
                and command.count("-m") == command.count("--run") == 1
                and run_option > module + 2
                and (self.root / command[run_option + 1]).resolve() == self.run
            )
        except (ValueError, IndexError):
            matches_command = False
        status = self._status()
        if (
            stat[0] in ("Z", "X")
            or stat[19] != self.started
            or int(stat[2]) != self.pid
            or int(stat[3]) != self.pid
            or base.stat().st_uid != os.geteuid()
            or (base / "cwd").resolve() != self.root
            or (base / "exe").resolve() != self.python
            or not matches_command
            or status.get("pid") != self.pid
            or int((self.run / "pipeline.pid").read_text()) != self.pid
            or self.poll() is not None
        ):
            raise ValueError(
                f"Refusing to attach/signal an unverified pipeline PID {self.pid}"
            )

    def _finished(self):
        try:
            status = self._status()
        except (OSError, ValueError):
            status = {}
        self.returncode = (
            {"complete": 0, "stopped": 130}.get(status.get("status"), 1)
            if status.get("pid") == self.pid
            else 1
        )
        return self.returncode

    def poll(self):
        if self.returncode is None and select.select([self.fd], [], [], 0)[0]:
            return self._finished()
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is not None:
            return self.returncode
        if not select.select([self.fd], [], [], timeout)[0]:
            raise subprocess.TimeoutExpired(self.args, timeout)
        return self._finished()

    def signal_group(self, signum):
        if self.poll() is None:
            self.validate()
            os.killpg(self.pid, signum)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
