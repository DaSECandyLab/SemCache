"""Phase cost accounting, separate from query execution feedback labels."""

import time
from contextlib import contextmanager
from typing import Any


class PhaseTimings:
    def __init__(self, on_checkpoint=None):
        self.values: dict[str, dict[str, Any]] = {}
        self._last = time.perf_counter(), time.process_time()
        self.on_checkpoint = on_checkpoint

    def _add(self, name, started, ended):
        row = self.values.setdefault(
            name, {"wall_seconds": 0.0, "cpu_seconds": 0.0, "calls": 0}
        )
        row["wall_seconds"] += ended[0] - started[0]
        row["cpu_seconds"] += ended[1] - started[1]
        row["calls"] += 1
        row.setdefault("intervals_ns", []).append(
            [round(started[0] * 1e9), round(ended[0] * 1e9)]
        )

    def checkpoint(self, name):
        ended = time.perf_counter(), time.process_time()
        self._add(name, self._last, ended)
        self._last = ended
        if self.on_checkpoint is not None:
            self.on_checkpoint(self.values)

    @contextmanager
    def measure(self, name):
        started = time.perf_counter(), time.process_time()
        try:
            yield
        finally:
            self._add(name, started, (time.perf_counter(), time.process_time()))
