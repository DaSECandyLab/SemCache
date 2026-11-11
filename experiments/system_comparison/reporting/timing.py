"""Wall-clock phase intervals with nested work counted exactly once."""

import time
from contextlib import contextmanager
from contextvars import ContextVar

from experiments.core.io import save

TIMING_SCHEMA = "native-phase-timing-v1"


def write_phase_timing(
    folder,
    *,
    query_seconds,
    load_seconds=None,
    sampling_seconds=None,
    planning_seconds=None,
    execution_seconds=None,
    host_seconds=None,
    unavailable=None,
    method="",
    details=None,
):
    """Persist non-overlapping phase timings.

    ``None`` means the native implementation did not expose that phase. A
    missing phase is never converted to zero, because zero would falsely claim
    that an optimizer or sampler did no work.
    """
    unavailable = dict(unavailable or {})
    value = {
        "schema": TIMING_SCHEMA,
        "query_seconds": float(query_seconds) if query_seconds is not None else None,
        "load_seconds": float(load_seconds) if load_seconds is not None else None,
        "sampling_seconds": float(sampling_seconds)
        if sampling_seconds is not None
        else None,
        "planning_seconds": float(planning_seconds)
        if planning_seconds is not None
        else None,
        "execution_seconds": float(execution_seconds)
        if execution_seconds is not None
        else None,
        "host_seconds": float(host_seconds) if host_seconds is not None else None,
        "unavailable": unavailable,
        "method": method,
        "details": details or {},
    }
    save(folder / "timing.json", value)
    return value


CURRENT = ContextVar("comparison_query_timer", default=None)


class QueryTimer:
    def __init__(self, folder):
        self.folder = folder
        self.intervals = []
        self.unavailable = {}

    def __enter__(self):
        self.parent = CURRENT.get()
        self.start = time.monotonic()
        self.token = CURRENT.set(self)
        return self

    @contextmanager
    def stage(self, phase, label=None):
        start = time.monotonic()
        try:
            yield
        finally:
            self.intervals.append(
                {
                    "phase": phase,
                    "label": label or phase,
                    "start": start - self.start,
                    "end": time.monotonic() - self.start,
                }
            )

    def summary(self):
        total = self.end - self.start
        points = sorted(
            {
                0.0,
                total,
                *[
                    max(0.0, min(total, x[k]))
                    for x in self.intervals
                    for k in ("start", "end")
                ],
            }
        )
        seconds = dict.fromkeys(
            ("sampling", "planning", "execution", "host", "other"), 0.0
        )
        priority = {"sampling": 5, "planning": 4, "host": 3, "execution": 2, "other": 1}
        for a, b in zip(points, points[1:]):
            middle = (a + b) / 2
            active = [
                x["phase"] for x in self.intervals if x["start"] <= middle < x["end"]
            ]
            phase = max(active, key=lambda name: priority[name]) if active else "other"
            seconds[phase] += b - a
        result = {name + "_seconds": value for name, value in seconds.items()}
        result["query_seconds"] = total
        result["phase_intervals"] = self.intervals
        result["unavailable"] = self.unavailable
        for name in self.unavailable:
            result[name + "_seconds"] = None
        result["method"] = (
            "native phase boundaries; exclusive wall time; nested sampling removed from parent phases"
        )
        return result

    def __exit__(self, *exc):
        self.end = time.monotonic()
        CURRENT.reset(self.token)
        self.result = self.summary()
        self.result["schema"] = TIMING_SCHEMA
        save(self.folder / "timing.json", self.result)
        if self.parent is not None:
            offset = self.start - self.parent.start
            self.parent.intervals.extend(
                {
                    **event,
                    "start": event["start"] + offset,
                    "end": event["end"] + offset,
                }
                for event in self.intervals
            )


@contextmanager
def stage(phase, label=None):
    timer = CURRENT.get()
    if timer is None:
        yield
    else:
        with timer.stage(phase, label):
            yield
