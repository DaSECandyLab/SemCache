"""Execution budgets and diagnostic extrapolation; never manufacture labels."""

import math
from dataclasses import dataclass


@dataclass
class TimeoutPolicy:
    maximum: float = 1800.0
    minimum: float = 60.0
    slack: float = 2.0

    def __post_init__(self):
        if (
            not all(
                math.isfinite(v) and v > 0
                for v in (self.maximum, self.minimum, self.slack)
            )
            or self.minimum > self.maximum
            or self.slack < 1
        ):
            raise ValueError("Invalid execution timeout budget")

    def threshold(self, completed=()):
        times = [
            v
            for v in completed
            if type(v) in (int, float) and math.isfinite(v) and v >= 0
        ]
        return (
            min(self.maximum, max(self.minimum, self.slack * min(times)))
            if times
            else self.maximum
        )


def execution_progress(events, expected=None, elapsed=0.0):
    """Track processed inputs separately from physical model requests.

    Estimated remaining serial request work is diagnostic only. It excludes
    unknown relational/blocking work and cannot act as an execution label.
    """
    expected = expected or {}
    rows = {}
    for event in events:
        if (
            event.get("event")
            not in (
                "semantic_row",
                "semantic_result_reuse",
                "semantic_result_cache_hit",
            )
            or event.get("parse_status") != "validated"
        ):
            continue
        oid = str(event["operator_id"])
        row = rows.setdefault(
            oid,
            {
                "completed_requests": 0,
                "completed_inputs": 0,
                "result_reuses": 0,
                "emitted_rows": 0,
                "request_seconds": 0.0,
            },
        )
        row["completed_inputs"] += event.get("sampling_batch_size", 1)
        if event["event"] == "semantic_row":
            row["completed_requests"] += 1
            row["request_seconds"] += event.get("latency_ms", 0) / 1000
        else:
            row["result_reuses"] += 1
        row["emitted_rows"] += int(bool(event.get("emitted")))
    remaining = 0.0
    known = bool(expected)
    for oid in set(rows) | set(expected):
        row = rows.setdefault(
            oid,
            {
                "completed_requests": 0,
                "completed_inputs": 0,
                "result_reuses": 0,
                "emitted_rows": 0,
                "request_seconds": 0.0,
            },
        )
        prior = expected.get(oid, {})
        count, prior_latency = prior.get("input_rows"), prior.get("seconds_per_request")
        row["expected_input_rows"] = count
        row["observed_seconds_per_request"] = (
            row["request_seconds"] / row["completed_requests"]
            if row["completed_requests"]
            else None
        )
        latency = (
            row["request_seconds"] / row["completed_inputs"]
            if row["completed_inputs"]
            else prior_latency
        )
        if count is None or latency is None:
            known = False
        else:
            remaining += max(0, count - row["completed_inputs"]) * latency
    return {
        "operators": rows,
        "estimated_serial_semantic_completion_seconds": elapsed + remaining
        if known
        else None,
        "estimate_is_label": False,
        "basis": "estimated input cardinalities and observed request work per input; excludes unknown relational work",
    }
