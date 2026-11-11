"""Observed two-objective execution feedback; prediction scores are separate."""

import statistics


def execution_reward(record):
    times = record.get("times", [])
    seconds = statistics.mean(times) if times else None
    accuracy = record.get("accuracy")
    return {
        "vector": [-seconds if seconds is not None else None, accuracy],
        "latency_seconds": seconds,
        "accuracy": accuracy,
        "complete": seconds is not None and accuracy is not None,
        "time_lower_bound": record.get("time_lower_bound"),
        "basis": "negative observed full-plan execution seconds, positive result F1; excludes Beam and sampling",
    }
