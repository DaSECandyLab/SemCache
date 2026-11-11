"""Pool real response latencies by model and semantic operator kind."""

import math

REQUEST_TIMING_PROTOCOL = 2


def merge_request_profiles(profile, phase):
    """Merge a once-owned phase's moments; cache hits are excluded upstream."""
    for model, kinds in phase.get("models", {}).items():
        bank = profile.setdefault("request_profiles", {}).setdefault(model, {})
        for kind, group in kinds.items():
            incoming = group["response_latency"]
            n = incoming["samples"]
            if not n:
                continue
            mean = incoming["seconds_per_response"]
            m2 = incoming["latency_m2"]
            previous = bank.get(kind, {})
            old_n = previous.get("samples", 0)
            old_mean = previous.get("seconds_per_request", 0.0)
            old_m2 = previous.get("latency_m2", 0.0)
            if any(
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                for value in (n, mean, m2, old_n, old_mean, old_m2)
            ):
                raise ValueError("Invalid observed request-time moments")
            total = old_n + n
            delta = mean - old_mean
            combined_mean = old_mean + delta * n / total
            combined_m2 = old_m2 + m2 + delta * delta * old_n * n / total
            bank[kind] = {
                "samples": total,
                "seconds_per_request": combined_mean,
                "latency_m2": combined_m2,
                "latency_cv": math.sqrt(max(0.0, combined_m2) / (total - 1))
                / combined_mean
                if total > 1 and combined_mean > 0
                else 0.0,
                "basis": "counted initial and online atomic request latency",
            }
    profile["request_timing_protocol"] = REQUEST_TIMING_PROTOCOL
