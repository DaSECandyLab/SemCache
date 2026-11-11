"""Observed response timings and optional updates to global model/kind means."""

import math

from planning.state import digest

LAYOUT_FIELDS = ("block_order", "field_order", "left_field_order", "right_field_order")
HISTOGRAM_BOUNDS = (0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128)


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def add_moment(entry, value):
    n = entry.get("samples", 0) + 1
    mean = entry.get("seconds_per_response", 0.0)
    delta = value - mean
    mean += delta / n
    m2 = entry.get("latency_m2", 0.0) + delta * (value - mean)
    entry.update(
        samples=n,
        seconds_per_response=mean,
        latency_m2=m2,
        stddev_seconds=math.sqrt(max(0.0, m2) / (n - 1)) if n > 1 else None,
        min_seconds=min(entry.get("min_seconds", value), value),
        max_seconds=max(entry.get("max_seconds", value), value),
    )
    histogram = entry.setdefault(
        "histogram",
        {
            "upper_bounds_seconds": list(HISTOGRAM_BOUNDS),
            "counts": [0] * (len(HISTOGRAM_BOUNDS) + 1),
            "last_bin": "above final bound",
        },
    )
    index = next(
        (i for i, bound in enumerate(HISTOGRAM_BOUNDS) if value <= bound),
        len(HISTOGRAM_BOUNDS),
    )
    histogram["counts"][index] += 1


def response(event):
    lo, hi = event.get("submit_ns"), event.get("response_ns")
    if lo == 0 and hi == 0:
        return None, None  # Serving answer-cache sentinel, not a physical request.
    if type(lo) is int and type(hi) is int and lo > 0 and hi >= lo:
        return (hi - lo) / 1e9, (lo, hi)
    latency = event.get("latency_ms")
    return (latency / 1000, None) if number(latency) else (None, None)


def union_seconds(intervals):
    merged = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return sum(hi - lo for lo, hi in merged) / 1e9


def context(event, *, source, scope):
    layout = {key: event.get(key) for key in LAYOUT_FIELDS}
    cached = event.get("cached_input_tokens")
    cache = (
        "uncached"
        if number(cached) and cached == 0
        else ("cached" if number(cached) else "unknown")
    )
    complete = all(isinstance(layout[key], list) for key in LAYOUT_FIELDS)
    result = {
        "source": source,
        "layout": layout,
        "cache_condition": cache,
        "sampling_batch_size": event.get("sampling_batch_size", 1),
        "prompt_serialization_version": event.get("prompt_serialization_version"),
        "sampling_prompt_version": event.get("sampling_prompt_version"),
    }
    if cache == "unknown" or not complete:
        result["unknown_condition_scope"] = scope
    return result


def record_phase(
    profile,
    events,
    *,
    source,
    scope,
    wall_seconds=None,
    wall_basis=None,
    merge_costs=False,
):
    """Add one phase. Unknown cache/layout cohorts stay in its local artifact.

    Global buckets distinguish model/kind/source/full layout/batch/cache class.
    Cached means a positive reported prefix count, not a fixed cache-hit ratio.
    When merge_costs is set, also pool physical responses into the model/kind
    cost mean. Native sampling already updates that mean and leaves it unset.
    """
    summary = {
        "source": source,
        "scope": scope,
        "requests": 0,
        "reported_responses": 0,
        "interval_responses": 0,
        "response_latency": {},
        "contexts": {},
        "models": {},
        "phase_wall_seconds": None,
        "phase_wall_basis": wall_basis,
    }
    intervals, seen, group_intervals = [], set(), {}
    physical_submits = scheduler_submits = 0
    for event in events:
        if event.get("event") in ("sampling_llm_submit", "sampling_llm_retransmit"):
            physical_submits += 1
        elif event.get("event") == "submit":
            scheduler_submits += 1
        if event.get("event") != "semantic_row":
            continue
        identity = (
            event.get("query_id"),
            event.get("operator_id"),
            event.get("request_seq"),
        )
        if identity[-1] is not None:
            if identity in seen:
                continue
            seen.add(identity)
        value, interval = response(event)
        if value is None:
            continue
        summary["reported_responses"] += 1
        add_moment(summary["response_latency"], value)
        if interval is not None:
            summary["interval_responses"] += 1
            intervals.append(interval)
        model, kind = event.get("model_id"), event.get("operator_kind")
        if not model or not kind:
            continue
        kind = kind.upper().replace(" ", "_")
        group = (
            summary["models"]
            .setdefault(model, {})
            .setdefault(
                kind,
                {
                    "response_latency": {},
                    "interval_responses": 0,
                },
            )
        )
        add_moment(group["response_latency"], value)
        if interval is not None:
            group["interval_responses"] += 1
            group_intervals.setdefault((model, kind), []).append(interval)
        cohort = context(event, source=source, scope=scope)
        key = digest([model, kind, cohort])
        entry = summary["contexts"].setdefault(
            key,
            {
                "model_id": model,
                "operator_kind": kind,
                "context": cohort,
            },
        )
        add_moment(entry, value)
        global_entry = (
            profile.setdefault("online_request_profiles", {})
            .setdefault(model, {})
            .setdefault(kind, {"samples": 0, "sources": {}})
        )
        global_entry["samples"] += 1
        global_entry["response_seconds_sum"] = (
            global_entry.get("response_seconds_sum", 0.0) + value
        )
        global_entry["basis"] = (
            "descriptive response mixture; never a cost/service-time prior"
        )
        bank = global_entry["sources"].setdefault(
            source,
            {
                "samples": 0,
                "unclassified_samples": 0,
                "contexts": {},
            },
        )
        bank["samples"] += 1
        if "unknown_condition_scope" in cohort:
            bank["unclassified_samples"] += 1
        else:
            bucket = bank["contexts"].setdefault(digest(cohort), {"context": cohort})
            add_moment(bucket, value)
    active = union_seconds(intervals) if intervals else None
    summary["request_interval_union_seconds"] = active
    count = summary["interval_responses"]
    wall = wall_seconds() if callable(wall_seconds) else wall_seconds
    wall = wall if number(wall) else None
    summary["phase_wall_seconds"] = wall
    summary["responses_per_active_second"] = count / active if active else None
    summary["responses_per_wall_second"] = (
        summary["reported_responses"] / wall if wall else None
    )
    summary["requests"] = (
        physical_submits
        if physical_submits
        else (scheduler_submits if scheduler_submits else None)
    )
    summary["request_count_source"] = (
        "physical HTTP sends including retries"
        if physical_submits
        else "scheduler submit records"
        if scheduler_submits
        else "unknown"
    )
    for model, kinds in summary["models"].items():
        for kind, group in kinds.items():
            spans = group_intervals.get((model, kind), [])
            active = union_seconds(spans) if spans else None
            group["request_interval_union_seconds"] = active
            group["phase_wall_seconds"] = wall
            group["responses_per_active_second"] = (
                group["interval_responses"] / active if active else None
            )
            group["responses_per_wall_second"] = (
                group["response_latency"]["samples"] / wall if wall else None
            )
            totals = profile["online_request_profiles"][model][kind]
            if active is not None:
                totals["interval_responses"] = (
                    totals.get("interval_responses", 0) + group["interval_responses"]
                )
                totals["phase_interval_union_seconds_sum"] = (
                    totals.get("phase_interval_union_seconds_sum", 0.0) + active
                )
            if wall is not None:
                totals["phase_wall_responses"] = (
                    totals.get("phase_wall_responses", 0)
                    + group["response_latency"]["samples"]
                )
                totals["phase_wall_seconds_sum"] = (
                    totals.get("phase_wall_seconds_sum", 0.0) + wall
                )
    summary["basis"] = (
        "completed response latency includes queue/overlap; phase mean is a descriptive mixture; "
        "active throughput uses responses with intervals in this phase only; "
        "wall throughput uses all reported responses; no concurrency division or service-time inference"
    )
    if merge_costs:
        from .request_profiles import merge_request_profiles

        merge_request_profiles(profile, summary)
    return summary


def merge_moments(target, incoming):
    """Combine a saved descriptive cohort without replaying individual requests."""
    n, k = target.get("samples", 0), incoming.get("samples", 0)
    if not k:
        return
    mean, other = (
        target.get("seconds_per_response", 0.0),
        incoming["seconds_per_response"],
    )
    size = n + k
    delta = other - mean
    m2 = (
        target.get("latency_m2", 0.0)
        + incoming.get("latency_m2", 0.0)
        + delta * delta * n * k / size
    )
    target.update(
        samples=size,
        seconds_per_response=(mean * n + other * k) / size,
        latency_m2=m2,
        stddev_seconds=math.sqrt(max(0.0, m2) / (size - 1)) if size > 1 else None,
        min_seconds=min(
            target.get("min_seconds", incoming["min_seconds"]), incoming["min_seconds"]
        ),
        max_seconds=max(
            target.get("max_seconds", incoming["max_seconds"]), incoming["max_seconds"]
        ),
    )
    histogram = target.setdefault(
        "histogram",
        {
            "upper_bounds_seconds": list(HISTOGRAM_BOUNDS),
            "counts": [0] * (len(HISTOGRAM_BOUNDS) + 1),
            "last_bin": "above final bound",
        },
    )
    histogram["counts"] = [
        a + b
        for a, b in zip(
            histogram["counts"], incoming["histogram"]["counts"], strict=True
        )
    ]


def merge_phase(profile, summary):
    """Rebuild a history summary from saved phase moments after an interruption."""
    source = summary["source"]
    for model, kinds in summary.get("models", {}).items():
        for kind, group in kinds.items():
            entry = (
                profile.setdefault("online_request_profiles", {})
                .setdefault(model, {})
                .setdefault(
                    kind,
                    {
                        "samples": 0,
                        "sources": {},
                        "basis": "descriptive response mixture; never a cost/service-time prior",
                    },
                )
            )
            latency = group["response_latency"]
            entry["samples"] += latency["samples"]
            entry["response_seconds_sum"] = (
                entry.get("response_seconds_sum", 0.0)
                + latency["seconds_per_response"] * latency["samples"]
            )
            active, wall = (
                group.get("request_interval_union_seconds"),
                group.get("phase_wall_seconds"),
            )
            if active is not None:
                entry["interval_responses"] = (
                    entry.get("interval_responses", 0) + group["interval_responses"]
                )
                entry["phase_interval_union_seconds_sum"] = (
                    entry.get("phase_interval_union_seconds_sum", 0.0) + active
                )
            if wall is not None:
                entry["phase_wall_responses"] = (
                    entry.get("phase_wall_responses", 0) + latency["samples"]
                )
                entry["phase_wall_seconds_sum"] = (
                    entry.get("phase_wall_seconds_sum", 0.0) + wall
                )
            bank = entry["sources"].setdefault(
                source, {"samples": 0, "unclassified_samples": 0, "contexts": {}}
            )
            bank["samples"] += latency["samples"]
    for cohort in summary.get("contexts", {}).values():
        context = cohort["context"]
        bank = profile["online_request_profiles"][cohort["model_id"]][
            cohort["operator_kind"]
        ]["sources"][source]
        if "unknown_condition_scope" in context:
            bank["unclassified_samples"] += cohort["samples"]
        else:
            target = bank["contexts"].setdefault(digest(context), {"context": context})
            merge_moments(target, cohort)
