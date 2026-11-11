"""Read Sema's published native profiling counters (milliseconds, not tokens)."""

from __future__ import annotations

import json


def phases(profile_path, system, *, execution_wall_seconds, planning_seconds):
    profile = json.loads(profile_path.read_text())
    operators = []
    format_failures = 0

    def walk(node):
        nonlocal format_failures
        extra = node.get("extra_info") or {}
        if "SEM_" in str(node.get("operator_name", "")):
            format_failures += int(extra.get("parse_failure", 0))
        if node.get("operator_name") == "SEM_FILTER_MULTIPLEXER":
            keys = (
                "expression_exploration_time_ms",
                "path_exploration_time_ms",
                "exploitation_time_ms",
            )
            if all(name in extra for name in keys):
                operators.append(
                    {
                        "sampling_seconds": (
                            float(extra[keys[0]]) + float(extra[keys[1]])
                        )
                        / 1000.0,
                        "execution_seconds": float(extra[keys[2]]) / 1000.0,
                        "expression_samples": extra.get(
                            "expression_exploration_samples"
                        ),
                        "path_samples": extra.get("path_exploration_samples"),
                        "native_fields": {name: extra[name] for name in keys},
                    }
                )
            else:
                operators.append({"sampling_seconds": None, "native_fields": extra})
        for child in node.get("children", []):
            walk(child)

    walk(profile)
    unavailable = {}
    if system == "plop" or not operators:
        sampling = 0.0
        execution = execution_wall_seconds
    elif len(operators) == 1 and operators[0]["sampling_seconds"] is not None:
        sampling = operators[0]["sampling_seconds"]
        if sampling > execution_wall_seconds:
            sampling = None
            execution = None
            unavailable["sampling_seconds"] = (
                "Native exploration counters exceed query wall time; retained per operator"
            )
            unavailable["execution_seconds"] = (
                "Native phase clocks cannot be subtracted from query wall time"
            )
        else:
            execution = max(0.0, execution_wall_seconds - sampling)
    else:
        sampling = None
        execution = None
        unavailable["sampling_seconds"] = (
            "Multiple native AQE operators may overlap; per-operator phase counters retained"
        )
        unavailable["execution_seconds"] = (
            "No global execution boundary for overlapping native AQE operators"
        )
    return {
        "sampling_seconds": sampling,
        "planning_seconds": planning_seconds,
        "execution_seconds": execution,
        "load_seconds": None,
        "host_seconds": None,
        "native_format_fallbacks": format_failures,
        "unavailable": unavailable,
        "details": {
            "native_aqe_operators": operators,
            "execute_wall_seconds": execution_wall_seconds,
        },
        "method": "separate PREPARE wall clock; Sema native exploration milliseconds; remaining EXECUTE wall clock",
    }
