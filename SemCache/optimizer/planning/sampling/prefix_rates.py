"""Use measured depth rates, falling back to the whole-field operator rate."""

import math


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _overall_rate(info, field):
    # Join fields describe source-row matches, not pair acceptance. Their
    # measured depth-zero rate is the matching whole-field population rate.
    for name in ("depth_selectivity_statistics", "prefix_selectivity_statistics"):
        entry = info.get(name, {}).get(field, {}).get("0", {})
        if finite(entry.get("rate")) and (
            entry.get("observed", 0) > 0
            or name == "prefix_selectivity_statistics"
            and entry.get("source") in ("operator_scalar", "exact", "empty_population")
        ):
            return entry["rate"]
    return info.get("selectivity")


def prefix_rate(info, field, depth):
    """Ignore historical interpolation and use the operator's overall rate."""
    for name in ("depth_selectivity_statistics", "prefix_selectivity_statistics"):
        entry = info.get(name, {}).get(field, {}).get(str(depth), {})
        if finite(entry.get("rate")) and (
            entry.get("observed", 0) > 0
            or name == "prefix_selectivity_statistics"
            and entry.get("source") in ("exact", "empty_population")
        ):
            return entry["rate"]
    scalar = _overall_rate(info, field)
    return scalar if finite(scalar) else None


def complete_prefix_selectivity(info):
    """Publish a separate estimate bank; the measured depth bank stays intact."""
    measured = info.get("depth_selectivity_statistics", {})
    summaries = {
        item["survival_field"]: item["summary"]
        for item in info.get("input_field_summaries", {}).values()
    }
    result = {}
    for field in sorted(measured.keys() | summaries.keys()):
        table = measured.get(field, {})
        summary = summaries.get(field, {})
        sketches = summary.get("sketches", {})
        depths = sorted({0, *map(int, table), *map(int, sketches)})
        scalar = _overall_rate(info, field)
        target = result[field] = {}
        for depth in depths:
            key = str(depth)
            raw = table.get(key, {})
            empty = (
                summary.get("count") == 0
                or depth > 0
                and key in sketches
                and not sketches[key].get("entries")
            )
            entry = {
                "rate": None,
                "observed": 0,
                "effective_samples": 0.0,
                "source": "unknown",
                "anchor_depths": [],
            }
            if finite(raw.get("rate")) and raw.get("observed", 0) > 0:
                entry.update(raw)
                entry.update(source="measured", anchor_depths=[depth])
            elif empty:
                entry.update(rate=0.0, source="empty_population")
            elif finite(scalar):
                entry.update(rate=scalar, source="operator_scalar", anchor_depths=[0])
            target[key] = entry
        if any(entry["source"] == "unknown" for entry in target.values()):
            raise ValueError(
                f"Cannot estimate prefix rates for field {field}: "
                "no finite measured depth or overall operator selectivity"
            )
    info["prefix_selectivity_statistics"] = result
    return result


def local_prefix_selectivity(info):
    """Prefer completed estimates while retaining old measured-only profiles."""
    return info.get(
        "prefix_selectivity_statistics", info.get("depth_selectivity_statistics", {})
    )
