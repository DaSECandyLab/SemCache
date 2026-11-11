"""Summarize row F1 against each configuration's own system reference."""

from math import isfinite
from statistics import mean


def _number(value):
    return type(value) in (int, float) and isfinite(value)


def _scored(record):
    value = record.get("accuracy")
    return (
        not record.get("aggregate")
        and record.get("scoring_status") == "scored"
        and _number(value)
        and 0 <= value <= 1
    )


def _positive_reference(record):
    value = record.get("reference_rows")
    return type(value) is int and value > 0


def _row_counts(record):
    counts = [
        record.get(name)
        for name in ("result_rows", "reference_rows", "matched_rows")
    ]
    return (
        all(type(value) is int and value >= 0 for value in counts)
        and counts[2] <= min(counts[:2])
    )


def _average(values):
    return mean(values) if values else None


def summarize_native(report):
    """Return macro/micro F1 and coverage without mixing aggregate deviations.

    A query is in the common set only when every configuration has a scored
    non-aggregate result and its own reference contains at least one row.
    Macro averages retain the matcher's empty/empty convention. Micro F1 pools
    row counts; when all included relations are empty, it is likewise one.
    Missing or pending scores never become zero. Unknown reference row counts
    are excluded from the nonempty-reference scope.
    """
    records = report["records"]
    variants = report["variants"]
    row_queries = {r["query_id"] for r in records if not r.get("aggregate")}
    aggregate_queries = {r["query_id"] for r in records if r.get("aggregate")}
    by_variant = {
        variant["name"]: [r for r in records if r["variant"] == variant["name"]]
        for variant in variants
    }
    shared = set(row_queries) if by_variant else set()
    for rows in by_variant.values():
        shared &= {
            r["query_id"] for r in rows if _scored(r) and _positive_reference(r)
        }

    output = {}
    for variant in variants:
        rows = by_variant[variant["name"]]
        row_records = [r for r in rows if not r.get("aggregate")]
        scored = [r for r in row_records if _scored(r)]
        nonempty = [r for r in scored if _positive_reference(r)]
        common = [r["accuracy"] for r in scored if r["query_id"] in shared]
        micro = [r for r in scored if _row_counts(r)]
        denominator = sum(r["result_rows"] + r["reference_rows"] for r in micro)
        aggregate_scored = [
            r
            for r in rows
            if r.get("aggregate") and r.get("scoring_status") == "aggregate_scored"
        ]
        finite_deviations = [
            r["relative_deviation"]
            for r in aggregate_scored
            if _number(r.get("relative_deviation"))
            and r["relative_deviation"] >= 0
            and not r.get("deviation_infinite")
            and r.get("deviation_status") in {"numeric", "zero_reference"}
        ]
        nonzero_deviations = [
            r["relative_deviation"]
            for r in aggregate_scored
            if r.get("deviation_status") == "numeric"
            and _number(r.get("relative_deviation"))
            and r["relative_deviation"] >= 0
        ]
        output[variant["name"]] = {
            "system": variant["system"],
            "row_f1_mean": _average([r["accuracy"] for r in scored]),
            "row_f1_count": len(scored),
            "row_query_total": len(row_queries),
            "nonempty_row_f1_mean": _average([r["accuracy"] for r in nonempty]),
            "nonempty_row_f1_count": len(nonempty),
            "nonempty_row_query_total": sum(
                _positive_reference(r) for r in row_records
            ),
            "empty_reference_queries": sorted(
                r["query_id"]
                for r in row_records
                if type(r.get("reference_rows")) is int and r["reference_rows"] == 0
            ),
            "micro_row_f1": (
                2 * sum(r["matched_rows"] for r in micro) / denominator
                if denominator
                else 1.0
                if micro
                else None
            ),
            "micro_row_f1_count": len(micro),
            "common_row_f1_mean": _average(common),
            "common_row_f1_count": len(common),
            "aggregate_query_total": len(aggregate_queries),
            "aggregate_scored_count": len(aggregate_scored),
            "finite_aggregate_deviation_mean": _average(finite_deviations),
            "finite_aggregate_deviation_count": len(finite_deviations),
            "nonzero_aggregate_deviation_mean": _average(nonzero_deviations),
            "nonzero_aggregate_deviation_count": len(nonzero_deviations),
            "aggregate_both_null_count": sum(
                r.get("deviation_status") == "both_null" for r in aggregate_scored
            ),
            "aggregate_null_mismatch_count": sum(
                r.get("deviation_status") == "null_mismatch" for r in aggregate_scored
            ),
            "aggregate_zero_reference_count": sum(
                r.get("deviation_status") == "zero_reference"
                for r in aggregate_scored
            ),
            "aggregate_infinite_deviation_count": sum(
                bool(r.get("deviation_infinite")) for r in aggregate_scored
            ),
        }
    return {
        "common_nonempty_row_queries": sorted(shared),
        "variants": output,
    }
