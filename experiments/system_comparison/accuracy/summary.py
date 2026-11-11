"""Explicit averaging scopes and coverage for the two separate quality metrics."""

from statistics import mean


def summarize(report):
    rows = report["records"]
    empty = sorted(
        q
        for q, info in report["reference_info"].items()
        if not info.get("aggregate") and info["rows"] == 0
    )
    row_queries = {
        q for q, info in report["reference_info"].items() if not info.get("aggregate")
    }
    aggregate_queries = set(report["query_ids"]) - row_queries
    nonzero_aggregates = {
        q
        for q, info in report["reference_info"].items()
        if info.get("aggregate")
        and info.get("aggregate_reference_value") not in (None, 0, "0")
    }
    by_variant = {v: [r for r in rows if r["variant"] == v] for v in report["variants"]}
    shared = row_queries - set(empty)
    for records in by_variant.values():
        shared &= {
            r["query_id"]
            for r in records
            if r.get("scoring_status") == "scored" and r.get("accuracy") is not None
        }
    output = {}
    for variant, records in by_variant.items():
        scored = [
            r
            for r in records
            if not r.get("aggregate") and r.get("accuracy") is not None
        ]
        nonempty = [r for r in scored if r["query_id"] not in empty]
        common = [r["accuracy"] for r in scored if r["query_id"] in shared]
        numeric = [
            r["relative_deviation"]
            for r in records
            if r.get("deviation_status") == "numeric"
            and r.get("relative_deviation") is not None
        ]
        zeros = {
            r["query_id"]: {
                k: r.get(k)
                for k in (
                    "aggregate_value",
                    "relative_deviation",
                    "deviation_infinite",
                    "deviation_status",
                )
            }
            for r in records
            if r.get("deviation_status") == "zero_reference"
        }
        output[variant] = {
            "row_f1_mean": mean(r["accuracy"] for r in scored) if scored else None,
            "row_f1_count": len(scored),
            "row_query_total": len(row_queries),
            "nonempty_row_f1_mean": mean(r["accuracy"] for r in nonempty)
            if nonempty
            else None,
            "nonempty_row_f1_count": len(nonempty),
            "nonempty_row_query_total": len(row_queries) - len(empty),
            "common_row_f1_mean": mean(common) if common else None,
            "nonzero_aggregate_deviation_mean": mean(numeric) if numeric else None,
            "nonzero_aggregate_deviation_count": len(numeric),
            "nonzero_aggregate_query_total": len(nonzero_aggregates),
            "aggregate_query_total": len(aggregate_queries),
            "zero_reference_results": zeros,
            "aggregate_null_mismatches": sum(
                r.get("deviation_status") == "null_mismatch" for r in records
            ),
        }
    return {
        "empty_reference_queries": empty,
        "common_nonempty_row_queries": sorted(shared),
        "variants": output,
    }
