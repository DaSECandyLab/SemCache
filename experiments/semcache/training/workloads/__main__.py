"""Check or rebuild the approved training catalog from its final SQL definitions."""

import argparse
from collections import Counter
from pathlib import Path

from planning.settings.configuration import (
    ROOT,
    activate_native,
    load_config,
    runtime_path,
)

from training.workloads.catalog import compile_catalog, publish_catalog


def bind_queries(payload, settings, *, probe_inputs=False):
    """Validate all SQL with the production binder, without model requests."""
    from planning.backends.duckdb import DuckDBConnection
    from planning.backends.execution import DuckDBEvaluator

    from training.collection.features import normalize_explain
    from training.query_validation.probes import input_coverage
    from training.workloads.checks import (
        validate_bound_output,
        validate_native_budget,
        workload_operator_limit,
    )

    activate_native(settings)
    build = runtime_path(settings, "build")
    databases = runtime_path(settings, "source_databases")
    models = {
        "offline-audit": {
            "base_url": "http://127.0.0.1:1",
            "model": "offline-audit",
            "max_tokens": 64,
        }
    }
    for query in payload["workloads"]:
        with DuckDBEvaluator(
            build, models, query["sql"], database=databases[query["dataset"]]
        ) as evaluator:
            bound = normalize_explain(evaluator.run(explain=True))
        query["operator_counts"] = validate_native_budget(
            bound, maximum=workload_operator_limit(query)
        )
        if query["operator_counts"]["semantic_nodes"] != query["semantic_nodes"]:
            raise ValueError(f"Native semantic operator count differs: {query['id']}")
        query["operator_type_counts"] = dict(
            sorted(
                Counter(
                    node["type"]
                    for node in bound["operators"]
                    if node["type"] != "EXPLAIN"
                ).items()
            )
        )
        query["result_schema"] = validate_bound_output(query, bound)
        if probe_inputs:
            connection = DuckDBConnection(build, databases[query["dataset"]])
            try:
                query["input_coverage"] = input_coverage(
                    connection,
                    query,
                    minimum_distinct_inputs=payload["requirements"][
                        "minimum_distinct_semantic_inputs"
                    ],
                    minimum_result_rows=payload["requirements"]["minimum_result_rows"],
                )
            finally:
                connection.close()
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Versioned manifest beside the dataset SQL directories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Publish the manifest and exact SQL files; omitted means read-only validation",
    )
    parser.add_argument(
        "--bind",
        action="store_true",
        help="Bind every SQL using DuckDB; no model calls",
    )
    parser.add_argument(
        "--probe-inputs",
        action="store_true",
        help="Also count relational opportunities without calling models",
    )
    args = parser.parse_args()
    settings = load_config()
    path = (args.manifest or ROOT / settings["training"]["workload_manifest"]).resolve()
    payload, audit = compile_catalog(path)
    if args.bind or args.probe_inputs:
        bind_queries(payload, settings, probe_inputs=args.probe_inputs)
        audit["native_binding"] = f"{len(payload['workloads'])} queries passed"
    if args.output is not None:
        publish_catalog(path, args.output, payload, audit)
    counts = Counter(query["dataset"] for query in payload["workloads"])
    print(
        f"Validated {len(payload['workloads'])} queries, version {payload['version']}: {dict(sorted(counts.items()))}"
    )
    if args.output is not None:
        print(f"Published exact SQL definitions and catalog to {args.output.resolve()}")


if __name__ == "__main__":
    main()
