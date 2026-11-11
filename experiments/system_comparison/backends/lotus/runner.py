"""Run the source-faithful LOTUS queries on shared frozen inputs.

LOTUS is isolated from the historical campaign. All 28 selected queries
use native LOTUS semantic operators; ordinary SQL/aggregation uses the disclosed
DuckDB host. This runner does not enable whole-query LOTUS optimization.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

from experiments.core.paths import ROOT

QUERY_ROOT = ROOT / "benchmarks/queries/translated/lotus"
TABLES = {
    "estate": "raw_nirvana_estate",
    "movies": "raw_nirvana_movies",
    "appstore": "raw_user_reviews",
    "tpch": "raw_lineitem",
    "movies_sema": "movie",
}


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("lotus_generated_query", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def json_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        return json_value(value.item())
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    return str(value)


def run_one(path: Path, database: Path, model):
    dataset = path.parent.name
    query = path.stem
    started = time.monotonic()
    if dataset not in TABLES:
        return {
            "system": "lotus",
            "query": f"{dataset}/{query}",
            "status": "unsupported",
            "reason": "LOTUS adapter has no SQL/CTE executor for SQL-origin workload",
            "seconds": round(time.monotonic() - started, 3),
        }
    sys.path.insert(0, str(ROOT / "systems/lotus/lotus-upstream"))
    module = load_module(path)
    import lotus

    lotus.settings.configure(lm=model)
    runtime_setup_seconds = time.monotonic() - started
    from experiments.validation.faithful_runtime import load_common_tables

    if database is None:
        from experiments.system_comparison.data.memory import database_path

        database = database_path("lotus", dataset)
    load_started = time.monotonic()
    tables = load_common_tables(database, dataset)
    load_seconds = time.monotonic() - load_started
    query_started = time.monotonic()
    result = module.run(tables, {"id": f"{dataset}/{query}"}, model=model)
    query_seconds = time.monotonic() - query_started
    schema = result.attrs.get("duckdb_schema")
    columns = (
        [[str(c), str(t)] for c, t in schema]
        if schema
        else [
            [
                str(c),
                "BIGINT"
                if getattr(module, "AGGREGATE", {}).get("function") == "COUNT"
                else "DOUBLE"
                if c == "result"
                else "VARCHAR",
            ]
            for c in result.columns
        ]
    )
    rows = [
        [json_value(value) for value in row]
        for row in result.itertuples(index=False, name=None)
    ]
    return {
        "system": "lotus",
        "query": f"{dataset}/{query}",
        "status": "completed",
        "result": {"columns": columns, "rows": rows},
        "seconds": query_seconds,
        "load_seconds": load_seconds,
        "worker_wall_seconds": time.monotonic() - started,
        "timing": {
            "query_seconds": query_seconds,
            "load_seconds": load_seconds,
            "details": {
                "runtime_setup_seconds": runtime_setup_seconds,
                "input_policy": "preload_before_query",
            },
        },
        "model_id": getattr(model, "model", None),
        "execution_mode": "lotus_native_semantics_with_disclosed_relational_host",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database", type=Path, help="Override the per-dataset LOTUS database"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True, help="LOTUS/LiteLLM model ID")
    parser.add_argument("--only", help="dataset/Q prefix")
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT / "systems/lotus/lotus-upstream"))
    from lotus.models import LM

    model = LM(model=args.model)
    records = []
    for path in sorted(QUERY_ROOT.glob("*/*.py")):
        if args.only and args.only not in f"{path.parent.name}/{path.stem}":
            continue
        try:
            records.append(run_one(path, args.database, model))
        except (
            Exception
        ) as exc:   
            records.append(
                {
                    "system": "lotus",
                    "query": f"{path.parent.name}/{path.stem}",
                    "status": "failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    payload = {
        "system": "lotus",
        "model_id": args.model,
        "query_manifest": str(ROOT / "benchmarks/queries/translated/manifest.json"),
        "records": records,
        "native_support": "28 queries: native semantic APIs plus disclosed ordinary DuckDB host; no whole-query optimization",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "completed": sum(r["status"] == "completed" for r in records),
                "unsupported": sum(r["status"] == "unsupported" for r in records),
                "failed": sum(r["status"] == "failed" for r in records),
                "total": len(records),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
