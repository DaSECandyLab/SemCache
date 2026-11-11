"""Freeze original full tables once and bind output schemas without LLM calls."""

import hashlib
import json
import uuid
from pathlib import Path

import sqlglot
from sqlglot import exp

from experiments.core.io import digest as digest
from experiments.core.io import save as save
from experiments.core.paths import ROOT

TABLES = {
    "appstore": ("user_reviews", "playstore"),
    "movies": ("nirvana_movies",),
    "estate": ("nirvana_estate",),
    "movies_sema": (
        "movie",
        "genre",
        "movie_genres",
        "language",
        "movie_languages",
        "country",
        "production_country",
    ),
    "tpch": (
        "lineitem",
        "orders",
        "customer",
        "part",
        "supplier",
        "partsupp",
        "nation",
        "region",
    ),
}
CSV_NULL = "__SEMCACHE_PAPER_NULL_073d47c220744db7__"


def choose_csv_null(connection, table, schema):
    """Select a NULL marker absent from every textual value in a source table."""
    candidates = [CSV_NULL]
    candidates.extend(f"__SEMCACHE_PAPER_NULL_{uuid.uuid4().hex}__" for _ in range(4))
    text_columns = [
        name
        for name, kind in schema
        if any(token in kind.upper() for token in ("CHAR", "TEXT", "STRING"))
    ]
    for candidate in candidates:
        occupied = any(
            connection.execute(
                f'SELECT EXISTS (SELECT 1 FROM "{table}" WHERE CAST("{name}" AS VARCHAR) = ?)',
                [candidate],
            ).fetchone()[0]
            for name in text_columns
        )
        if not occupied:
            return candidate
    raise ValueError(f"Could not find collision-free CSV NULL marker for {table}")


def freeze_tables(database, output):
    import duckdb

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tables = {}
    with duckdb.connect(str(database), read_only=True) as source:
        for (raw,) in source.execute("SHOW TABLES").fetchall():
            if not raw.startswith("raw_"):
                continue
            table = raw.removeprefix("raw_")
            schema = [
                [name, kind]
                for name, kind, *_ in source.execute(f'DESCRIBE "{raw}"').fetchall()
            ]
            if any(
                name in {"__source_rowid", "__validation_rowid"} for name, _ in schema
            ):
                raise ValueError(
                    "Original source unexpectedly contains adapter row identity"
                )
            parquet, csv = output / (table + ".parquet"), output / (table + ".csv")
            csv_null = choose_csv_null(source, raw, schema)
            source.execute(
                f'COPY (SELECT *,rowid AS __source_rowid,rowid AS __validation_rowid FROM "{raw}" ORDER BY rowid) TO ? (FORMAT PARQUET)',
                [str(parquet)],
            )
            source.execute(
                f'COPY (SELECT *,rowid AS __source_rowid,rowid AS __validation_rowid FROM "{raw}" ORDER BY rowid) TO ? (FORMAT CSV, HEADER TRUE, NULL ?)',
                [str(csv), csv_null],
            )
            tables[table] = {
                "path": str(parquet),
                "csv": str(csv),
                "rows": source.execute(f'SELECT count(*) FROM "{raw}"').fetchone()[0],
                "columns": schema,
                "sha256": digest(parquet),
                "csv_sha256": digest(csv),
                "csv_null": csv_null,
            }
    frozen = output / "inputs.duckdb"
    with duckdb.connect(str(frozen)) as connection:
        for table, entry in tables.items():
            # Replace partially exported input tables when resuming preparation before manifest publication.
            connection.execute(
                f'CREATE OR REPLACE TABLE "{table}" AS SELECT * EXCLUDE(__source_rowid,__validation_rowid) FROM read_parquet(?) ORDER BY __validation_rowid',
                [entry["path"]],
            )
    return tables


def output_schema(connection, sql):
    tree = sqlglot.parse_one(sql, read="duckdb")

    def stub(node):
        if isinstance(node, exp.Anonymous):
            if node.name.lower() in {"semantic_filter", "semantic_join"}:
                return exp.Boolean(this=True)
            if node.name.lower() == "semantic_project":
                schema = node.expressions[-1].args["to"]
                # DuckDB folds NULL::STRUCT(...).field to untyped NULL. A real
                # struct with typed fields preserves the binder's output type.
                return exp.Anonymous(
                    this="STRUCT_PACK",
                    expressions=[
                        exp.PropertyEQ(
                            this=field.this.copy(),
                            expression=exp.Cast(
                                this=exp.Null(), to=field.args["kind"].copy()
                            ),
                        )
                        for field in schema.expressions
                    ],
                )
        return node

    ordinary = tree.transform(stub).sql(dialect="duckdb")
    return [
        [name, str(kind)]
        for name, kind, *_ in connection.execute(
            "SELECT * FROM (" + ordinary + ") AS __contract LIMIT 0"
        ).description
    ]


def prepare(output, *, root=ROOT):
    import duckdb

    output = Path(output).resolve()
    existing = output / "manifest.json"
    if existing.exists():
        manifest = json.loads(existing.read_text())
        if (
            digest(root / "benchmarks/queries/manifest.json")
            != manifest["query_manifest_sha256"]
        ):
            raise ValueError(
                "Query manifest changed after freeze; use a new experiment directory"
            )
        for entry in manifest["tables"].values():
            if (
                digest(entry["path"]) != entry["sha256"]
                or digest(entry["csv"]) != entry["csv_sha256"]
            ):
                raise ValueError("Frozen full input changed")
        for job in manifest["jobs"]:
            if hashlib.sha256(job["sql"].encode()).hexdigest() != job["sql_sha256"]:
                raise ValueError("Frozen query changed")
        return manifest
    source = root / "experiments/inputs/inputs.duckdb"
    tables = freeze_tables(source, output)
    jobs = []
    with duckdb.connect(str(output / "inputs.duckdb"), read_only=True) as connection:
        for query in json.loads(
            (root / "benchmarks/queries/manifest.json").read_text()
        )["queries"]:
            text = (root / "benchmarks/queries" / query["sql"]).read_text()
            actual_hash = hashlib.sha256(text.encode()).hexdigest()
            if actual_hash != query["sql_sha256"]:
                raise ValueError("Canonical query differs from selection manifest")
            jobs.append(
                {
                    **query,
                    "sql": text,
                    "sql_sha256": actual_hash,
                    "original": str(
                        (root / "benchmarks/queries" / query["original"]).resolve()
                    ),
                    "input_database": str(output / "inputs.duckdb"),
                    "inputs": {name: tables[name] for name in TABLES[query["dataset"]]},
                    "output_schema": output_schema(connection, text),
                    "input_mode": "full_original",
                    "input_manifest": str(existing),
                }
            )
    manifest = {
        "input_mode": "full_original",
        "source_database": str(source),
        "source_database_sha256": digest(source),
        "query_manifest_sha256": digest(root / "benchmarks/queries/manifest.json"),
        "tables": tables,
        "jobs": jobs,
    }
    save(existing, manifest)
    return manifest
