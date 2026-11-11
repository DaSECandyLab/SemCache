"""Build dataset sources and pinned SF=0.005 TPC-H training tables."""

import json
import tempfile
from pathlib import Path

from planning.backends.duckdb import DuckDBConnection
from planning.backends.execution import identifier, quote
from planning.settings.configuration import ROOT
from reference_answers.storage.files import file_digest

CORPORA = {
    "movies_nirvana": ("raw_nirvana_movies", '"Plot"', '"Title"', '"IMDB_rating"'),
    "movies_sema": ("raw_movie", '"overview"', '"title"', '"vote_average"'),
    "estate": ("raw_nirvana_estate", '"Details"', '"Title"', 'length("Details")'),
    "appstore": (
        "raw_user_reviews",
        '"Translated_Review"',
        '"App"',
        'length("Translated_Review")',
    ),
    "tpch": ("raw_part", "p_type", "p_container", "p_retailprice"),
}

TPCH_GENERATION = {"method": "duckdb_tpch_dbgen", "scale_factor": 0.005}

DATASET_TABLES = {
    "movies_nirvana": {"raw_nirvana_movies": "movies/data/nirvana/movie_data.csv"},
    "movies_sema": {
        f"raw_{name}": f"movies/data/sema/csv/{name}.csv"
        for name in (
            "movie",
            "movie_genres",
            "genre",
            "movie_languages",
            "language",
            "production_country",
            "country",
        )
    },
    "estate": {
        "raw_nirvana_estate": "estate/data/nirvana/multimodal_real_estate.parquet"
    },
    "appstore": {
        f"raw_{name}": f"appstore/data/sema/csv/{name}.csv"
        for name in ("user_reviews", "playstore")
    },
    "tpch": {
        f"raw_{name}": f"tpch/data/plop/{name}.parquet"
        for name in (
            "customer",
            "lineitem",
            "nation",
            "orders",
            "part",
            "partsupp",
            "region",
            "supplier",
        )
    },
}


def _source_files(datasets):
    manifest = json.loads((datasets / "manifest.json").read_text())
    expected = {
        "estate/nirvana",
        "movies/nirvana",
        "movies/sema",
        "appstore/sema",
        "tpch/plop",
    }
    if {entry["id"] for entry in manifest["datasets"]} != expected:
        raise ValueError("Benchmark manifest must describe the five supported datasets")
    files = []
    for entry in manifest["files"]:
        path = datasets / entry["path"]
        digest = file_digest(path)
        if digest != entry["sha256"]:
            raise ValueError(f"Dataset checksum differs: {path}")
        files.append(
            {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
        )
    return manifest, files


def verify_sources(databases, build, datasets=None):
    """Validate existing source provenance, tables and counts without rebuilding."""
    datasets = Path(datasets or ROOT.parent / "benchmarks/datasets").resolve()
    manifest, files = _source_files(datasets)
    report_path = Path(next(iter(databases.values()))).parent / "sources.json"
    report = json.loads(report_path.read_text())
    if report["datasets"] != manifest["datasets"] or report["files"] != files:
        raise ValueError("Source provenance changed; regenerate with --rebuild-sources")
    if set(databases) != set(DATASET_TABLES) or set(report["databases"]) != set(
        databases
    ):
        raise ValueError("Existing source manifest must describe all five datasets")
    for dataset, database in databases.items():
        recorded = report["databases"][dataset]
        if dataset == "tpch" and recorded.get("generation", {}) != TPCH_GENERATION:
            raise ValueError("TPC-H source must be generated with dbgen(sf=0.005)")
        if Path(recorded["path"]).resolve() != Path(database).resolve():
            raise ValueError(f"Source path differs for {dataset}")
        if set(recorded["tables"]) != set(DATASET_TABLES[dataset]):
            raise ValueError(f"Source tables differ for {dataset}")
        connection = DuckDBConnection(build, database)
        try:
            for table, expected in recorded["tables"].items():
                actual = connection.execute(f"DESCRIBE {identifier(table)}", named=True)
                count = int(
                    connection.execute(f"SELECT count(*) FROM {identifier(table)}")[0][
                        0
                    ]
                )
                if actual != expected["columns"] or count != expected["rows"]:
                    raise ValueError(
                        f"Source schema or row count changed: {dataset}/{table}"
                    )
        finally:
            connection.close()
    return report


def build_tpch_source(database, build):
    """Replace only TPC-H with complete dbgen SF=0.005 tables, never a row slice."""
    database, build = Path(database).resolve(), Path(build).resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    extension = build / "extension/tpch/tpch.duckdb_extension"
    if not extension.is_file():
        raise FileNotFoundError(f"Build the pinned TPC-H extension first: {extension}")
    if Path(str(database) + ".wal").exists():
        raise ValueError(
            "Checkpoint and close the TPC-H writer before replacing its database"
        )
    tables = {}
    with tempfile.TemporaryDirectory(prefix=".tpch-", dir=database.parent) as folder:
        staged = Path(folder) / "tpch.duckdb"
        connection = DuckDBConnection(build, staged, readonly=False)
        try:
            connection.execute(f"LOAD {quote(extension)}")
            connection.execute(f"CALL dbgen(sf={TPCH_GENERATION['scale_factor']})")
            connection.execute("BEGIN TRANSACTION")
            for table in DATASET_TABLES["tpch"]:
                original = table.removeprefix("raw_")
                connection.execute(
                    f"CREATE TABLE {identifier(table)} AS SELECT * FROM {identifier(original)}"
                )
                tables[table] = {
                    "source": "CALL dbgen(sf=0.005)",
                    "rows": int(
                        connection.execute(f"SELECT count(*) FROM {identifier(table)}")[
                            0
                        ][0]
                    ),
                    "columns": connection.execute(
                        f"DESCRIBE {identifier(table)}", named=True
                    ),
                }
            for table in DATASET_TABLES["tpch"]:
                connection.execute(
                    f"DROP TABLE {identifier(table.removeprefix('raw_'))}"
                )
            connection.execute("COMMIT")
            connection.execute("CHECKPOINT")
        finally:
            connection.close()
        staged.replace(database)
    return {
        "path": str(database),
        "generation": dict(TPCH_GENERATION),
        "tables": tables,
    }


def build_file_source(dataset, database, build, datasets=None):
    """Import one complete file-backed dataset, preserving original row order."""
    if dataset not in DATASET_TABLES or dataset == "tpch":
        raise ValueError(f"Not a file-backed training dataset: {dataset}")
    datasets = Path(datasets or ROOT.parent / "benchmarks/datasets").resolve()
    database = Path(database).resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = DuckDBConnection(build, database, readonly=False)
    table_report = {}
    try:
        connection.execute("BEGIN TRANSACTION")
        for table, relative in DATASET_TABLES[dataset].items():
            path = datasets / relative
            scan = (
                f"read_parquet({quote(path)})"
                if path.suffix == ".parquet"
                else f"read_csv({quote(path)}, header=true, auto_detect=true, sample_size=-1)"
            )
            columns = "* EXCLUDE (image)" if table == "raw_nirvana_estate" else "*"
            connection.execute(
                f"CREATE OR REPLACE TABLE {identifier(table)} AS SELECT {columns} FROM {scan}"
            )
            schema = connection.execute(f"DESCRIBE {identifier(table)}", named=True)
            for column in schema:
                name, kind = column["column_name"], column["column_type"].upper()
                if any(
                    token in kind for token in ("DOUBLE", "FLOAT", "REAL", "DECIMAL")
                ):
                    quoted = identifier(name)
                    connection.execute(
                        f"UPDATE {identifier(table)} SET {quoted}=NULL WHERE isnan({quoted}) OR isinf({quoted})"
                    )
            table_report[table] = {
                "source": relative,
                "rows": int(
                    connection.execute(f"SELECT count(*) FROM {identifier(table)}")[0][
                        0
                    ]
                ),
                "columns": schema,
            }
        connection.execute("COMMIT")
        connection.execute("CHECKPOINT")
    finally:
        connection.close()
    return {"path": str(database), "tables": table_report}


def build_sources(databases, build, datasets=None):
    """Import complete original corpora and generate the pinned TPC-H source."""
    datasets = Path(datasets or ROOT.parent / "benchmarks/datasets").resolve()
    manifest, files = _source_files(datasets)
    resolved = {name: Path(path).resolve() for name, path in databases.items()}
    if set(resolved) != set(DATASET_TABLES):
        raise ValueError("source_databases must contain exactly the five dataset names")
    report = {"datasets": manifest["datasets"], "files": files, "databases": {}}
    for dataset, database in resolved.items():
        report["databases"][dataset] = (
            build_tpch_source(database, build)
            if dataset == "tpch"
            else build_file_source(dataset, database, build, datasets)
        )
    (Path(next(iter(resolved.values()))).parent / "sources.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    return report


def inspect_corpora(databases, build):
    """Describe complete source tables without generating template parameter domains."""
    report = {}
    for dataset, database in databases.items():
        table, text, auxiliary, metric = CORPORA[dataset]
        connection = DuckDBConnection(build, database)
        try:
            report[dataset] = {
                "database": str(database),
                "table": table,
                "columns": [text, auxiliary, metric],
                "row_count": int(
                    connection.execute(f"SELECT count(*) FROM {identifier(table)}")[0][
                        0
                    ]
                ),
                "source_row_counts": {
                    name: int(
                        connection.execute(f"SELECT count(*) FROM {identifier(name)}")[
                            0
                        ][0]
                    )
                    for name in DATASET_TABLES[dataset]
                },
                "text_policy": "unmodified original text and auxiliary fields",
                "input_scope": "complete source tables",
            }
        finally:
            connection.close()
    return report


def main():
    import argparse

    from planning.settings.configuration import (
        activate_native,
        load_config,
        runtime_path,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-import complete sources instead of verifying existing databases",
    )
    args = parser.parse_args()
    settings = load_config()
    activate_native(settings)
    databases, build = (
        runtime_path(settings, "source_databases"),
        runtime_path(settings, "build"),
    )
    report = (
        build_sources(databases, build)
        if args.rebuild
        else verify_sources(databases, build)
    )
    print(f"Verified complete sources for {len(report['databases'])} datasets")


if __name__ == "__main__":
    main()
