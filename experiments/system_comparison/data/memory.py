"""Preload ordinary input data only; never execute semantic operators here."""

from pathlib import Path

from experiments.core.paths import ROOT

DATABASE_NAMES = {"movies": "movies_nirvana"}


def database_path(system, dataset, root=ROOT):
    return (
        root
        / "benchmarks/databases"
        / system
        / (DATABASE_NAMES.get(dataset, dataset) + ".duckdb")
    )


def quote(value):
    return '"' + str(value).replace('"', '""') + '"'


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def source_schema(database):
    import duckdb

    with duckdb.connect(str(database), read_only=True) as con:
        objects = con.execute(
            "SELECT table_name,table_type FROM information_schema.tables WHERE table_schema='main' ORDER BY table_name"
        ).fetchall()
        return {
            name: {
                "type": kind,
                "columns": [
                    (r[0], r[1])
                    for r in con.execute("DESCRIBE " + quote(name)).fetchall()
                ],
            }
            for name, kind in objects
        }


def preload_sql(database, tables=None):
    """Detach disk input after copying full rows to the native memory catalog."""
    # A native artifact may use a different DuckDB vector size (PLOP uses 16).
    # Its database must be read by that binary, not the shared Python reader.
    if tables is None:
        schema = source_schema(database)
        names = list(schema)
    else:
        names = list(tables)
        schema = {
            n: {
                "type": "VIEW"
                if n.startswith("raw_") and n[4:] in names
                else "BASE TABLE"
            }
            for n in names
        }
    missing = set(names) - set(schema)
    if missing:
        raise ValueError("Missing input relations: " + str(sorted(missing)))
    # Raw aliases must refer to already materialized base tables, rather than
    # retaining a lazy dependency on the persistent source catalog.
    aliases = [
        n
        for n in names
        if n.startswith("raw_") and n[4:] in names and schema[n]["type"] == "VIEW"
    ]
    names = [n for n in names if n not in aliases]
    statements = [
        f"ATTACH {literal(Path(database).resolve())} AS __benchmark_input (READ_ONLY);"
    ]
    for name in names:
        order = " ORDER BY rowid" if schema[name]["type"] == "BASE TABLE" else ""
        statements.append(
            f"CREATE TEMP TABLE {quote(name)} AS SELECT * FROM __benchmark_input.main.{quote(name)}{order};"
        )
    for name in aliases:
        statements.append(
            f"CREATE TEMP VIEW {quote(name)} AS SELECT * FROM {quote(name[4:])};"
        )
    statements.append("DETACH __benchmark_input;")
    return "\n".join(statements)


def bind_database(job):
    """Select the system-specific source for new jobs; preserve frozen old jobs."""
    job = dict(job)
    if job.get("input_loading") == "database_memory":
        candidates = job.get("input_databases", {})
        source = candidates.get(job["system"])
        job["input_database"] = (
            source["path"]
            if source
            else str(database_path(job["system"], job["dataset"]))
        )
        if source:
            job["input_database_sha256"] = source["sha256"]
    return job


def prepare_campaign(run, config, manifest, query_files, root=ROOT):
    """Freeze source identities, not another CSV-to-memory conversion path."""
    import json

    import duckdb

    from experiments.core.io import digest, save
    from experiments.system_comparison.data.inputs import TABLES, output_schema

    catalog_path = root / "benchmarks/databases/manifest.json"
    catalog = json.loads(catalog_path.read_text())
    databases = {}
    tables = {}
    for system, datasets in catalog["systems"].items():
        databases[system] = {}
        for dataset, item in datasets.items():
            path = root / "benchmarks/databases" / item["database"]
            if digest(path) != item["sha256"]:
                raise ValueError("Prepared database changed: " + str(path))
            databases[system][dataset] = {"path": str(path), "sha256": item["sha256"]}
            if system == "semcache":
                schema = source_schema(path)
                for name, info in item["tables"].items():
                    tables[name] = {
                        "columns": schema[name]["columns"],
                        "rows": info["rows"],
                    }
    jobs = {}
    for query_id in config["query_ids"]:
        dataset, query = query_id.split("/")
        database = DATABASE_NAMES.get(dataset, dataset)
        generic = next(
            r
            for r in manifest["records"]
            if r["system"] == "semcache" and r["id"] == query_id
        )
        sql = (root / "benchmarks/queries" / generic["query_file"]).read_text()
        candidates = {s: paths[database] for s, paths in databases.items()}
        with duckdb.connect(candidates["semcache"]["path"], read_only=True) as con:
            schema = output_schema(con, sql)
        jobs[query_id] = {
            "id": query_id,
            "dataset": dataset,
            "query": query,
            "input_loading": "database_memory",
            "input_databases": candidates,
            "inputs": {n: tables[n] for n in TABLES[dataset]},
            "output_schema": schema,
            "workload_contract": generic.get("contract", config["contract"]),
        }
    frozen = {
        "input_loading": "database_memory",
        "databases": databases,
        "tables": tables,
        "jobs": jobs,
        "query_files": query_files,
        "database_manifest_sha256": digest(catalog_path),
    }
    save(run / "inputs/manifest.json", frozen)
    return frozen
