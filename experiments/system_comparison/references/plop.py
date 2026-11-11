"""Losslessly migrate PLOP's vector16 input files to the vector128 runtime.

Only the matching native executable opens either database format.  Parquet is
an interchange format, and PyArrow compares typed rows in physical rowid order.
No model client or reference-answer cache is involved.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
SOURCE_BINARY = ROOT / "systems/plop/duckdb-local-8902"
SOURCE_ROOT = ROOT / "benchmarks/databases/plop"
DATASETS = ("estate", "movies_nirvana", "movies_sema", "appstore", "tpch")
FORMAT = 1
ROWID = "__plop_migration_physical_rowid__"


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _literal(value) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql(binary: Path, database: Path, sql: str, *, readonly=True) -> str:
    command = [str(binary), "-init", "/dev/null", "-batch", "-bail", "-json"]
    if readonly:
        command.append("-readonly")
    result = subprocess.run(
        command + [str(database)],
        input=sql,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"PLOP migration {database.name}: exit {result.returncode}: "
            + result.stderr[-2000:]
        )
    return result.stdout


def _query(binary: Path, database: Path, sql: str) -> list:
    output = _sql(binary, database, sql)
    return json.loads(output) if output.strip() else []


def _catalog(binary: Path, database: Path) -> dict:
    objects = _query(
        binary,
        database,
        """
        SELECT 'table' AS kind, table_name AS name, sql FROM duckdb_tables()
        WHERE schema_name='main' AND NOT internal
        UNION ALL
        SELECT 'view' AS kind, view_name AS name, sql FROM duckdb_views()
        WHERE schema_name='main' AND NOT internal
        ORDER BY kind,name;
    """,
    )
    columns = _query(
        binary,
        database,
        """
        SELECT table_name,column_name,data_type,is_nullable,column_default,
               ordinal_position
        FROM information_schema.columns WHERE table_schema='main'
        ORDER BY table_name,ordinal_position;
    """,
    )
    schemas = _query(
        binary,
        database,
        """
        SELECT schema_name FROM information_schema.schemata
        WHERE schema_name NOT IN ('main','information_schema','pg_catalog');
    """,
    )
    if schemas:
        raise ValueError(
            "PLOP migration requires explicit support for non-main schemas"
        )
    for column in columns:
        if column["column_name"].lower() in {"rowid", ROWID.lower()}:
            raise ValueError("Source column shadows the migration's physical rowid")
    return {"objects": objects, "columns": columns}


def _rowids(binary: Path, database: Path, table: str) -> dict:
    stats = _query(
        binary,
        database,
        f"""
        SELECT count(*) AS rows,min(rowid) AS first,max(rowid) AS last,
               count(DISTINCT rowid) AS distinct_rowids FROM {_quote(table)};
    """,
    )[0]
    count = stats["rows"]
    if stats["distinct_rowids"] != count or (
        count and (stats["first"] != 0 or stats["last"] != count - 1)
    ):
        raise ValueError(
            f"Non-contiguous physical rowids: {database.name}/{table}: {stats}"
        )
    return stats


def _export(binary: Path, database: Path, table: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".building.parquet")
    if temporary.exists():
        temporary.unlink()
    _sql(
        binary,
        database,
        f"""
        SET threads=1;
        SET preserve_insertion_order=true;
        COPY (SELECT rowid AS {_quote(ROWID)},* FROM {_quote(table)} ORDER BY rowid)
        TO {_literal(temporary)} (FORMAT PARQUET, COMPRESSION ZSTD);
    """,
    )
    os.replace(temporary, path)


def _prepare_source(dataset: str, folder: Path, expected_hash: str) -> dict:
    source = SOURCE_ROOT / (dataset + ".duckdb")
    source_hash = _digest(source)
    if source_hash != expected_hash:
        raise ValueError(f"Frozen PLOP source changed: {dataset}")
    binary_hash = _digest(SOURCE_BINARY)
    receipt = folder / "source.json"
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if (
            record["source_sha256"] != source_hash
            or record["source_binary_sha256"] != binary_hash
        ):
            raise ValueError(f"PLOP source evidence changed: {dataset}")
        for table in record["tables"].values():
            if _digest(Path(table["parquet"])) != table["parquet_sha256"]:
                raise ValueError(f"PLOP exported source changed: {dataset}")
        return record
    catalog = _catalog(SOURCE_BINARY, source)
    tables = {}
    for obj in catalog["objects"]:
        if obj["kind"] != "table":
            continue
        name = obj["name"]
        stats = _rowids(SOURCE_BINARY, source, name)
        path = folder / "source" / (name + ".parquet")
        _export(SOURCE_BINARY, source, name, path)
        tables[name] = {**stats, "parquet": str(path), "parquet_sha256": _digest(path)}
    if _digest(source) != source_hash:
        raise ValueError(f"PLOP source changed during export: {dataset}")
    record = {
        "source_path": str(source),
        "source_sha256": source_hash,
        "source_binary_sha256": binary_hash,
        "catalog": catalog,
        "tables": tables,
    }
    _save(receipt, record)
    return record


def _build(binary: Path, target: Path, source: dict) -> None:
    staging = target.with_suffix(".building.duckdb")
    # Only an incomplete, privately owned staging file is disposable.
    for path in (staging, Path(str(staging) + ".wal")):
        if path.exists():
            path.unlink()
    statements = [
        "SET threads=1;",
        "SET force_compression='uncompressed';",
        "SET preserve_insertion_order=true;",
        "BEGIN TRANSACTION;",
    ]
    catalog = source["catalog"]
    for obj in catalog["objects"]:
        if obj["kind"] != "table":
            continue
        name = obj["name"]
        names = [
            _quote(c["column_name"])
            for c in catalog["columns"]
            if c["table_name"] == name
        ]
        columns = ",".join(names)
        statements.extend(
            [
                obj["sql"],
                (
                    f"INSERT INTO {_quote(name)} ({columns}) SELECT {columns} FROM "
                    f"read_parquet({_literal(source['tables'][name]['parquet'])}) "
                    f"ORDER BY {_quote(ROWID)};"
                ),
            ]
        )
    statements.extend(obj["sql"] for obj in catalog["objects"] if obj["kind"] == "view")
    statements.extend(["COMMIT;", "CHECKPOINT;"])
    _sql(binary, staging, "\n".join(statements), readonly=False)
    os.replace(staging, target)


def _verify(binary: Path, target: Path, source: dict, folder: Path) -> dict:
    if _catalog(binary, target) != source["catalog"]:
        raise ValueError(f"PLOP DDL or schema changed: {target.name}")
    tables = {}
    for name, original in source["tables"].items():
        stats = _rowids(binary, target, name)
        if any(stats[key] != original[key] for key in stats):
            raise ValueError(f"PLOP physical rowids changed: {target.name}/{name}")
        path = folder / "target" / (name + ".parquet")
        _export(binary, target, name, path)
        old = pq.read_table(original["parquet"])
        new = pq.read_table(path)
        # The synthetic physical rowid is included in the typed comparison.
        # Equality in this order also proves bag equality, preserving duplicates.
        if not old.schema.equals(new.schema, check_metadata=False) or not old.equals(
            new
        ):
            raise ValueError(f"PLOP typed rows/order changed: {target.name}/{name}")
        rowids = old[ROWID].combine_chunks()
        if not rowids.equals(pa.array(range(old.num_rows), type=pa.int64())):
            raise ValueError(
                f"PLOP exported rowid ordering changed: {target.name}/{name}"
            )
        nulls = {
            column: old[column].null_count
            for column in old.column_names
            if column != ROWID
        }
        if nulls != {
            column: new[column].null_count
            for column in new.column_names
            if column != ROWID
        }:
            raise ValueError(f"PLOP NULL counts changed: {target.name}/{name}")
        tables[name] = {
            **stats,
            "schema_equal": True,
            "typed_rows_equal": True,
            "row_order_equal": True,
            "physical_rowids_equal": True,
            "bag_equal": True,
            "null_counts": nulls,
            "arrow_schema": [
                {
                    "name": field.name,
                    "type": str(field.type),
                    "nullable": field.nullable,
                }
                for field in old.schema
            ],
            "source_parquet_sha256": original["parquet_sha256"],
            "target_parquet_sha256": _digest(path),
        }
    return {
        "catalog_equal": True,
        "views_equal": True,
        "source_catalog": source["catalog"],
        "target_catalog": source["catalog"],
        "view_names": [
            o["name"] for o in source["catalog"]["objects"] if o["kind"] == "view"
        ],
        "verification_method": "PyArrow typed equality of all rows in physical rowid order",
        "tables": tables,
    }


def _reuse_retired(run: Path, binary_hash: str, records: dict, frozen: dict) -> dict:
    """Validate promoted files without opening them with the retired reader."""
    base = run / "runtime/plop-vector128"
    catalog = json.loads((ROOT / "benchmarks/databases/manifest.json").read_text())
    plop_catalog = json.loads((SOURCE_ROOT / "manifest.json").read_text())
    for dataset in DATASETS:
        record = records[dataset]
        folder = base / "verification" / dataset
        source = json.loads((folder / "source.json").read_text())
        receipt = json.loads((folder / "verified.json").read_text())
        target = SOURCE_ROOT / (dataset + ".duckdb")
        if (
            record != receipt
            or record["format"] != FORMAT
            or record["binary_sha256"] != binary_hash
            or record["path"] != str(target)
            or record["sha256"] != _digest(target)
            or record["source_sha256"] != source["source_sha256"]
            or record["source_path"] != source["source_path"]
            or record["source_catalog"] != source["catalog"]
            or record["target_catalog"] != source["catalog"]
            or frozen[dataset]["sha256"]
            not in {record["sha256"], record["source_sha256"]}
        ):
            raise ValueError(f"Retired PLOP database or receipt changed: {dataset}")
        local = json.loads((SOURCE_ROOT / dataset / "manifest.json").read_text())
        for item in (catalog["systems"]["plop"][dataset], plop_catalog[dataset], local):
            if (
                item["sha256"] != record["sha256"]
                or item["bytes"] != target.stat().st_size
                or item["binary_sha256"] != binary_hash
                or item["standard_vector_size"] != 128
            ):
                raise ValueError(f"Promoted PLOP database catalog changed: {dataset}")
        if set(source["tables"]) != set(record["tables"]):
            raise ValueError(f"Retired PLOP table evidence changed: {dataset}")
        for name, evidence in record["tables"].items():
            original = source["tables"][name]
            old_path = Path(original["parquet"])
            new_path = folder / "target" / (name + ".parquet")
            if (
                _digest(old_path) != original["parquet_sha256"]
                or original["parquet_sha256"] != evidence["source_parquet_sha256"]
                or _digest(new_path) != evidence["target_parquet_sha256"]
            ):
                raise ValueError(
                    f"Retired PLOP parquet evidence changed: {dataset}/{name}"
                )
            old, new = pq.read_table(old_path), pq.read_table(new_path)
            if (
                not old.schema.equals(new.schema, check_metadata=False)
                or not old.equals(new)
                or old.num_rows != evidence["rows"]
                or not old[ROWID]
                .combine_chunks()
                .equals(pa.array(range(old.num_rows), type=pa.int64()))
            ):
                raise ValueError(
                    f"Retired PLOP typed evidence changed: {dataset}/{name}"
                )
        print(
            f"PLOP vector128 {dataset}: retired source; verified retained evidence",
            flush=True,
        )
    return records


def prepare_databases(run: Path, binary: Path) -> dict[str, dict]:
    """Return validated vector128 database paths and evidence.

    Existing completed outputs are reused only after source, binary, database,
    and retained verification-file hashes have been checked. A per-dataset
    receipt makes interrupted preparation resumable without replacing a source.
    After an explicitly authorized retirement, retained typed exports and
    receipts verify the old data without opening it with the retired reader.
    """
    run, binary = Path(run).resolve(), Path(binary).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise FileNotFoundError(f"Vector128 executable is not ready: {binary}")
    binary_hash = _digest(binary)
    frozen = json.loads((run / "inputs/manifest.json").read_text())["databases"]["plop"]
    base = run / "runtime/plop-vector128"
    manifest = base / "manifest.json"
    if manifest.exists():
        prior = json.loads(manifest.read_text())["databases"]
        retired = [record.get("source_retired", False) for record in prior.values()]
        if any(retired):
            if set(prior) != set(DATASETS) or not all(retired):
                raise ValueError(
                    "PLOP source retirement is incomplete; finish promotion first"
                )
            return _reuse_retired(run, binary_hash, prior, frozen)
    if binary == SOURCE_BINARY.resolve():
        raise ValueError(
            "The vector128 executable must differ from the vector16 reader"
        )
    database_folder = base / "databases"
    database_folder.mkdir(parents=True, exist_ok=True)
    sources = {
        dataset: _prepare_source(
            dataset, base / "verification" / dataset, frozen[dataset]["sha256"]
        )
        for dataset in DATASETS
    }
    records = {}
    for dataset in DATASETS:
        folder = base / "verification" / dataset
        source = sources[dataset]
        target = database_folder / (dataset + ".duckdb")
        receipt = folder / "verified.json"
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if (
                record["format"] != FORMAT
                or record["binary_sha256"] != binary_hash
                or record["source_sha256"] != source["source_sha256"]
                or record["sha256"] != _digest(target)
            ):
                raise ValueError(f"Previously verified PLOP runtime changed: {dataset}")
            for name, evidence in record["tables"].items():
                if (
                    _digest(folder / "target" / (name + ".parquet"))
                    != evidence["target_parquet_sha256"]
                ):
                    raise ValueError(f"PLOP target evidence changed: {dataset}/{name}")
            print(f"PLOP vector128 {dataset}: verified hashes; reusing", flush=True)
        else:
            if not target.exists():
                _build(binary, target, source)
            verified = _verify(binary, target, source, folder)
            record = {
                "format": FORMAT,
                "path": str(target),
                "sha256": _digest(target),
                "source_path": source["source_path"],
                "source_sha256": source["source_sha256"],
                "binary_sha256": binary_hash,
                "standard_vector_size": 128,
                "storage_compression": "uncompressed",
                **verified,
            }
            _save(receipt, record)
            print(
                f"PLOP vector128 {dataset}: all typed rows, rowids, NULLs and views verified",
                flush=True,
            )
        records[dataset] = record
        _save(base / "manifest.json", {"format": FORMAT, "databases": records})
    if _digest(binary) != binary_hash:
        raise ValueError("PLOP vector128 binary changed during preparation")
    return records
