"""Freeze original training SQL without translating it to another system."""

import hashlib
import json
import re
from pathlib import Path


def load_training_sources(path, databases):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    directory = path.parent / "queries"
    frozen = directory.is_dir() or (path.parent / "protocol.json").is_file()
    workloads, seen = [], set()
    for original in manifest["workloads"]:
        identity, dataset = original["id"], original["dataset"]
        if (
            not re.fullmatch(r"[A-Za-z0-9._-]+", identity)
            or identity in seen
            or original.get("split") != "train"
            or dataset not in databases
            or not Path(databases[dataset]).is_file()
        ):
            raise ValueError(
                f"Invalid training query identity/dataset/split: {identity}"
            )
        seen.add(identity)
        source = (
            directory / f"{identity}.sql"
            if frozen
            else path.parent / dataset / f"{identity}.sql"
        )
        stored = source.read_text()
        sql = original["sql"]
        digest = hashlib.sha256(sql.encode()).hexdigest()
        # Accept the writer-added trailing newline while executing the manifest SQL text.
        if digest != original["sql_sha256"] or stored != sql and stored != sql + "\n":
            raise ValueError(f"Training SQL and frozen manifest differ: {source}")
        workloads.append(
            {
                **original,
                "source_sql": sql,
                "source_sql_sha256": digest,
                "source_path": str(source),
                "source_file_sha256": hashlib.sha256(stored.encode()).hexdigest(),
            }
        )
    if not workloads:
        raise ValueError("Training manifest has no queries")
    return workloads, {
        "source_manifest": str(path),
        "source_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "workload_version": manifest["version"],
    }
