"""Explicit offline table statistics; query-time access is strictly read-only."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from planning.settings.configuration import ROOT, load_config, runtime_path


def database_snapshot(database):
    """Bind artifacts to a local immutable database/WAL generation."""
    path = Path(database).resolve()

    def stamp(p):
        if not p.exists():
            return None
        s = p.stat()
        return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]

    return {
        "database": str(path),
        "file": stamp(path),
        "wal": stamp(Path(str(path) + ".wal")),
    }


def _table_key(schema, table):
    return hashlib.sha256(
        json.dumps([schema, table], ensure_ascii=False).encode()
    ).hexdigest()


class OfflineStatisticsStore:
    """Load each needed table once; never create, scan or sample in this class."""

    def __init__(self, directory, database):
        self.directory = (ROOT / directory).resolve()
        self.database = Path(database).resolve()
        self.cache = {}

    def path(self, schema, table):
        database = hashlib.sha256(str(self.database).encode()).hexdigest()[:20]
        return self.directory / database / (_table_key(schema, table) + ".json")

    def load(self, schema, table):
        path = self.path(schema, table)
        if not path.exists():
            return None
        key = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        if key not in self.cache:
            value = json.loads(path.read_text())
            if value.get("version") != 2:
                raise ValueError("Unsupported offline prefix-statistics version")
            self.cache = {k: v for k, v in self.cache.items() if k[0] != str(path)}
            self.cache[key] = value
        value = self.cache[key]
        return value if value["snapshot"] == database_snapshot(self.database) else None

    def for_query(self, query, *, required=False):
        result = {}
        for relation in query["relations"]:
            if not relation.get("table") or not relation.get("schema"):
                continue
            table = self.load(relation["schema"], relation["table"])
            if table is None:
                if required:
                    raise ValueError(
                        "Missing or stale offline prefix/cooccurrence statistics for "
                        f"{relation['schema']}.{relation['table']}; "
                        "run semcache_statistics_build before value-guided search"
                    )
                continue
            if relation.get("catalog") not in (None, table["catalog"]):
                if required:
                    raise ValueError("Offline prefix-statistics catalog mismatch")
                continue
            from .cache import require_hash_entries

            require_hash_entries(table["cooccurrence"])
            missing = [
                c["name"] for c in relation["columns"]
                if c["name"] not in table["summaries"]
            ]
            if required and missing:
                raise ValueError(
                    "Missing offline prefix-statistics columns: " + ", ".join(missing)
                )
            columns = {
                json.dumps(c["binding"], separators=(",", ":")): c["name"]
                for c in relation["columns"]
                if c["name"] in table["summaries"]
            }
            result[str(relation["node_id"])] = {
                "columns": columns,
                "summaries": {
                    key: table["summaries"][name] for key, name in columns.items()
                },
                "cooccurrence": table["cooccurrence"],
            }
        return result


def build_database(
    database, directory, build, *, tables=None, config=None, parallelism=None
):
    """Launch the native builder; Python neither scans nor constructs statistics."""
    if parallelism is None:
        parallelism = load_config()["statistics"]["build_parallelism"]
    command = [
        str(Path(build).resolve() / "tools/semcache_statistics_build"),
        "--database",
        str(Path(database).resolve()),
        "--output",
        str((ROOT / directory).resolve()),
        "--parallelism",
        str(parallelism),
    ]
    if config is not None:
        command += ["--adaptive-config", json.dumps(config, allow_nan=False)]
    if tables:
        command += ["--tables", *tables]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    print(result.stdout, end="", flush=True)
    return [json.loads(line) for line in result.stdout.splitlines() if line]


def prefix_common(bank, left, right, f, g):
    """Reference for the read-only native estimator; no input-table access."""
    from .cache import ENTRY_ENCODING, require_hash_entries

    require_hash_entries(bank)
    fields, common = bank["fields"], {"entry_encoding": ENTRY_ENCODING}
    fi, gi = fields.index(f), fields.index(g)
    for depth, sketch in left["sketches"].items():
        if depth not in right["sketches"]:
            continue
        require_hash_entries(sketch)
        require_hash_entries(right["sketches"][depth])
        other = right["sketches"][depth]["entries"]
        for p, a in sketch["entries"].items():
            if p not in other:
                continue
            b = other[p]
            source, target = (f, gi) if (a, f) < (b, g) else (g, fi)
            ids = bank["samples"].get(source, {}).get(depth, {}).get(p, [])
            source_sketch = sketch if source == f else right["sketches"][depth]
            text = source_sketch["prefixes"][p] if ids else None
            matches = sum(bank["rows"][i][target].startswith(text) for i in ids)
            common.setdefault(depth, {})[p] = (
                min(a, b) * matches / len(ids) if ids else 0.0
            )
    return common


def main():
    settings = load_config()
    executable = runtime_path(settings, "build") / "tools/semcache_statistics_build"
    os.execv(str(executable), [str(executable), *sys.argv[1:]])


if __name__ == "__main__":
    main()
