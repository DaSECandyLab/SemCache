"""Explicit offline table statistics; query-time access is strictly read-only."""

import hashlib
import json
import os
import subprocess
import sys
import threading
from collections import OrderedDict
from pathlib import Path

from planning.settings.configuration import ROOT, load_config, runtime_path

_PROCESS_OFFLINE_LOCK = threading.Lock()
_PROCESS_OFFLINE_CACHE: OrderedDict[
    tuple[str, int, int], tuple[dict, bool]
] = OrderedDict()


def select_joint_orders(order_counts, budget=8):
    """Select up to B_joint multi-column orders via length-stratified round robin."""
    if type(budget) is not int or budget < 0:
        raise ValueError("Joint sketch budget must be a nonnegative integer")
    if budget == 0:
        return []
    buckets = {}
    items = (
        order_counts.items()
        if isinstance(order_counts, dict)
        else order_counts
    )
    for raw_order, count in items:
        order = (
            tuple(raw_order)
            if isinstance(raw_order, (list, tuple))
            else ()
        )
        if len(order) < 2 or len(set(order)) != len(order) or count <= 0:
            continue
        prev = buckets.setdefault(len(order), {}).get(order, 0)
        if count > prev:
            buckets[len(order)][order] = count
    queues = {
        length: sorted(entries, key=lambda o: (-entries[o], o))
        for length, entries in buckets.items()
    }
    selected = []
    index = 0
    while len(selected) < budget and queues:
        progressed = False
        for length in sorted(queues):
            queue = queues[length]
            if index < len(queue):
                selected.append(list(queue[index]))
                progressed = True
                if len(selected) == budget:
                    break
            else:
                queues.pop(length, None)
        if not progressed:
            break
        index += 1
    return selected


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
        st = path.stat()
        key = (str(path), st.st_mtime_ns, st.st_size)
        if key not in self.cache:
            value = self._load_cached(key, path)
            self.cache = {k: v for k, v in self.cache.items() if k[0] != str(path)}
            self.cache[key] = value
        value = self.cache[key]
        return value if value["snapshot"] == database_snapshot(self.database) else None

    @staticmethod
    def _load_cached(key, path):
        with _PROCESS_OFFLINE_LOCK:
            cached = _PROCESS_OFFLINE_CACHE.get(key)
            if cached is not None:
                _PROCESS_OFFLINE_CACHE.move_to_end(key)
                return cached[0]
        value = json.loads(path.read_text())
        if value.get("version") != 2:
            raise ValueError("Unsupported offline prefix-statistics version")
        with _PROCESS_OFFLINE_LOCK:
            _PROCESS_OFFLINE_CACHE[key] = (value, False)
            _PROCESS_OFFLINE_CACHE.move_to_end(key)
            while len(_PROCESS_OFFLINE_CACHE) > 64:
                _PROCESS_OFFLINE_CACHE.popitem(last=False)
        return value

    def _validated_relations(self, query, *, required=False):
        from .cache import require_hash_entries

        entries = []
        for relation in query["relations"]:
            if not relation.get("table") or not relation.get("schema"):
                continue
            path = self.path(relation["schema"], relation["table"])
            if not path.exists():
                if required:
                    raise ValueError(
                        "Missing or stale offline prefix/cooccurrence statistics for "
                        f"{relation['schema']}.{relation['table']}; "
                        "run semcache_statistics_build before value-guided search"
                    )
                continue
            st = path.stat()
            cache_key = (str(path), st.st_mtime_ns, st.st_size)
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
            with _PROCESS_OFFLINE_LOCK:
                validated = _PROCESS_OFFLINE_CACHE.get(cache_key, (None, False))[1]
            if not validated:
                require_hash_entries(table["cooccurrence"])
                with _PROCESS_OFFLINE_LOCK:
                    if cache_key in _PROCESS_OFFLINE_CACHE:
                        _PROCESS_OFFLINE_CACHE[cache_key] = (table, True)
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
            name_to_key = {name: key for key, name in columns.items()}
            joint_columns = {}
            for raw_cols in table.get("joint_summaries") or {}:
                cols = json.loads(raw_cols)
                if (
                    isinstance(cols, list)
                    and len(cols) >= 2
                    and all(c in name_to_key for c in cols)
                ):
                    joint_columns[raw_cols] = json.dumps(
                        [name_to_key[c] for c in cols], separators=(",", ":")
                    )
            entries.append((relation, path, st, table, columns, joint_columns))
        return entries

    def for_query(self, query, *, required=False):
        result = {}
        for relation, _, _, table, columns, joint_columns in self._validated_relations(
            query, required=required
        ):
            table_joints = table.get("joint_summaries") or {}
            result[str(relation["node_id"])] = {
                "columns": columns,
                "summaries": {
                    key: table["summaries"][name] for key, name in columns.items()
                },
                "joint_summaries": {
                    joint_key: table_joints[raw_cols]
                    for raw_cols, joint_key in joint_columns.items()
                },
                "cooccurrence": table["cooccurrence"],
            }
        return result

    def for_query_files(self, query, *, required=False):
        files = {}
        row_counts = {}
        for relation, path, st, table, columns, joint_columns in self._validated_relations(
            query, required=required
        ):
            key = str(relation["node_id"])
            files[key] = {
                "path": str(path),
                "mtime_ns": st.st_mtime_ns,
                "size": st.st_size,
                "columns": columns,
                "joint_columns": joint_columns,
            }
            row_counts[key] = table["cooccurrence"]["row_count"]
        return files, row_counts


def build_database(
    database,
    directory,
    build,
    *,
    tables=None,
    config=None,
    parallelism=None,
    joint_orders=None,
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
    if joint_orders is not None:
        command += ["--joint-orders", str(Path(joint_orders).resolve())]
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
