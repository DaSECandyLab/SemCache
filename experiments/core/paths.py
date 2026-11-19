"""Workspace paths shared by experiment tools; independent of cwd."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUERY_ROOT = ROOT / "benchmarks/queries"
TRANSLATED = QUERY_ROOT / "translated"
QUERY_SYSTEMS = ("semcache", "sema", "plop", "abacus", "nirvana", "lotus")
_SQL_SYSTEMS = frozenset({"semcache", "sema", "plop"})
_AGGREGATES = {
    "estate/Q9": {"function": "AVG", "column": "Price"},
    "estate/Q10": {"function": "MIN", "column": "Price"},
    "estate/Q11": {"function": "MIN", "column": "Price"},
    "estate/Q12": {"function": "AVG", "column": "Price"},
    "movies/Q9": {"function": "MAX", "column": "IMDB_rating"},
    "movies/Q10": {"function": "COUNT", "column": None},
    "movies/Q11": {"function": "AVG", "column": "RuntimeMinutes"},
}


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_sha256(value):
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _query_sort_key(dataset, query):
    suffix = query[1:] if query.startswith("Q") and query[1:].isdigit() else None
    return (dataset, 0 if suffix is not None else 1, int(suffix) if suffix is not None else query)


def load_query_selection(query_root=QUERY_ROOT):
    """Load `manifest.json` or synthesize it directly from active query files."""
    query_root = Path(query_root)
    manifest_path = query_root / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    queries = []
    for path in sorted((query_root / "original").glob("*/*/*")):
        if not path.is_file() or path.suffix not in {".py", ".sql"}:
            continue
        rel = path.relative_to(query_root)
        _, dir0, origin, _ = rel.parts
        dataset = "movies_sema" if (dir0 == "movies" and origin == "sema") else dir0
        query = path.stem
        query_id = f"{dataset}/{query}"
        sql_rel = f"translated/semcache/{dataset}/{query}.sql"
        queries.append(
            {
                "id": query_id,
                "dataset": dataset,
                "query": query,
                "original": rel.as_posix(),
                "original_sha256": _file_sha256(path),
                "sql": sql_rel,
                "sql_sha256": _file_sha256(query_root / sql_rel),
            }
        )
    queries.sort(key=lambda item: _query_sort_key(item["dataset"], item["query"]))
    return {"queries": queries}


def query_selection_digest(query_root=QUERY_ROOT):
    query_root = Path(query_root)
    manifest_path = query_root / "manifest.json"
    if manifest_path.exists():
        return _file_sha256(manifest_path)
    return _json_sha256(load_query_selection(query_root))


def load_translated_manifest(query_root=QUERY_ROOT):
    """Load `translated/manifest.json` or synthesize it from active source & translated files."""
    query_root = Path(query_root)
    manifest_path = query_root / "translated/manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    queries = load_query_selection(query_root)["queries"]
    records = []
    for item in queries:
        query_id = item["id"]
        dataset = item["dataset"]
        query = item["query"]
        original = item["original"]
        agg = _AGGREGATES.get(query_id)
        numeric_cols = (
            ["Price"]
            if (agg and query_id.startswith("estate/"))
            else (["RuntimeMinutes"] if query_id == "movies/Q11" else [])
        )
        contract = (
            "selected28-extension-v1"
            if query_id == "movies_sema/Q10"
            else "selected19-common-numeric-v4"
        )
        for system in QUERY_SYSTEMS:
            suffix = ".sql" if system in _SQL_SYSTEMS else ".py"
            query_file = f"translated/{system}/{dataset}/{query}{suffix}"
            query_path = query_root / query_file
            if system in _SQL_SYSTEMS:
                execution_mode = "native_sql"
            elif original.endswith(".py") and not agg:
                execution_mode = "native_semantic_pipeline"
            elif original.endswith(".py"):
                execution_mode = "native_semantic_pipeline_with_common_numeric_host"
            elif query_id == "movies_sema/Q10":
                execution_mode = "native_semantic_pipeline_with_relational_prefix"
            elif system == "lotus":
                execution_mode = "relational_host_lotus_native_semantic_api"
            else:
                execution_mode = "relational_host_native_semantic_api"
            record = {
                "system": system,
                "id": query_id,
                "source": original,
                "source_sha256": item["original_sha256"],
                "contract": contract,
                "execution_mode": execution_mode,
                "aggregate": dict(agg) if agg is not None else None,
                "numeric_projection_columns": list(numeric_cols),
                "query_file": query_file,
                "sha256": _file_sha256(query_path),
            }
            meta_rel = f"translated/{system}/{dataset}/{query}.json"
            meta_path = query_root / meta_rel
            if meta_path.exists():
                record["metadata_file"] = meta_rel
                record["metadata_sha256"] = _file_sha256(meta_path)
            records.append(record)
    return {
        "contract": "selected19-common-numeric-v4",
        "query_count_per_system": dict.fromkeys(QUERY_SYSTEMS, len(queries)),
        "source_reduce_rewrite_count": sum(item["id"] in _AGGREGATES for item in queries),
        "records": records,
    }


def translated_manifest_digest(query_root=QUERY_ROOT):
    query_root = Path(query_root)
    manifest_path = query_root / "translated/manifest.json"
    if manifest_path.exists():
        return _file_sha256(manifest_path)
    return _json_sha256(load_translated_manifest(query_root))


def experiment_sources(root=ROOT):
    """Enumerate executable helpers recursively, excluding tests and artifacts."""
    return sorted(
        path
        for package in ("core", "system_comparison", "validation")
        for path in (root / "experiments" / package).rglob("*.py")
    )
