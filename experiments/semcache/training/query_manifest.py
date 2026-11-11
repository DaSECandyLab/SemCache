"""Read frozen query manifests and describe training/testing SQL without generation."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import sqlglot
from planning.settings.configuration import ROOT
from sqlglot import exp

from training.sql_fingerprint import structural_fingerprint

DATASETS = ("estate", "movies_nirvana", "movies_sema", "appstore", "tpch")
PROJECT = ROOT.parent


def frozen_workload_versions(run_dir, current_default):
    """Keep the completed bootstrap's provenance when feedback SQL is updated."""
    run_dir = Path(run_dir)
    current = run_dir / "workloads.json"
    bootstrap = run_dir / "estimated/round-1/workloads.json"
    version = (
        json.loads(current.read_text())["version"]
        if current.exists()
        else current_default
    )
    pretraining = (
        json.loads(bootstrap.read_text())["version"] if bootstrap.exists() else version
    )
    return version, pretraining


def validate_round_workload(payload, manifest):
    """A repeated workload ID cannot substitute for the actual supervised SQL."""
    from planning.state import digest

    expected = {w["id"]: digest(w["sql"]) for w in manifest["workloads"]}
    for record in payload["records"]:
        metadata = record.get("metadata", {})
        workload = metadata.get("workload")
        if (
            workload not in expected
            or metadata.get("workload_sql_digest") != expected[workload]
        ):
            raise ValueError(
                f"Round feedback SQL differs from its frozen workload: {workload}"
            )


def sql_sha256(sql):
    return hashlib.sha256(sql.encode()).hexdigest()


def query_shape(sql):
    """Report syntax counts, never the final total of native logical nodes.

    Scans plus semantic calls form a lower bound; ordinary filter, projection,
    join, aggregate and order nodes are counted only by native binding.
    """
    tree = sqlglot.parse_one(sql, read="duckdb")
    semantic_types = [
        node.name.lower()
        for node in tree.find_all(exp.Anonymous)
        if node.name.lower() in ("semantic_filter", "semantic_project", "semantic_join")
    ]
    functions = Counter(semantic_types)
    relation_count = len(list(tree.find_all(exp.Table)))
    semantic_nodes = sum(functions.values())
    return {
        "semantic_counts": dict(sorted(functions.items())),
        # This is deterministic SQL-tree traversal order, not the physical
        # execution order selected later by an optimizer.
        "semantic_operator_types": semantic_types,
        "semantic_nodes": semantic_nodes,
        "relation_count": relation_count,
        "operator_lower_bound": relation_count + semantic_nodes,
    }


def count_operators(query):
    """Describe a bound query without imposing workload-generation limits."""
    nodes = [
        node for node in query.get("operators", []) if node.get("type") != "EXPLAIN"
    ]
    semantic = sum("semantic_operator_id" in node for node in nodes)
    return {
        "total_operators": len(nodes),
        "semantic_nodes": semantic,
        "relation_nodes": len(nodes) - semantic,
    }


def heldout_manifest(root=None):
    """Read only executable selected SQL; image drafts and PLOP originals stay out."""
    root = Path(root) if root is not None else PROJECT / "benchmarks/queries"
    paths = sorted(root.glob("translated/semcache/*/*.sql"))
    if not paths:
        raise ValueError("Heldout SQL suite is missing")
    return [
        {
            "id": str(path.relative_to(root).with_suffix("")),
            "path": str(path.resolve()),
            "sql_sha256": sql_sha256(path.read_text()),
            "structure_fingerprint": structural_fingerprint(path.read_text()),
        }
        for path in paths
    ]


def ordered_heldout_manifest(manifest):
    """Match the five-system public workload: dataset order, then numeric Q ID."""
    datasets = ("estate", "movies", "movies_sema", "appstore", "tpch")

    def key(item):
        parts = item["id"].split("/")
        dataset = parts[-2] if len(parts) > 1 else ""
        query = parts[-1]
        return (
            datasets.index(dataset) if dataset in datasets else len(datasets),
            int(query[1:]) if query.startswith("Q") and query[1:].isdigit() else 0,
            query,
        )

    return sorted(manifest, key=key)


def load_manifest(path):
    """Read a nonempty versioned catalog and verify its embedded SQL identities."""
    manifest = json.loads(Path(path).read_text())
    workloads = manifest.get("workloads")
    if type(manifest.get("version")) is not int or manifest["version"] < 1:
        raise ValueError("Query manifest needs a positive integer version")
    if not isinstance(workloads, list) or not workloads:
        raise ValueError("Query manifest must contain queries")
    seen = set()
    for workload in workloads:
        identity = workload.get("id")
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("Query manifest IDs must be nonempty and unique")
        seen.add(identity)
        if workload.get("dataset") not in DATASETS:
            raise ValueError(f"Unknown query dataset: {identity}")
        sql = workload.get("sql")
        if (
            not isinstance(sql, str)
            or not sql.strip()
            or sql_sha256(sql) != workload.get("sql_sha256")
        ):
            raise ValueError(f"Query SQL hash differs: {identity}")
    return manifest


def workload_counts(value, datasets=DATASETS):
    """Resolve a common CLI count or an explicit count for each dataset."""
    counts = dict.fromkeys(datasets, value) if type(value) is int else value
    if (
        not isinstance(counts, dict)
        or set(counts) != set(datasets)
        or any(type(n) is not int or n < 1 for n in counts.values())
    ):
        raise ValueError(
            "Query counts must specify a positive integer for each dataset"
        )
    return dict(counts)
