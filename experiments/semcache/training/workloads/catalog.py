"""Compile the approved SQL definitions into a reproducible training catalog."""

import copy
import re
from collections import Counter
from pathlib import Path

import sqlglot
from sqlglot import exp

from training.artifacts import save
from training.query_manifest import load_manifest, query_shape, sql_sha256
from training.sql_fingerprint import structural_fingerprint
from training.workloads.checks import audit_workloads, validate_output_contract
from training.workloads.provenance import load_template_rewrites

PROTOCOL = "fixed-sql-definitions-v1"


def definition_path(directory, workload):
    """Resolve only a dataset's owned SQL, never a path supplied inside SQL text."""
    dataset, identity = workload["dataset"], workload["id"]
    if any(
        Path(value).name != value or value in (".", "..")
        for value in (dataset, identity)
    ):
        raise ValueError(f"Invalid SQL definition path: {identity}")
    path = (Path(directory) / dataset / (identity + ".sql")).resolve()
    if not path.is_relative_to(Path(directory).resolve()):
        raise ValueError(f"SQL definition leaves the catalog: {identity}")
    return path


def compile_query(workload, sql):
    """Rebuild SQL-derived metadata while retaining the executed SQL hash and evidence."""
    # Allow one extra trailing newline while preserving the manifest SQL definition.
    if sql.strip() != workload["sql"].strip():
        raise ValueError(
            f"SQL definition differs from its versioned manifest: {workload['id']}"
        )
    query = copy.deepcopy(workload)
    tree = sqlglot.parse_one(query["sql"], read="duckdb")
    identities = [projection.alias_or_name for projection in tree.selects]
    query["output_contract"] = {
        "version": "identity-only-v1",
        "identity_columns": identities,
        "generated_columns": {},
        "generated_group_keys": [],
    }
    validate_output_contract(query)
    query.update(query_shape(query["sql"]))
    query["sql_sha256"] = sql_sha256(query["sql"])
    query["structure_fingerprint"] = structural_fingerprint(query["sql"])
    families = {
        task["task"]: family
        for task, family in zip(
            workload.get("semantic_tasks", []), workload.get("predicate_families", [])
        )
    }
    tasks = []
    origin = query.get("template_id", query["id"])
    variant = query.get("template_lineage", {}).get("variant", 1)
    for node in tree.find_all(exp.Anonymous):
        kind = node.name.lower()
        if kind not in ("semantic_filter", "semantic_join"):
            continue
        task = node.expressions[0]
        arguments = node.expressions[1:]
        if not isinstance(task, exp.Literal) or not task.is_string:
            raise ValueError(f"Semantic task must be a literal: {query['id']}")
        if {int(index) for index in re.findall(r"\{(\d+)\}", task.this)} != set(
            range(len(arguments))
        ):
            raise ValueError(f"Semantic field references differ: {query['id']}")
        tasks.append(
            {
                "task_id": f"{origin}/v{variant}/{len(tasks)}",
                "kind": kind,
                "task": task.this,
                "input_fields": [
                    column.sql(dialect="duckdb")
                    for argument in arguments
                    for column in argument.find_all(exp.Column)
                ],
                "field_expressions": [
                    argument.sql(dialect="duckdb") for argument in arguments
                ],
            }
        )
    query["semantic_tasks"] = tasks
    query["tasks"] = [task["task"] for task in tasks]
    query["input_fields"] = list(
        dict.fromkeys(field for task in tasks for field in task["input_fields"])
    )
    query["predicate_families"] = [
        families.get(task["task"], task["kind"]) for task in tasks
    ]
    query["source_tables"] = list(
        dict.fromkeys(
            table.name
            for table in tree.find_all(exp.Table)
            if table.name.startswith("raw_")
        )
    )
    if not query["source_tables"]:
        raise ValueError(f"Training SQL has no source tables: {query['id']}")
    query["operator_budget"] = {
        **query.get("operator_budget", {}),
        "minimum_semantic": 3,
    }
    return query


def compile_catalog(manifest_path):
    """Read final SQL files, including rewritten TPC-H, without template regeneration."""
    manifest_path = Path(manifest_path).resolve()
    payload = load_manifest(manifest_path)
    compiled = copy.deepcopy(payload)
    compiled["workloads"] = [
        compile_query(query, definition_path(manifest_path.parent, query).read_text())
        for query in payload["workloads"]
    ]
    requirements = compiled.setdefault("requirements", {})
    requirements["total_queries"] = len(compiled["workloads"])
    requirements["queries_per_dataset"] = dict(
        sorted(Counter(query["dataset"] for query in compiled["workloads"]).items())
    )
    counts = Counter(
        query.get("template_id", query["id"]) for query in compiled["workloads"]
    )
    requirements["variants_per_template"] = sorted(set(counts.values()))
    compiled["catalog_protocol"] = PROTOCOL
    audit = audit_workloads(
        compiled["workloads"],
        generating=True,
        template_rewrites=load_template_rewrites(manifest_path.parent),
    )
    return compiled, audit


def publish_catalog(manifest_path, output, payload, audit):
    """Publish the exact approved SQL bytes; keep existing definitions untouched."""
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    for query in payload["workloads"]:
        source = definition_path(manifest_path.parent, query)
        destination = definition_path(output, query)
        if source == destination:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        temporary = destination.with_suffix(".sql.tmp")
        temporary.write_bytes(data)
        temporary.replace(destination)
    proof = manifest_path.parent / "template-rewrites.json"
    if proof.exists() and proof.parent != output:
        destination = output / proof.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(proof.read_bytes())
    save(output / "workloads.json", payload)
    save(output / "audit.json", audit)
