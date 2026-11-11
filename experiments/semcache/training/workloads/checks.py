"""Offline training-query contracts, native budgets and template provenance checks."""

from collections import Counter

import sqlglot
from sqlglot import exp

from training.query_manifest import (
    count_operators,
    heldout_manifest,
    query_shape,
    sql_sha256,
)
from training.sql_fingerprint import FINGERPRINT_VERSION, structural_fingerprint

MAX_TOTAL_OPERATORS = 32
MIN_SEMANTIC_OPERATORS = 3


def workload_operator_limit(workload):
    """Only declared public TPC-H mutations use the larger structural budget."""
    lineage = workload.get("template_lineage", {})
    topology = (
        workload.get("dataset") == "tpch"
        and lineage.get("source") == "public28_light_mutations"
        and lineage.get("heldout_query_used_as_template") is True
        and lineage.get("public_query")
        in {f"tpch/Q{number}" for number in (16, 17, 25, 26, 27, 28, 30)}
    )
    ceiling = 96 if topology else MAX_TOTAL_OPERATORS
    declared = workload.get("operator_budget", {}).get("maximum_total", ceiling)
    if type(declared) is not int or not 1 <= declared <= ceiling:
        raise ValueError("Invalid workload operator budget")
    return declared


def validate_native_budget(
    query, *, maximum=MAX_TOTAL_OPERATORS, minimum_semantic=MIN_SEMANTIC_OPERATORS
):
    """Enforce operator limits only while creating a generated workload."""
    counts = count_operators(query)
    total, semantic = counts["total_operators"], counts["semantic_nodes"]
    if total > maximum:
        raise ValueError(f"Native QueryIR has {total} operators; maximum is {maximum}")
    if type(minimum_semantic) is not int or minimum_semantic < 1:
        raise ValueError("Invalid minimum semantic operator count")
    if semantic < minimum_semantic:
        raise ValueError(
            f"Native QueryIR has {semantic} semantic operators; minimum is {minimum_semantic}"
        )
    return counts


def validate_output_contract(query):
    """Require stable identity outputs and no semantic projections or aggregates."""
    contract = query.get("output_contract", {})
    if contract.get("version") == "identity-only-v1":
        identities = contract.get("identity_columns")
        if (
            not isinstance(identities, list)
            or not identities
            or any(not isinstance(name, str) or not name for name in identities)
            or len(set(identities)) != len(identities)
            or contract.get("generated_columns") != {}
            or contract.get("generated_group_keys") != []
        ):
            raise ValueError(
                "Identity outputs require distinct stable IDs and no generated fields"
            )
        tree = sqlglot.parse_one(query["sql"], read="duckdb")
        if any(
            f.name.lower() == "semantic_project" for f in tree.find_all(exp.Anonymous)
        ):
            raise ValueError(
                "Identity-only training SQL must not contain semantic projection"
            )
        names = [p.alias_or_name for p in tree.selects]
        if names != identities or any(p.is_star for p in tree.selects):
            raise ValueError(
                "Identity contract differs from explicit SQL result fields"
            )
        if any(
            tree.find(kind) is not None
            for kind in (exp.AggFunc, exp.Group, exp.Having, exp.Window)
        ):
            raise ValueError(
                "Identity-only training SQL must not contain aggregates or windows"
            )
        if any(p.find(exp.Subquery) for p in tree.selects):
            raise ValueError("Identity outputs must not be scalar subquery results")
        return 0
    raise ValueError("Training SQL requires an identity-only output contract")


def validate_bound_output(workload, query):
    """Verify final result types against the actual native binding, not metadata."""
    validate_output_contract(workload)
    nodes = [node for node in query["operators"] if node["type"] != "EXPLAIN"]
    children = {child for node in nodes for child in node.get("children", [])}
    roots = [node for node in nodes if node["node_id"] not in children]
    if len(roots) != 1:
        raise ValueError("Generated workload must bind to one result root")
    names = [
        p.alias_or_name
        for p in sqlglot.parse_one(workload["sql"], read="duckdb").selects
    ]
    fields = roots[0].get("output_schema", [])
    if len(fields) != len(names):
        raise ValueError("Bound result schema differs from the SQL output columns")
    schema = [
        {"name": name, "type": field["type"].upper()}
        for name, field in zip(names, fields)
    ]
    integer_types = {
        "TINYINT",
        "SMALLINT",
        "INTEGER",
        "BIGINT",
        "HUGEINT",
        "UTINYINT",
        "USMALLINT",
        "UINTEGER",
        "UBIGINT",
        "UHUGEINT",
    }
    for field in schema:
        if field["type"] not in integer_types:
            raise ValueError(f"Non-closed native result column: {field}")
    return schema


def audit_workloads(workloads, *, heldout=None, generating=False, template_rewrites=()):
    """Fail before collection on exact duplicates or structural split leakage."""
    heldout = heldout_manifest() if heldout is None else heldout
    heldout_structures = {entry["structure_fingerprint"] for entry in heldout}
    seen_sql = set()
    fingerprints = {}
    reused_templates = set()
    shared_structures = []
    split_structures = {"train": set(), "validation": set()}
    split_groups = {"train": set(), "validation": set()}
    for query in workloads:
        split = query["split"]
        if split not in split_structures:
            raise ValueError("Generated workloads require a train or validation split")
        digest = sql_sha256(query["sql"])
        if digest not in fingerprints:
            fingerprints[digest] = structural_fingerprint(query["sql"])
        fingerprint = fingerprints[digest]
        if (
            query.get("sql_sha256", digest) != digest
            or query.get("structure_fingerprint", fingerprint) != fingerprint
        ):
            raise ValueError("Workload fingerprint metadata differs from its SQL")
        shape = query_shape(query["sql"])
        maximum = workload_operator_limit(query)
        lineage = query.get("template_lineage", {})
        reused = lineage.get("heldout_query_used_as_template") is True
        if reused:
            public = lineage.get("public_query")
            originals = [
                item for item in heldout if item["id"].endswith("/" + str(public))
            ]
            if (
                lineage.get("source") != "public28_light_mutations"
                or query.get("dataset")
                != {"movies": "movies_nirvana"}.get(
                    str(public).split("/")[0], str(public).split("/")[0]
                )
                or len(originals) != 1
                or (
                    lineage.get("canonical_sql_sha256") != originals[0]["sql_sha256"]
                    and (
                        public,
                        lineage.get("canonical_sql_sha256"),
                        originals[0]["sql_sha256"],
                    )
                    not in template_rewrites
                )
                or digest == originals[0]["sql_sha256"]
            ):
                raise ValueError(
                    "Public topology reuse lacks a valid changed-query lineage"
                )
            reused_templates.add(public)
        if generating and shape["operator_lower_bound"] > maximum:
            raise ValueError(
                f"Workload exceeds the {maximum}-operator budget: "
                f"{query.get('id', '<unknown>')} has "
                f"at least {shape['operator_lower_bound']} operators"
            )
        minimum_semantic = MIN_SEMANTIC_OPERATORS if generating else 1
        if shape["semantic_nodes"] < minimum_semantic:
            raise ValueError(
                f"A training query must contain at least {minimum_semantic} semantic operators"
            )
        keys = ["semantic_nodes", "relation_count", "operator_lower_bound"]
        if generating or "semantic_operator_types" in query:
            keys.append("semantic_operator_types")
        for key in keys:
            if query.get(key) != shape[key]:
                raise ValueError(f"Workload operator metadata differs for {key}")
        if generating:
            validate_output_contract(query)
        key = (query["dataset"], digest)
        if key in seen_sql:
            raise ValueError("Duplicate SQL within a dataset")
        seen_sql.add(key)
        if fingerprint in heldout_structures:
            if not reused:
                raise ValueError("Generated SQL matches a heldout query structure")
            shared_structures.append(query["id"])
        split_structures[split].add(fingerprint)
        split_groups[split].add(query.get("template_id", query["family"]))
    if split_structures["train"] & split_structures["validation"]:
        raise ValueError("Train and validation contain the same SQL structure")
    if split_groups["train"] & split_groups["validation"]:
        raise ValueError("Train and validation contain the same template family")
    return {
        "fingerprint_version": FINGERPRINT_VERSION,
        "heldout_queries": len(heldout),
        "heldout_disjoint": not shared_structures,
        "heldout_template_disjoint": not reused_templates,
        "heldout_templates_reused": sorted(reused_templates),
        "authorized_shared_structure_queries": shared_structures,
        "sql_unique_per_dataset": True,
        "train_structures": sorted(split_structures["train"]),
        "validation_structures": sorted(split_structures["validation"]),
        "train_groups": sorted(split_groups["train"]),
        "validation_groups": sorted(split_groups["validation"]),
        "dataset_queries": dict(
            sorted(Counter(q["dataset"] for q in workloads).items())
        ),
        "evaluation_feedback_used": False,
        **(
            {
                "output_contracts": sorted(
                    {q["output_contract"]["version"] for q in workloads}
                ),
                "closed_output_queries": len(workloads),
                "aggregate_free": all(
                    sqlglot.parse_one(q["sql"], read="duckdb").find(exp.AggFunc) is None
                    for q in workloads
                ),
                "semantic_projection_free": all(
                    query_shape(q["sql"])["semantic_counts"].get("semantic_project", 0)
                    == 0
                    for q in workloads
                ),
                "free_text_output": False,
            }
            if generating
            else {}
        ),
    }
