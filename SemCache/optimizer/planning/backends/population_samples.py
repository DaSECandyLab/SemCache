"""Bound, model-free SQL probes for relational filter and inner-join rates."""

import json

from .relational_estimates import (
    _SEMANTIC_CALL,
    _TRANSPORT_ID,
    _estimation_filter_sql,
    _field,
)


def prepare_population_samples(problem):
    """Compile independent probes; their execution overlaps semantic sampling.

    A join samples one input and matches it against the full other input.
    Sampling two tiny independent pools loses rare matching keys. Predicates
    in the same reorderable region are pushed to their owning base side,
    consistent with both policies placing ready relational filters first.
    """
    sources = {r["node_id"]: r for r in problem.query.get("relations", [])}
    regions = {
        nid: region["region_id"]
        for region in problem.query.get("tree_regions", [])
        for nid in region["nodes"]
    }
    filters = []
    for op in problem.all_ops.values():
        op.pop("population_sample", None)
        if op.get("kind", op.get("type")) != "FILTER":
            continue
        required = set(op.get("sampling_required_fields", []))
        sql = op.get("sampling_row_local_sql", "")
        owners = [
            r for r in sources.values()
            if required and required <= {
                json.dumps(c["binding"], separators=(",", ":"))
                for c in r.get("columns", []) if "binding" in c
            }
        ]
        if not op.get("sampling_row_local") or not sql or len(owners) != 1:
            continue
        source_sql = _TRANSPORT_ID.sub(
            "NULL::VARCHAR __sample_id", owners[0]["source_sql"]
        )
        op["population_sample"] = {
            "kind": "filter", "inputs": [source_sql], "sql": sql,
            "conditioned_filters": [],
        }
        filters.append((op, required, sql))

    def supported(nid, visited):
        if nid in visited:
            return True
        visited.add(nid)
        node = problem.logical[nid]
        kind = node.get("kind", node.get("type", ""))
        if nid not in sources and kind not in (
            "FILTER", "SEMANTIC_FILTER", "PROJECTION", "ORDER_BY",
            "COMPARISON_JOIN", "ANY_JOIN",
        ):
            return False
        if "JOIN" in kind and node.get("join_type", "INNER") != "INNER":
            return False
        return all(supported(c, visited) for c in node.get("children", []))

    for op in problem.all_ops.values():
        kind = op.get("kind", op.get("type", ""))
        if kind not in ("COMPARISON_JOIN", "ANY_JOIN"):
            continue
        if op.get("join_type", "INNER") != "INNER":
            continue
        nid = op["node_id"]
        children = problem.logical[nid].get("children", [])
        inputs = op.get("population_input_sql", [])
        template = op.get("sql", op.get("sampling_sql", ""))
        if len(children) != 2 or len(inputs) != 2 or "$LEFT" not in template:
            continue
        if not supported(nid, set()) or _SEMANTIC_CALL.search(template):
            continue
        inputs, conditioned = list(inputs), []
        for side, child in enumerate(children):
            fields = {_field(f) for f in problem.logical[child]["output_schema"]}
            for predicate, required, sql in filters:
                if (
                    nid in regions
                    and regions.get(predicate["node_id"]) == regions[nid]
                    and required <= fields
                ):
                    inputs[side] = sql.replace("$INPUT", "(" + inputs[side] + ")")
                    conditioned.append(predicate["node_id"])
        op["population_sample"] = {
            "kind": "join", "inputs": inputs, "sql": template,
            "conditioned_filters": sorted(set(conditioned)),
        }

    for op in problem.all_ops.values():
        if "population_sample" in op:
            spec = op["population_sample"]
            # Average local rates only after giving EXPLAIN exactly the same
            # base inputs and ready relational filters as the random probe.
            op["population_input_sql"] = list(spec["inputs"])
            op["population_estimate_sql"] = population_estimate_sql(spec)


def population_estimate_sql(spec):
    """EXPLAIN counterpart of the probe, using full inputs rather than a sample."""
    inputs = spec["inputs"]
    ctes = [f"__pe_left AS NOT MATERIALIZED ({inputs[0]})"]
    template = _TRANSPORT_ID.sub("NULL::VARCHAR __sample_id", spec["sql"])
    if spec["kind"] == "join":
        ctes.append(f"__pe_right AS NOT MATERIALIZED ({inputs[1]})")
        output = template.replace("$LEFT", "__pe_left").replace("$RIGHT", "__pe_right")
    else:
        output = template.replace("$INPUT", "__pe_left")
    if _SEMANTIC_CALL.search(output):
        raise ValueError("Relational estimate must not invoke a semantic function")
    return _estimation_filter_sql("WITH " + ",".join(ctes) + " " + output)


def population_probe_sql(spec, rows, seed):
    """Return four counts, without materializing the join's result records."""
    if type(rows) is not int or rows < 1 or type(seed) is not int:
        raise ValueError("Relational sample rows and seed must be integers")
    inputs = spec["inputs"]
    ctes = [f"__pc_left AS MATERIALIZED ({inputs[0]})"]
    ctes.append(
        "__pc_sample AS MATERIALIZED (SELECT * FROM __pc_left "
        f"USING SAMPLE reservoir({rows} ROWS) REPEATABLE ({seed % (2**31)}))"
    )
    if spec["kind"] == "join":
        ctes.append(f"__pc_right AS MATERIALIZED ({inputs[1]})")
        output = spec["sql"].replace("$LEFT", "__pc_sample").replace(
            "$RIGHT", "__pc_right"
        )
        right = "(SELECT count(*) FROM __pc_right)"
    else:
        output = spec["sql"].replace("$INPUT", "__pc_sample")
        right = "1::BIGINT"
    if _SEMANTIC_CALL.search(output):
        raise ValueError("Relational probe must not invoke a semantic function")
    return (
        "WITH " + ",".join(ctes)
        + " SELECT (SELECT count(*) FROM __pc_left),"
        + right + ",(SELECT count(*) FROM __pc_sample),"
        + "(SELECT count(*) FROM (" + output + ") __pc_output)"
    )
