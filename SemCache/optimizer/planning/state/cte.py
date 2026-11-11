"""Connect bound CTE scans to their single materialized producer for sampling."""

import json


def connect_references(query):
    nodes = {node["node_id"]: node for node in query["operators"]}
    definitions = {
        node["cte_index"]: node
        for node in nodes.values()
        if node.get("type") == "CTE" and "cte_index" in node
    }

    def identifier(field):
        # Bindings come from DuckDB, never from human-readable EXPLAIN text.
        return '"' + json.dumps(field["binding"], separators=(",", ":")) + '"'

    for node in nodes.values():
        if node.get("type") == "CTE":
            node["sampling_passthrough_input"] = 1
            node["sampling_sql"] = "SELECT * FROM $RIGHT"
        elif node.get("type") == "CTE_SCAN":
            definition = definitions.get(node.get("cte_index"))
            if definition is None:
                raise ValueError("CTE scan has no bound materialized producer")
            producer = nodes[definition["children"][0]]
            fields = producer["output_schema"]
            outputs = node["output_schema"]
            if len(fields) != len(outputs) or any(
                left["type"] != right["type"]
                for left, right in zip(fields, outputs, strict=True)
            ):
                raise ValueError("CTE scan schema differs from its bound producer")
            node["children"] = [producer["node_id"]]
            node["required_columns"] = [field["binding"] for field in fields]
            columns = ["__sample_id"] + [
                identifier(left) + " AS " + identifier(right)
                for left, right in zip(fields, outputs, strict=True)
            ]
            node["sampling_sql"] = "SELECT " + ",".join(columns) + " FROM $INPUT"
