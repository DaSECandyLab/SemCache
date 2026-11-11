"""QueryIR / PlanSpec contracts shared by search, replay and socket transport."""


def validate_spec(query, spec):
    """Mirror the C++ trust-boundary checks before returning a PlanSpec."""

    required = {"version", "kind", "query_id", "actions"}
    if not isinstance(spec, dict) or not required <= set(spec) <= required | {
        "predicate_order",
        "tree_choices",
    }:
        raise ValueError("PlanSpec keys mismatch")
    if (
        type(spec["version"]) is not int
        or spec["version"] != 1
        or spec["kind"] != "PlanSpec"
        or spec["query_id"] != query["query_id"]
    ):
        raise ValueError("PlanSpec identity mismatch")
    if not isinstance(spec["actions"], list):
        raise ValueError("PlanSpec actions must be an array")  # noqa: TRY004 — uniform protocol error
    candidates = {op["operator_id"]: op for op in query["semantic_operators"]}
    order = []
    for action in spec["actions"]:
        if not isinstance(action, dict) or set(action) != {
            "operator_id",
            "model_id",
            "layout",
        }:
            raise ValueError("PlanSpec must contain semantic decisions only")
        operator_id = action["operator_id"]
        if (
            type(operator_id) is not int
            or operator_id not in candidates
            or operator_id in order
        ):
            raise ValueError("unknown or duplicate operator_id")
        order.append(operator_id)
        candidate = candidates[operator_id]
        if (
            not isinstance(action["model_id"], str)
            or action["model_id"] not in candidate["models"]
        ):
            raise ValueError("model is not a candidate")
        layout = action["layout"]
        default = candidate["default_layout"]
        if not isinstance(layout, dict) or set(layout) != set(default):
            raise ValueError("layout keys mismatch")
        for key, fields in default.items():
            if (
                not isinstance(layout[key], list)
                or not all(isinstance(f, str) for f in layout[key])
                or sorted(layout[key]) != sorted(fields)
            ):
                raise ValueError(f"invalid layout permutation: {key}")
    if set(order) != candidates.keys():
        raise ValueError("missing/extra actions")
    if "tree_choices" not in spec:
        offset = 0
        for group in query["order_groups"]:
            if sorted(order[offset : offset + len(group)]) != sorted(group):
                raise ValueError("order violates dependencies or supported regions")
            offset += len(group)
    if "predicate_order" in spec:
        if not isinstance(spec["predicate_order"], list):
            raise ValueError("Predicate region order must be an array")
        regions = {r["region_id"]: r for r in query.get("predicate_regions", [])}
        logical = {n["node_id"]: n for n in query["operators"]}
        seen = set()
        for entry in spec["predicate_order"]:
            if (
                not isinstance(entry, dict)
                or set(entry) != {"region_id", "nodes"}
                or type(entry["region_id"]) is not int
                or entry["region_id"] not in regions
                or entry["region_id"] in seen
            ):
                raise ValueError("Invalid predicate region order")
            seen.add(entry["region_id"])
            region, nodes = regions[entry["region_id"]], entry["nodes"]
            if (
                not isinstance(nodes, list)
                or any(type(n) is not int for n in nodes)
                or sorted(nodes) != sorted(region["nodes"])
            ):
                raise ValueError("Predicate region must contain each node exactly once")
            from planning.state.trees import field_key

            available = {
                field_key(c) for c in logical[region["base_node_id"]]["output_schema"]
            }
            semantic_order = []
            for nid in nodes:
                node = logical[nid]
                if (
                    not {field_key(c) for c in node.get("required_columns", [])}
                    <= available
                ):
                    raise ValueError("Predicate order uses a field before its producer")
                old_input = {
                    field_key(c)
                    for c in logical[node["children"][0]].get("output_schema", [])
                }
                available.update(
                    field_key(c)
                    for c in node["output_schema"]
                    if field_key(c) not in old_input
                )
                if "semantic_operator_id" in node:
                    semantic_order.append(node["semantic_operator_id"])
            if semantic_order != [oid for oid in order if oid in semantic_order]:
                raise ValueError("Predicate order disagrees with semantic stage order")
        if seen != regions.keys():
            raise ValueError("Missing predicate region order")
    if "tree_choices" in spec:
        from planning.state.trees import validate_choices

        validate_choices(query, spec)
