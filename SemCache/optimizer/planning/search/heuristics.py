"""Priority tiers for candidate expansion; dependencies still govern legality.

Priorities include the prerequisites of filters and joins. State-specific
cardinalities, cache effects and input legality remain with the normal planner.
"""

from planning.prompt_layouts import BLOCK_ORDERS
from planning.state.identity import field_key

PRIORITY_POLICY = "predicate_prerequisites_v2"


def operator_prerequisites(problem):
    """Priority dependencies include fixed inputs around reorderable regions."""
    prerequisites = {
        oid: set(op.get("depends_on", [])) for oid, op in problem.all_ops.items()
    }
    for edge in problem.query.get("dependencies", []):
        prerequisites[edge["after"]].add(edge["before"])
    # LogicalQuery can express generated-column dependencies through fields
    # alone. QueryIR exports those edges explicitly, using semantic operator IDs.
    producers = {
        field_key(field): oid
        for oid, op in problem.all_ops.items()
        for field in op.get("produced_fields", [])
    }
    for oid, op in problem.all_ops.items():
        prerequisites[oid].update(
            producers[key]
            for field in op.get("required_columns", [])
            if (key := field_key(field)) in producers and producers[key] != oid
        )
    if problem.native:
        by_node = {op["node_id"]: oid for oid, op in problem.all_ops.items()}

        def boundary(nid):
            region = problem.tree_nodes.get(nid) or problem.predicate_nodes.get(nid)
            if region:
                return {by_node[n] for n in region["nodes"]}
            return {by_node[nid]} if nid in by_node else set()

        for oid, op in problem.all_ops.items():
            nid = op["node_id"]
            region = problem.tree_nodes.get(nid)
            if region:
                required = {field_key(f) for f in op.get("required_columns", [])}
                node = problem.logical[nid]
                for base in region["base_nodes"]:
                    fields = {field_key(f) for f in problem.logical[base]["output_schema"]}
                    if required & fields or (
                        not required and len(node["children"]) == 2
                    ):
                        prerequisites[oid].update(boundary(base))
            elif nid in problem.predicate_nodes:
                prerequisites[oid].update(
                    boundary(problem.predicate_nodes[nid]["base_node_id"])
                )
            else:
                for child in problem.logical[nid]["children"]:
                    prerequisites[oid].update(boundary(child))
    return prerequisites


def action_priorities(problem, mode="heuristic_requests"):
    """Prefer predicates and their prerequisites over unrelated projections."""
    if mode == "value":
        return {}
    if mode not in ("heuristic_requests", "heuristic_value"):
        raise ValueError("Unknown Beam search policy")
    result = {}
    for oid, op in problem.all_ops.items():
        kind = op.get("kind", op.get("type", "")).lower()
        semantic = oid in problem.ops
        if not semantic and kind == "filter":
            group = 0
        elif semantic and "projection" in kind:
            group = 5 if mode == "heuristic_requests" else 3
        elif not semantic and "projection" in kind:
            group = 4 if mode == "heuristic_requests" else 2
        elif mode == "heuristic_value":
            # Relational joins and semantic filters/joins share one group.
            group = 1
        elif not semantic:
            group = 1
        elif "filter" in kind:
            group = 2
        else:
            group = 3
        result[oid] = [group, 0.0, 0.0]
    prerequisites = operator_prerequisites(problem)

    base_groups = {oid: priority[0] for oid, priority in result.items()}
    for oid, op in problem.all_ops.items():
        kind = op.get("kind", op.get("type", "")).lower()
        if "filter" not in kind and "join" not in kind and kind != "cross_product":
            continue
        group = base_groups[oid]
        pending, visited = list(prerequisites[oid]), {oid}
        while pending:
            dependency = pending.pop()
            if dependency in visited:
                continue
            visited.add(dependency)
            result[dependency][0] = min(result[dependency][0], group)
            pending.extend(prerequisites[dependency])
    # A ready predicate wins over a promoted projection in the same tier. The
    # latter is considered only while the predicate still needs its output.
    for oid, priority in result.items():
        priority[1] = base_groups[oid]
    return result


def fixed_model(problem, configured=None):
    """Choose one model for the whole request-guided search, never per state."""
    domains = [problem.domain[oid] for oid in problem.ops]
    if not domains:
        return None
    common = set.intersection(*[{a["model_id"] for a in domain} for domain in domains])
    chosen = configured or "gemma4-e2b"
    if chosen not in common:
        raise ValueError(
            "The fixed request-search model must implement every semantic operator"
        )
    return chosen


GOTDF_BLOCK_ORDER = BLOCK_ORDERS[0]


def fixed_layout(op):
    """GOTDF with the original bound field order on each input side."""
    import copy

    layout = copy.deepcopy(op["default_layout"])
    layout["block_order"] = list(GOTDF_BLOCK_ORDER)
    return layout
