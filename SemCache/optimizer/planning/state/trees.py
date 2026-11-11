"""Native bound SPJ regions: schema legality and executable tree contracts."""

import itertools
from contextlib import contextmanager
from contextvars import ContextVar

_VALIDATION_CACHE = ContextVar("plan_spec_validation_cache", default=None)


@contextmanager
def cached_validation():
    """Reuse immutable query bindings while exporting one completed batch."""
    if _VALIDATION_CACHE.get() is not None:
        yield
        return
    token = _VALIDATION_CACHE.set({"fields": {}, "produced": {}})
    try:
        yield
    finally:
        _VALIDATION_CACHE.reset(token)


def field_key(column):
    cache = _VALIDATION_CACHE.get()
    if cache is None:
        from planning.state import field_key as key

        return key(column)
    identity = id(column)
    cached = cache["fields"].get(identity)
    if cached is None:
        from planning.state import field_key as key

        cached = cache["fields"][identity] = (column, key(column))
    return cached[1]


def produced(logical, node):
    cache = _VALIDATION_CACHE.get()
    if cache is not None:
        children = tuple(logical[child] for child in node["children"])
        identity = (id(node), tuple(id(child) for child in children))
        if identity in cache["produced"]:
            return list(cache["produced"][identity][2])
    original_inputs = {
        field_key(f)
        for child in node["children"]
        for f in logical[child]["output_schema"]
    }
    result = [f for f in node["output_schema"] if field_key(f) not in original_inputs]
    if cache is not None:
        cache["produced"][identity] = (children, node, tuple(result))
    return result


def output_fields(logical, nid, inputs):
    node = logical[nid]
    return [f for side in inputs for f in side] + produced(logical, node)


def roles_valid(node, inputs):
    if len(inputs) != 2:
        return True
    fields = [{field_key(f) for f in side} for side in inputs]
    left = {field_key(f) for f in node.get("left_required_columns", [])}
    right = {field_key(f) for f in node.get("right_required_columns", [])}
    return (
        left <= fields[0]
        and right <= fields[1]
        or node.get("join_commutative", False)
        and left <= fields[1]
        and right <= fields[0]
    )


def pair_source_valid(logical, node, root, graph, fields):
    """Unary semantic joins consume matched pairs with their original roles."""
    roles = node.get("candidate_pair_columns")
    if not roles:
        return True
    if len(roles) != 2:
        return False
    current = root
    candidate_pipeline = False
    for _ in range(len(logical)):
        source = logical[current]
        children = graph.get(current, source["children"])
        if (
            source["type"] in ("FILTER", "SEMANTIC_FILTER", "MODEL_BINDING_ORDER")
            and len(children) == 1
        ):
            candidate_pipeline |= source["type"] != "FILTER"
            current = children[0]
            continue
        if (
            source["type"] in ("COMPARISON_JOIN", "ANY_JOIN")
            or source["type"] == "CROSS_PRODUCT"
            and candidate_pipeline
        ) and len(children) == 2:
            available = {
                field_key(f)
                for child in children
                for f in fields.get(child, logical[child]["output_schema"])
            }
            return all({field_key(f) for f in role} <= available for role in roles)
        return (
            source["type"] in ("COMPARISON_JOIN", "ANY_JOIN", "CROSS_PRODUCT")
            and len(children) == 2
            and all(
                {field_key(f) for f in role}
                <= {
                    field_key(f)
                    for f in fields.get(child, logical[child]["output_schema"])
                }
                for role, child in zip(roles, children, strict=True)
            )
        )
    return False


def canonical_join_inputs(problem, state, oid, choices):
    """Deduplicate mirrored ordinary joins when generating new actions only."""
    op = problem.all_ops[oid]
    if oid in problem.ops:
        return choices
    node = problem.logical[op["node_id"]]
    if not (
        node.get("join_commutative", False)
        if problem.native
        else op.get("kind", "").lower() == "join"
    ):
        return choices
    roots = {
        n.node_id: n
        for n in state.partial_plans.nodes
        if n.node_id in state.partial_plans.roots
    }

    def rank(ids):
        if problem.native:
            left = {field_key(f) for f in node.get("left_required_columns", [])}
            right = {field_key(f) for f in node.get("right_required_columns", [])}
            direct = left <= {field_key(f) for f in roots[ids[0]].fields} and right <= {
                field_key(f) for f in roots[ids[1]].fields
            }
        else:
            direct = (
                set(op.get("left_scope", [])) <= roots[ids[0]].relations
                and set(op.get("right_scope", [])) <= roots[ids[1]].relations
            )
        return not direct, *ids

    result, pairs = [], {}
    for ids in choices:
        if len(ids) != 2:
            result.append(ids)
            continue
        key = tuple(sorted(ids))
        if key in pairs:
            index = pairs[key]
            if rank(ids) < rank(result[index]):
                result[index] = ids
        else:
            pairs[key] = len(result)
            result.append(ids)
    return result


def pair_target(problem, state, oid, ids):
    """A cheap binary filter builds candidates for one still-pending semantic join."""
    node = problem.logical[problem.all_ops[oid]["node_id"]]
    if len(ids) != 2 or node.get("type", node.get("kind", "")).upper() != "FILTER":
        return None
    roots = {n.node_id: n for n in state.partial_plans.nodes}
    fields = [{field_key(f) for f in roots[n].fields} for n in ids]
    required = {field_key(f) for f in node.get("required_columns", [])}
    if any(roots[n].pending_join is not None for n in ids) or any(
        not required & side for side in fields
    ):
        return None
    done = {problem.all_ops[i]["node_id"] for i in state.done}
    for target in node.get("pair_targets", []):
        if target in done:
            continue
        target_node = problem.logical[target]
        if not {
            field_key(f) for f in target_node.get("required_columns", [])
        } <= set.union(*fields):
            continue
        valid = (
            roles_valid(target_node, [roots[n].fields for n in ids])
            if problem.native
            else (
                set(target_node.get("left_scope", [])) <= roots[ids[0]].relations
                and set(target_node.get("right_scope", [])) <= roots[ids[1]].relations
            )
        )
        if valid:
            return target
    return None


def preserves_pending_joins(problem, state, oid, fields, relations, pair_for, inputs):
    """Do not merge inputs that a remaining binary predicate must keep separate."""
    fields = {field_key(f) for f in fields}
    for pending, op in problem.all_ops.items():
        if pending == oid or pending in state.done:
            continue
        if op["node_id"] == pair_for:
            continue
        if problem.native:
            node = problem.logical[op["node_id"]]
            left = {field_key(f) for f in node.get("left_required_columns", [])}
            right = {field_key(f) for f in node.get("right_required_columns", [])}
            if left & fields and right & fields:
                return False
        elif "join" in op["kind"].lower():
            if (
                set(op.get("left_scope", [])) & relations
                and set(op.get("right_scope", [])) & relations
            ):
                return False
            required = {field_key(f) for f in op.get("required_columns", [])}
            if required and required <= fields:
                return False
    roots = {
        n.node_id: n
        for n in state.partial_plans.nodes
        if n.node_id in state.partial_plans.roots
    }
    components = {nid: nid for nid in roots}

    def representative(nid):
        while components[nid] != nid:
            nid = components[nid]
        return nid

    for pending, op in problem.all_ops.items():
        if pending == oid or pending in state.done or op["node_id"] == pair_for:
            continue
        node = problem.logical[op["node_id"]] if problem.native else op
        left = (
            {field_key(f) for f in node.get("left_required_columns", [])}
            if problem.native
            else set(op.get("left_scope", []))
        )
        right = (
            {field_key(f) for f in node.get("right_required_columns", [])}
            if problem.native
            else set(op.get("right_scope", []))
        )
        if not left or not right:
            continue
        sides = [[], []]
        for nid, root in roots.items():
            available = (
                {field_key(f) for f in root.fields}
                if problem.native
                else root.relations
            )
            if left <= available:
                sides[0].append(nid)
            if right <= available:
                sides[1].append(nid)
        if len(sides[0]) == len(sides[1]) == 1:
            components[representative(sides[1][0])] = representative(sides[0][0])
    return len(inputs) != 2 or representative(inputs[0]) != representative(inputs[1])


def input_choices(problem, state, oid):
    op = problem.all_ops[oid]
    region = problem.tree_nodes[op["node_id"]]
    allowed = set(region["nodes"]) | set(region["base_nodes"])
    roots = {
        n.node_id: n
        for n in state.partial_plans.nodes
        if n.node_id in state.partial_plans.roots and n.node_id in allowed
    }
    node = problem.logical[op["node_id"]]
    arity = len(node["children"])
    built = state.partial_plans.nodes if node.get("candidate_pair_columns") else ()
    pair_graph = {n.node_id: n.children for n in built if n.children}
    pair_fields = {n.node_id: n.fields for n in built}
    choices = list(itertools.permutations(roots, arity))
    if node["type"] == "FILTER" and node.get("pair_targets"):
        choices.extend(itertools.permutations(roots, 2))
    elif node["type"] == "SEMANTIC_JOIN":
        choices.extend(
            (n,) for n, root in roots.items() if root.pending_join == op["node_id"]
        )
    required = {field_key(f) for f in op.get("required_columns", [])}
    done = {problem.all_ops[i]["node_id"] for i in state.done}
    remaining = set(region["nodes"]) - done - {op["node_id"]}
    expected = {
        field_key(f) for f in problem.logical[region["region_id"]]["output_schema"]
    }
    result = []
    for ids in choices:
        inputs = [roots[n] for n in ids]
        pending = [n.pending_join for n in inputs if n.pending_join is not None]
        if pending and not (
            len(ids) == 1
            and (
                node["type"] in ("FILTER", "SEMANTIC_FILTER")
                or pending == [op["node_id"]]
            )
        ):
            continue
        target = pair_target(problem, state, oid, ids) if len(ids) != arity else None
        if len(ids) != arity and target is None and not pending:
            continue
        if (
            len(inputs) == 2
            and inputs[0].relations & inputs[1].relations
            and {field_key(f) for f in inputs[0].fields}
            & {field_key(f) for f in inputs[1].fields}
        ):
            continue
        fields = [n.fields for n in inputs]
        available = {field_key(f) for side in fields for f in side}
        if not required <= available or not roles_valid(node, fields):
            continue
        if len(ids) == 1 and not pair_source_valid(
            problem.logical, node, ids[0], pair_graph, pair_fields
        ):
            continue
        if len(ids) == 2 and not preserves_pending_joins(
            problem,
            state,
            oid,
            [f for side in fields for f in side],
            inputs[0].relations | inputs[1].relations,
            target,
            ids,
        ):
            continue
        output = {
            field_key(f) for f in output_fields(problem.logical, op["node_id"], fields)
        }
        if not remaining and (output != expected or set(roots) != set(ids)):
            continue
        result.append(tuple(ids))
    return result


def execution_order(logical, graph, root):
    order, visiting, visited = [], set(), set()

    def visit(nid):
        if nid in visited:
            return
        if nid in visiting:
            raise ValueError("Cyclic native model tree")
        visiting.add(nid)
        children = graph[nid]
        node = logical[nid]
        if len(children) == 2 and (
            "JOIN" in node["type"] or node["type"] in ("CROSS_PRODUCT", "FILTER")
        ):
            children = list(reversed(children))
        for child in children:
            visit(child)
        visiting.remove(nid)
        visited.add(nid)
        if "semantic_operator_id" in node:
            order.append(node["semantic_operator_id"])

    visit(root)
    return order


def validate_choices(query, spec):
    """Independent validator: each original operator/input is consumed once."""
    if not query.get("tree_regions"):
        raise ValueError("Model tree choices were not advertised by QueryIR")
    logical = {n["node_id"]: n for n in query["operators"]}
    regions = {r["region_id"]: r for r in query.get("tree_regions", [])}
    choices = spec["tree_choices"]
    if not isinstance(choices, list) or len(choices) != len(regions):
        raise ValueError("Missing model tree regions")
    graph = {nid: list(node["children"]) for nid, node in logical.items()}
    actual_fields = {nid: node["output_schema"] for nid, node in logical.items()}
    seen = set()
    replacements = {}
    for choice in choices:
        if (
            not isinstance(choice, dict)
            or set(choice) != {"region_id", "actions", "root"}
            or type(choice["region_id"]) is not int
            or choice["region_id"] not in regions
            or choice["region_id"] in seen
        ):
            raise ValueError("Invalid model tree region contract")
        rid = choice["region_id"]
        seen.add(rid)
        region = regions[rid]
        remaining = set(region["nodes"])
        roots = {nid: logical[nid]["output_schema"] for nid in region["base_nodes"]}
        pending = {}
        if not isinstance(choice["actions"], list) or len(choice["actions"]) != len(
            remaining
        ):
            raise ValueError("Model tree has missing operators")
        for action in choice["actions"]:
            if (
                not isinstance(action, dict)
                or not {"node_id", "inputs"}
                <= set(action)
                <= {"node_id", "inputs", "pair_for"}
                or type(action["node_id"]) is not int
                or action["node_id"] not in remaining
                or not isinstance(action["inputs"], list)
            ):
                raise ValueError("Unknown or duplicate model tree action")
            nid, ids = action["node_id"], action["inputs"]
            node = logical[nid]
            target = action.get("pair_for")
            candidate = (
                node["type"] == "FILTER"
                and len(ids) == 2
                and type(target) is int
                and target in node.get("pair_targets", [])
                and target in remaining
            )
            paired = (
                node["type"] == "SEMANTIC_JOIN"
                and len(ids) == 1
                and pending.get(ids[0]) == nid
            )
            if (
                (len(ids) != len(node["children"]) and not candidate and not paired)
                or any(type(n) is not int for n in ids)
                or len(set(ids)) != len(ids)
                or not set(ids) <= roots.keys()
            ):
                raise ValueError(
                    "Model tree consumes a missing or already consumed input"
                )
            tags = [pending[n] for n in ids if n in pending]
            if (
                "pair_for" in action
                and not candidate
                or tags
                and not (
                    len(ids) == 1
                    and (node["type"] in ("FILTER", "SEMANTIC_FILTER") or paired)
                )
            ):
                raise ValueError("Invalid candidate-pair consumer")
            inputs = [roots[n] for n in ids]
            available = {field_key(f) for side in inputs for f in side}
            if not {
                field_key(f) for f in node.get("required_columns", [])
            } <= available or not roles_valid(node, inputs):
                raise ValueError(
                    "Model tree has invalid field dependencies or join roles"
                )
            if len(ids) == 1 and not pair_source_valid(
                logical, node, ids[0], graph, actual_fields
            ):
                raise ValueError(
                    "Model semantic join changed bound candidate-pair roles"
                )
            if candidate:
                required = {field_key(f) for f in node.get("required_columns", [])}
                if not roles_valid(logical[target], inputs) or any(
                    not required & {field_key(f) for f in side} for side in inputs
                ):
                    raise ValueError(
                        "Candidate predicate must use both bound join sides"
                    )
            output = output_fields(logical, nid, inputs)
            actual_fields[nid] = output
            for child in ids:
                del roots[child]
                pending.pop(child, None)
            roots[nid] = output
            if candidate or tags and node["type"] in ("FILTER", "SEMANTIC_FILTER"):
                pending[nid] = target if candidate else tags[0]
            remaining.remove(nid)
            graph[nid] = ids
        root = choice["root"]
        if type(root) is not int or remaining or pending or set(roots) != {root}:
            raise ValueError("Model tree must have one root and consume all inputs")
        if sorted(field_key(f) for f in roots[root]) != sorted(
            field_key(f) for f in logical[rid]["output_schema"]
        ):
            raise ValueError("Model tree changed boundary fields")
        replacements[rid] = root
    for nid, children in graph.items():
        if not any(nid in r["nodes"] for r in regions.values()):
            graph[nid] = [replacements.get(child, child) for child in children]
    for entry in spec.get("predicate_order", []):
        region = next(
            r
            for r in query["predicate_regions"]
            if r["region_id"] == entry["region_id"]
        )
        root = region["base_node_id"]
        for nid in entry["nodes"]:
            graph[nid] = [root]
            root = nid
        for nid, children in graph.items():
            if nid not in region["nodes"]:
                graph[nid] = [root if c == region["region_id"] else c for c in children]
    original_roots = set(logical) - {c for n in logical.values() for c in n["children"]}
    if len(original_roots) != 1:
        raise ValueError("Native QueryIR requires one SQL root")
    root = replacements.get(next(iter(original_roots)), next(iter(original_roots)))
    order = execution_order(logical, graph, root)
    if order != [a["operator_id"] for a in spec["actions"]]:
        raise ValueError("Model tree disagrees with semantic execution order")
