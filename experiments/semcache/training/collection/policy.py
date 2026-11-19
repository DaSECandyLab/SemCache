"""Explicit round planners and reproducible query order for real feedback."""

import copy
import math
import random

from planning.search.pareto import pareto_fronts
from planning.state import digest

PLAN_GENERATION_POLICIES = {
    "heuristic_requests": "highest_legal_group_fixed_model_gotdf_calibrated_time_v2",
    "heuristic_value": "highest_legal_group_all_models_layouts_value_v1",
    "value": "all_legal_operators_models_layouts_value_v1",
}
FEEDBACK_PROTOCOL = "two_search_two_execution_v1"
ACCURACY_FEEDBACK_PROTOCOL = "three_accuracy_searches_v1"
DUPLICATE_PLAN_POLICY = "strict_distinct_operator_trees_and_join_positions_v5"


class DistinctStructureUnavailable(ValueError):
    """The final beam has no unseen logical structure or join position left."""

FEEDBACK_COLLECTION_ORDER = (
    "estimated bootstrap unchanged; measured queries run two independent Beam searches and executions: "
    "frontier random first, then highest predicted quality or fastest when the first was already highest quality; "
    "fit after the whole round"
)
ROLLOUT_POLICIES = ("pareto_random", "highest_quality", "fastest", "minimum_accuracy")


def plan_signature(spec):
    """Identify a plan's structure and every model/layout, without query identity."""
    normalized = copy.deepcopy(
        {key: value for key, value in spec.items() if key != "query_id"}
    )
    if "tree_choices" in normalized or "predicate_order" in normalized:
        normalized["actions"] = sorted(
            normalized.get("actions", []), key=lambda action: action["operator_id"]
        )
    if "tree_choices" in normalized:
        normalized["tree_choices"].sort(key=lambda region: region["region_id"])
        for region in normalized["tree_choices"]:
            region["actions"].sort(key=lambda action: action["node_id"])
    if "predicate_order" in normalized:
        normalized["predicate_order"].sort(key=lambda region: region["region_id"])
    return digest(normalized)


def logical_structure_signature(spec, query):
    """Compare logical operator trees within a query, ignoring models and layouts."""
    nodes = {n["node_id"]: n for n in query["operators"]}
    if spec.get("tree_choices"):
        trees = []
        for region in spec["tree_choices"]:
            actions = {a["node_id"]: a for a in region["actions"]}

            def tree(nid, visiting=frozenset(), region_actions=actions):
                if nid in visiting:
                    raise ValueError("Cyclic plan structure")
                kind = nodes[nid]["type"]
                action = region_actions.get(nid)
                if action is None:
                    return [kind, nid]
                children = [
                    tree(child, visiting | {nid}, region_actions)
                    for child in action.get("inputs", [])
                ]
                if nodes[nid].get("join_commutative"):
                    children.sort(key=digest)
                return [kind, nid, children]

            trees.append(tree(region["root"]))
        trees.sort(key=digest)
        return digest({"operator_trees": trees})
    kinds = {
        op["operator_id"]: op["kind"] for op in query.get("semantic_operators", [])
    }
    for node in nodes.values():
        kinds.setdefault(-node["node_id"] - 1, node["type"])
    if spec.get("predicate_order"):
        orders = [
            [(kinds[oid], oid) for oid in region["operator_ids"]]
            for region in spec["predicate_order"]
        ]
        return digest({"operator_orders": sorted(orders)})
    return digest(
        {
            "operator_ids": [
                (kinds[a["operator_id"]], a["operator_id"])
                for a in spec.get("actions", [])
            ]
        }
    )


def join_position_signature(spec, query):
    """Identify each join operator's position relative to semantic subtrees."""
    nodes = {n["node_id"]: n for n in query["operators"]}
    if not spec.get("tree_choices") or not any(
        "JOIN" in node["type"] for node in nodes.values()
    ):
        return None
    per_region = []
    for region in spec["tree_choices"]:
        actions = {a["node_id"]: a for a in region["actions"]}

        def sem_set(nid, visiting=frozenset(), region_actions=actions):
            if nid in visiting:
                raise ValueError("Cyclic plan structure")
            kind = nodes[nid]["type"]
            current = [(kind, nid)] if kind.startswith("SEMANTIC_") else []
            action = region_actions.get(nid)
            if action is None:
                return tuple(current)
            nested = [
                item
                for child in action.get("inputs", [])
                for item in sem_set(child, visiting | {nid}, region_actions)
            ]
            return tuple(sorted(set(current + nested)))

        def collect_joins(nid, visiting=frozenset(), region_actions=actions):
            if nid in visiting:
                raise ValueError("Cyclic plan structure")
            action = region_actions.get(nid)
            if action is None:
                return []
            positions = []
            kind = nodes[nid]["type"]
            if "JOIN" in kind:
                child_sems = [
                    list(sem_set(child, visiting | {nid}, region_actions))
                    for child in action.get("inputs", [])
                ]
                if nodes[nid].get("join_commutative"):
                    child_sems.sort()
                positions.append((kind, nid, child_sems))
            for child in action.get("inputs", []):
                positions.extend(collect_joins(child, visiting | {nid}, region_actions))
            return positions

        region_joins = sorted(collect_joins(region["root"]), key=digest)
        if region_joins:
            per_region.append(region_joins)
    if not per_region:
        return None
    per_region.sort(key=digest)
    return digest({"join_positions": per_region})


def validate_minimum_accuracy(minimum):
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, (int, float))
        or not math.isfinite(minimum)
        or not 0 <= minimum <= 1
    ):
        raise ValueError("minimum_accuracy must be a finite number in [0,1]")


def select_accuracy_rollout(
    scores,
    accuracies,
    times,
    plan_specs,
    seed,
    minimum_accuracy,
    excluded_plan_signatures=(),
    excluded_structure_signatures=(),
    excluded_join_positions=(),
    *,
    query,
):
    """Fastest feasible completion, with honest quality fallback and deduplication.

    Each accuracy target must select a new logical structure and, when joins
    are present, a new join-operator position. Among unseen candidates prefer
    feasible completions, then sample uniformly. Exhaustion requests another
    search; it never falls back to layout-only changes.
    """
    validate_minimum_accuracy(minimum_accuracy)
    if not scores or any(
        len(values) != len(scores) for values in (accuracies, times, plan_specs)
    ):
        raise ValueError("Accuracy rollout needs aligned, nonempty predictions")
    if any(
        value is not None and (not math.isfinite(value) or not 0 <= value <= 1)
        for value in accuracies
    ) or any(value is not None and not math.isfinite(value) for value in times):
        raise ValueError("Invalid completion accuracy or time prediction")
    signatures = [plan_signature(spec) for spec in plan_specs]
    structures = [logical_structure_signature(spec, query) for spec in plan_specs]
    join_positions = [join_position_signature(spec, query) for spec in plan_specs]
    # A search may export several construction histories for the same plan.
    distinct = {}
    for index, signature in enumerate(signatures):
        distinct.setdefault(signature, []).append(index)
    excluded = set(excluded_plan_signatures)
    excluded_structures = set(excluded_structure_signatures)
    excluded_joins = set(excluded_join_positions) - {None}
    for signature, structure, join_position in zip(
        signatures, structures, join_positions
    ):
        if signature in excluded:
            excluded_structures.add(structure)
            if join_position is not None:
                excluded_joins.add(join_position)

    def feasible(pool):
        return [
            i
            for i in pool
            if accuracies[i] is not None and accuracies[i] >= minimum_accuracy
        ]

    def highest_quality(pool):
        best = max(accuracies[i] if accuracies[i] is not None else -1.0 for i in pool)
        return [
            i
            for i in pool
            if (accuracies[i] if accuracies[i] is not None else -1.0) == best
        ]

    def fastest(pool):
        return min(
            pool,
            key=lambda i: (
                times[i] is None,
                times[i] if times[i] is not None else 0.0,
                -scores[i][0],
                -scores[i][1],
                i,
            ),
        )

    indices = [
        fastest(feasible(group) or highest_quality(group))
        for group in distinct.values()
    ]
    meeting = feasible(indices)
    preferred = fastest(meeting or highest_quality(indices))
    chosen = preferred
    replacement_pool = []
    replacement_scope = None
    unseen_structure_count = 0
    join_repeated = (
        join_positions[preferred] is not None
        and join_positions[preferred] in excluded_joins
    )
    duplicate = (
        signatures[preferred] in excluded
        or structures[preferred] in excluded_structures
        or join_repeated
    )
    if duplicate:
        unseen = [
            i
            for i in indices
            if structures[i] not in excluded_structures
            and (join_positions[i] is None or join_positions[i] not in excluded_joins)
        ]
        if not unseen:
            raise DistinctStructureUnavailable(
                "No unseen logical structure and distinct join position in the "
                "final beam; a different model or layout does not satisfy "
                "structural diversity"
            )
        eligible = feasible(unseen) or unseen
        groups = {}
        for i in eligible:
            groups.setdefault((structures[i], join_positions[i]), []).append(i)
        unseen_structure_count = len(groups)
        rng = random.Random(seed)
        group = rng.choice(sorted(groups, key=lambda pair: (pair[0], pair[1] or "")))
        replacement_pool = eligible
        chosen = rng.choice(groups[group])
        replacement_scope = (
            "logical_structure_and_join_position"
            if join_positions[chosen] is not None
            else "logical_structure"
        )
    satisfied = (
        accuracies[chosen] is not None and accuracies[chosen] >= minimum_accuracy
    )
    return chosen, {
        "policy": "minimum_accuracy",
        "seed": seed,
        "minimum_accuracy": minimum_accuracy,
        "constraint_satisfied": satisfied,
        "accuracy_fallback": not satisfied,
        "selection_basis": (
            "random_unseen_structure_meeting_minimum_accuracy"
            if satisfied and replacement_pool
            else "random_unseen_structure_accuracy_fallback"
            if replacement_pool
            else "fastest_meeting_minimum_accuracy"
            if satisfied
            else "highest_predicted_accuracy_fallback"
        ),
        "candidate_count": len(scores),
        "distinct_candidate_count": len(indices),
        "eligible_candidate_count": len(meeting),
        "selected_index": chosen,
        "preferred_index": preferred,
        "selected_is_highest_quality": chosen in highest_quality(indices),
        "predicted_scores": list(scores[chosen]),
        "predicted_accuracy": accuracies[chosen],
        "predicted_seconds": times[chosen],
        "plan_signature": signatures[chosen],
        "preferred_plan_signature": signatures[preferred],
        "logical_structure_signature": structures[chosen],
        "preferred_logical_structure_signature": structures[preferred],
        "join_position_signature": join_positions[chosen],
        "preferred_join_position_signature": join_positions[preferred],
        "duplicate_detected": duplicate,
        "duplicate_replaced": chosen != preferred,
        "repeat_unavoidable": signatures[chosen] in excluded,
        "replacement_candidate_count": len(replacement_pool),
        "replacement_scope": replacement_scope,
        "unseen_structure_candidate_count": unseen_structure_count,
        "structure_repeated": structures[chosen] in excluded_structures,
        "join_position_repeated": (
            join_positions[chosen] is not None
            and join_positions[chosen] in excluded_joins
        ),
        "structure_repeat_unavoidable": duplicate and structures[chosen] in excluded_structures,
        "replacement_policy": DUPLICATE_PLAN_POLICY
        if replacement_pool
        else None,
    }


def select_rollout(scores, seed, policy):
    """Select from this search's final frontier; both scores are larger-is-better."""
    if policy not in ROLLOUT_POLICIES or policy == "minimum_accuracy" or not scores:
        raise ValueError("Invalid feedback selection policy or empty candidates")
    front = pareto_fronts(scores)[0]
    if policy == "pareto_random":
        chosen = random.Random(seed).choice(front)
    else:
        primary, secondary = (1, 0) if policy == "highest_quality" else (0, 1)
        chosen = max(
            front, key=lambda i: (scores[i][primary], scores[i][secondary], -i)
        )
    return chosen, {
        "policy": policy,
        "seed": seed,
        "frontier_size": len(front),
        "selected_index": chosen,
        "selected_is_highest_quality": scores[chosen][1]
        == max(row[1] for row in scores),
        "predicted_scores": list(scores[chosen]),
    }


def ordered_workloads(workloads, order, seed):
    ordered = list(workloads)
    if order == "reverse":
        ordered.reverse()
    elif order == "shuffled":
        random.Random(seed).shuffle(ordered)
    elif order != "forward":
        raise ValueError("Query order must be forward, reverse or shuffled")
    return ordered
