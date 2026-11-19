"""Input binding choices, legal actions and output field resolution."""

import itertools
import math

from ..identity import (
    field_key,
    semantic_kind,
)
from ..model import ActionBudgetExceeded


class _ActionsMixin:
    def _native_inputs(self, state, oid):
        op = self.all_ops[oid]
        children = dict(self.fixed_inputs)
        if op["node_id"] in self.tree_nodes:
            from planning.state.trees import input_choices

            return input_choices(self, state, oid)
        region = self.predicate_nodes.get(op["node_id"])
        if region:
            if oid in self.ops and not self.tree_regions:
                first = next(
                    (set(g) - state.done for g in self.groups if set(g) - state.done),
                    set(),
                )
                if oid not in first:
                    return []
            chosen = [
                a
                for a in state.action_views
                if self.all_ops[a["operator_id"]]["node_id"] in region["nodes"]
            ]
            root = (
                self.all_ops[chosen[-1]["operator_id"]]["node_id"]
                if chosen
                else region["base_node_id"]
            )
            if root not in state.partial_plans.roots:
                return []
            fields = next(
                n.fields for n in state.partial_plans.nodes if n.node_id == root
            )
            required = {field_key(c) for c in op.get("required_columns", [])}
            return [(root,)] if required <= {field_key(c) for c in fields} else []
        inputs = children[op["node_id"]]
        if oid in self.ops and not self.tree_regions:
            for group in self.groups:
                remaining = set(group) - state.done
                if remaining:
                    if oid not in remaining:
                        return []
                    ids = {self.ops[i]["node_id"] for i in group}
                    bottom = next(n for n in ids if not set(children[n]) & ids)
                    chosen = [
                        a for a in state.action_views if a["operator_id"] in group
                    ]
                    inputs = (
                        (self.ops[chosen[-1]["operator_id"]]["node_id"],)
                        if chosen
                        else children[bottom]
                    )
                    break
        for region in self.tree_regions:
            chosen = [
                a
                for a in state.action_views
                if self.all_ops[a["operator_id"]]["node_id"] in region["nodes"]
            ]
            if region["region_id"] in inputs:
                if len(chosen) != len(region["nodes"]):
                    return []
                inputs = tuple(
                    (
                        self.all_ops[chosen[-1]["operator_id"]]["node_id"]
                        if n == region["region_id"]
                        else n
                    )
                    for n in inputs
                )
        for region in self.predicate_regions:
            chosen = [
                a
                for a in state.action_views
                if self.all_ops[a["operator_id"]]["node_id"] in region["nodes"]
            ]
            if region["region_id"] in inputs and len(chosen) != len(region["nodes"]):
                return []
            if len(chosen) == len(region["nodes"]):
                top = self.all_ops[chosen[-1]["operator_id"]]["node_id"]
                inputs = tuple(top if c == region["region_id"] else c for c in inputs)
        # Fixed and semantic parents both reference the original group top;
        # after reordering they must consume the actual final node instead.
        for group in self.groups:
            if any(self.ops[i]["node_id"] in self.predicate_nodes for i in group):
                continue
            ids = {self.ops[i]["node_id"] for i in group}
            tops = ids - {c for n in ids for c in children[n]}
            if oid not in group and set(inputs) & tops and not set(group) <= state.done:
                return []
            selected = [a for a in state.action_views if a["operator_id"] in group]
            if set(group) <= state.done and selected:
                inputs = tuple(
                    self.ops[selected[-1]["operator_id"]]["node_id"] if c in tops else c
                    for c in inputs
                )
        return [inputs] if set(inputs) <= set(state.partial_plans.roots) else []

    def input_choices(self, state, oid):
        native = self.native_problem
        if native is not None:
            return [
                tuple(ids) for ids in native.input_choices(state._native_handle, oid)
            ]
        op = self.all_ops[oid]
        if oid in state.done or not set(op.get("depends_on", [])) <= state.done:
            return []
        if self.native:
            return self._native_inputs(state, oid)
        roots = {
            n.node_id: n
            for n in state.partial_plans.nodes
            if n.node_id in state.partial_plans.roots
        }
        is_join = "join" in op["kind"].lower()
        choices: list[tuple[int, ...]] = (
            list(itertools.permutations(roots, 2)) if is_join else [(r,) for r in roots]
        )
        if op["kind"].lower() == "filter" and op.get("pair_targets"):
            choices.extend(itertools.permutations(roots, 2))
        elif is_join and semantic_kind(op):
            choices.extend(
                (r,) for r, node in roots.items() if node.pending_join == op["node_id"]
            )
        result = []
        required = {field_key(c) for c in op.get("required_columns", [])}
        for inputs in choices:
            nodes = [roots[i] for i in inputs]
            pending = [n.pending_join for n in nodes if n.pending_join is not None]
            if pending and not (
                len(inputs) == 1
                and (
                    op["kind"].lower() in ("filter", "semantic_filter")
                    or pending == [op["node_id"]]
                )
            ):
                continue
            from planning.state.trees import pair_target

            target = (
                pair_target(self, state, oid, inputs)
                if len(inputs) == 2 and not is_join
                else None
            )
            if len(inputs) == 2 and not is_join and target is None:
                continue
            available = {field_key(c) for n in nodes for c in n.fields}
            scope = frozenset().union(*(n.relations for n in nodes))
            if not required <= available or not set(op.get("scope", [])) <= scope:
                continue
            if len(inputs) == 2:
                if nodes[0].relations & nodes[1].relations:
                    continue
                if (
                    not set(op.get("left_scope", [])) <= nodes[0].relations
                    or not set(op.get("right_scope", [])) <= nodes[1].relations
                ):
                    continue
                # A join predicate must connect the two inputs; no invented cross join.
                if required and any(
                    not required & {field_key(c) for c in n.fields} for n in nodes
                ):
                    continue
                from planning.state.trees import preserves_pending_joins

                if not preserves_pending_joins(
                    self,
                    state,
                    oid,
                    [f for n in nodes for f in n.fields],
                    scope,
                    target,
                    inputs,
                ):
                    continue
            if "projection" in op["kind"].lower():
                output = {field_key(c) for c in self._output_fields(op, nodes)}
                outside = {
                    field_key(c)
                    for n in roots.values()
                    if n.node_id not in inputs
                    for c in n.fields
                }
                produced = {
                    field_key(c)
                    for i, other in self.all_ops.items()
                    if i not in state.done and i != oid
                    for c in (
                        other.get("produced_fields", [])
                        if semantic_kind(other)
                        else (
                            other.get("output_schema", [])
                            if "projection" in other.get("kind", "").lower()
                            else []
                        )
                    )
                }
                needed = {
                    field_key(c)
                    for i, other in self.all_ops.items()
                    if i not in state.done and i != oid
                    for c in other.get("required_columns", [])
                }
                needed |= {field_key(c) for c in self.query.get("output_schema", [])}
                if not needed <= output | outside | produced:
                    continue
            result.append(tuple(inputs))
        return result

    def actions(
        self,
        state,
        *,
        prioritize_relational=False,
        fixed_layout=False,
        highest_priorities=None,
        fixed_model_id=None,
    ):
        """Enumerate the first feasible priority group before applying its budget.

        None expands all groups; an empty mapping assigns every operator the
        shared fallback priority. Equal-priority actions remain beam candidates.
        """
        native = self.native_problem
        if native is not None:
            from ...kernels.state import core

            try:
                return native.actions(
                    state._native_handle,
                    prioritize_relational,
                    fixed_layout,
                    highest_priorities,
                    fixed_model_id,
                )
            except core().ActionBudgetExceeded as exc:
                raise ActionBudgetExceeded(str(exc)) from exc
        result = []
        operators = sorted(self.all_ops)
        if highest_priorities is not None:
            if any(
                len(priority) != 3 or any(not math.isfinite(v) for v in priority)
                for priority in highest_priorities.values()
            ):
                raise ValueError("Action priorities must contain three finite values")

            def priority(oid):
                return tuple(highest_priorities.get(oid, (5.0, 1.0, 0.0)))

        from planning.search.heuristics import fixed_layout as heuristic_layout
        from planning.state.trees import canonical_join_inputs, pair_target

        node_operators = {op["node_id"]: oid for oid, op in self.all_ops.items()}
        branches = []
        for oid in operators:
            op = self.all_ops[oid]
            own_priority = priority(oid) if highest_priorities is not None else None
            partitioned = (
                highest_priorities is not None
                and op.get("kind", op.get("type", "")).lower() == "filter"
                and bool(op.get("pair_targets"))
            )
            branches.append((oid, partitioned, None, own_priority))
            if partitioned:
                for target in dict.fromkeys(op["pair_targets"]):
                    if target not in node_operators:
                        raise ValueError(
                            "Candidate-pair target has no semantic operator"
                        )
                    # Binary candidate filtering is still a relational FILTER.
                    # Input legality, not the target join's tier, governs when
                    # it can run; retain every feasible branch in this tier.
                    branches.append(
                        (
                            oid,
                            True,
                            target,
                            own_priority,
                        )
                    )
        if highest_priorities is not None:
            branches.sort(key=lambda branch: branch[3])
        input_cache, domain_cache = {}, {}
        for semantic in (False, True) if prioritize_relational else (None,):
            best = None
            for i, partitioned, branch_target, branch_priority in branches:
                if semantic is not None and (i in self.ops) != semantic:
                    continue
                if best is not None and branch_priority != best:
                    break
                if i not in input_cache:
                    input_cache[i] = [
                        (inputs, pair_target(self, state, i, inputs))
                        for inputs in canonical_join_inputs(
                            self, state, i, self.input_choices(state, i)
                        )
                    ]
                choices = [
                    (inputs, target)
                    for inputs, target in input_cache[i]
                    if not partitioned or target == branch_target
                ]
                if not choices:
                    continue

                if i not in domain_cache:
                    layout = (
                        heuristic_layout(self.ops[i])
                        if fixed_layout and i in self.ops
                        else None
                    )
                    domain_cache[i] = [
                        a
                        for a in self.domain[i]
                        if (layout is None or a.get("layout") == layout)
                        and (
                            i not in self.ops
                            or fixed_model_id is None
                            or a["model_id"] == fixed_model_id
                        )
                        and self.feasible_action(state, a)
                    ]
                feasible = domain_cache[i]
                if not feasible:
                    continue
                if highest_priorities is not None:
                    best = branch_priority
                if len(result) + len(choices) * len(feasible) > self.max_actions:
                    raise ActionBudgetExceeded("Ready action set exceeds max_actions")
                for inputs, target in choices:
                    extra = {}
                    if target is not None:
                        extra["pair_for"] = target
                    result.extend(
                        dict(a, inputs=list(inputs), **extra) for a in feasible
                    )
            if result:
                return result
        return result

    def _output_fields(self, op, inputs):
        if self.native:
            if op["node_id"] in self.tree_nodes:
                from planning.state.trees import output_fields

                return output_fields(
                    self.logical, op["node_id"], [n.fields for n in inputs]
                )
            if op["node_id"] in self.predicate_nodes:
                fields = [c for n in inputs for c in n.fields]
                original = self.logical[op["node_id"]]
                old_input = {
                    field_key(c)
                    for c in self.logical[original["children"][0]].get(
                        "output_schema", []
                    )
                }
                return fields + [
                    c
                    for c in original.get("output_schema", [])
                    if field_key(c) not in old_input
                ]
            return self.logical[op["node_id"]].get("output_schema", [])
        if "projection" in op["kind"].lower() and not semantic_kind(op):
            return op["output_schema"]
        fields = [c for n in inputs for c in n.fields]
        return fields + op.get("produced_fields", [])
