"""PlanSpec replay and QueryIR serialization helpers."""

from collections.abc import Callable
from typing import Any

from ..model import SearchState


class _ReplayMixin:
    def replay(
        self,
        spec,
        *,
        structural_only=False,
        refresh: Callable[[SearchState], None] | None = None,
    ):
        """Replay a native decision, optionally refreshing each newly built prefix.

        Refresh runs before the next transition inherits this state's evidence.
        """
        if not self.native:
            raise ValueError("Native PlanSpec required for replay")
        from planning.search.protocol import validate_spec

        validate_spec(self.query, spec)
        semantic_order = {a["operator_id"]: i for i, a in enumerate(spec["actions"])}
        decisions = {a["operator_id"]: a for a in spec["actions"]}
        # An order group advertises semantic operators that may be reordered
        # without changing the logical result.  Replay must therefore enforce
        # the recorded order only between groups, not within one group.  The
        # native search can legitimately return (A, C, B) for a group [B, C];
        # if C has no legal input in the current physical state, B must still
        # be allowed to run first.
        semantic_group = {
            operator_id: group_index
            for group_index, group in enumerate(self.groups)
            for operator_id in group
        }
        trees = {
            entry["region_id"]: entry["actions"]
            for entry in spec.get("tree_choices", [])
        }
        orders = {r["region_id"]: r["nodes"] for r in spec.get("predicate_order", [])}
        for region in self.predicate_regions:
            if region["region_id"] in orders:
                continue
            members, start = list(region["nodes"]), 0
            while start < len(members):
                end = start
                while (
                    end < len(members)
                    and self.logical[members[end]]["type"] == "SEMANTIC_FILTER"
                ):
                    end += 1
                members[start:end] = sorted(
                    members[start:end],
                    key=lambda n: semantic_order[
                        self.logical[n]["semantic_operator_id"]
                    ],
                )
                start = max(end, start + 1)
            orders[region["region_id"]] = members
        state, states = self.initial, [self.initial]
        while not self.complete(state):
            choices = []
            next_semantic = next(
                (
                    a["operator_id"]
                    for a in spec["actions"]
                    if a["operator_id"] not in state.done
                ),
                None,
            )
            # Replay only the recorded implementation. Enumerating all layouts
            # would be wasteful and could prevent the original SQL fallback
            # from being reconstructed when max_actions is small.
            available = []
            from planning.state.trees import pair_target

            for oid in sorted(self.all_ops):
                for inputs in self.input_choices(state, oid):
                    action = dict(
                        decisions.get(oid, {"operator_id": oid}), inputs=list(inputs)
                    )
                    target = pair_target(self, state, oid, inputs)
                    if target is not None:
                        action["pair_for"] = target
                    available.append(action)
            for action in available:
                oid, nid = (
                    action["operator_id"],
                    self.all_ops[action["operator_id"]]["node_id"],
                )
                if oid in decisions and (
                    (
                        not trees
                        and semantic_group.get(oid) != semantic_group.get(next_semantic)
                    )
                    or {k: v for k, v in action.items() if k != "inputs"}
                    != decisions[oid]
                ):
                    continue
                if nid in self.tree_nodes and trees:
                    actions = trees[self.tree_nodes[nid]["region_id"]]
                    built = {
                        self.all_ops[a["operator_id"]]["node_id"]
                        for a in state.action_views
                    }
                    expected = next(a for a in actions if a["node_id"] not in built)
                    if (
                        nid != expected["node_id"]
                        or action["inputs"] != expected["inputs"]
                        or action.get("pair_for") != expected.get("pair_for")
                    ):
                        continue
                elif (
                    nid in self.tree_nodes
                    and action["inputs"] != self.logical[nid]["children"]
                ):
                    continue
                region = self.predicate_nodes.get(nid)
                if region:
                    built = {
                        self.all_ops[a["operator_id"]]["node_id"]
                        for a in state.action_views
                    }
                    if nid != next(
                        n for n in orders[region["region_id"]] if n not in built
                    ):
                        continue
                choices.append(action)
            if not choices:
                raise ValueError(
                    "Native decision cannot be replayed in the advertised action space"
                )
            if structural_only:
                native = self.native_problem
                owner = self._native_owner
                if native is None or owner is None:
                    raise ValueError("Structural replay requires the native planner")
                if not native.feasible_action(state.native_handle, choices[0]):
                    raise ValueError(
                        "Replayed action violates hard resource or model constraints"
                    )
                handle = native.transition(state.native_handle, choices[0])
                state = SearchState(
                    tuple(handle.history),
                    {},
                    {},
                    owner.partial_plans(handle),
                    {},
                    handle,
                )
            else:
                state = self.transition(state, choices[0], observe=False)
            if refresh is not None:
                refresh(state)
            states.append(state)
        return states

    def default_complete_state(self, fixed_model_id=None):
        """Construct the default SQL binding plan when legal under active constraints."""
        if not self.native:
            return None
        actions = []
        for group in self.query["order_groups"]:
            for oid in group:
                op = self.ops[oid]
                pool = self.domain.get(oid, ())
                if not pool:
                    raise ValueError(f"No legal action in domain for operator {oid}")
                preferred = fixed_model_id or op.get("default_model_id") or op["models"][0]
                default_layout = op["default_layout"]
                chosen = next(
                    (
                        a
                        for a in pool
                        if a["model_id"] == preferred and a["layout"] == default_layout
                    ),
                    None,
                )
                if chosen is None:
                    chosen = next(
                        (a for a in pool if a["layout"] == default_layout),
                        pool[0],
                    )
                actions.append(chosen)
        spec = {
            "version": 1,
            "kind": "PlanSpec",
            "query_id": self.query["query_id"],
            "actions": actions,
        }
        return self.replay(spec, structural_only=True)[-1]

    def spec(self, state) -> dict[str, Any]:
        if not self.complete(state):
            raise ValueError("A complete plan is required")
        if self.native:
            from planning.search.protocol import validate_spec

            exported_actions = state.actions
            actions = [
                {k: v for k, v in a.items() if k != "inputs"}
                for a in exported_actions
                if a["operator_id"] in self.ops
            ]
            if self.tree_regions:
                from planning.state.trees import execution_order

                order = execution_order(
                    self.logical,
                    dict(state.partial_plans.inputs),
                    state.partial_plans.roots[0],
                )
                chosen = {a["operator_id"]: a for a in actions}
                actions = [chosen[i] for i in order]
            result = {
                "version": 1,
                "kind": "PlanSpec",
                "query_id": self.query["query_id"],
                "actions": actions,
            }
            if self.tree_regions:
                result["tree_choices"] = [
                    {
                        "region_id": r["region_id"],
                        "actions": [
                            dict(
                                node_id=self.all_ops[a["operator_id"]]["node_id"],
                                inputs=a["inputs"],
                                **(
                                    {"pair_for": a["pair_for"]}
                                    if "pair_for" in a
                                    else {}
                                ),
                            )
                            for a in exported_actions
                            if self.all_ops[a["operator_id"]]["node_id"] in r["nodes"]
                        ],
                        "root": next(
                            self.all_ops[a["operator_id"]]["node_id"]
                            for a in reversed(exported_actions)
                            if self.all_ops[a["operator_id"]]["node_id"] in r["nodes"]
                        ),
                    }
                    for r in self.tree_regions
                ]
            if self.predicate_regions:
                result["predicate_order"] = [
                    {
                        "region_id": r["region_id"],
                        "nodes": [
                            self.all_ops[a["operator_id"]]["node_id"]
                            for a in exported_actions
                            if self.all_ops[a["operator_id"]]["node_id"] in r["nodes"]
                        ],
                    }
                    for r in self.predicate_regions
                ]
            validate_spec(self.query, result)
            return result
        return {
            "version": 1,
            "kind": "LogicalPlan",
            "query_id": self.query.get("query_id", self.key),
            "actions": state.actions,
            "root": state.partial_plans.roots[0],
        }
