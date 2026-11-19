"""SearchProblem state transitions, observations and cost estimation."""

import copy
import json
from typing import Any

from planning.state.partial_plans import PartialPlanNode, PartialPlanSet, descendants

from ..copying import copy_statistics
from ..identity import (
    action_key,
    canonical,
    field_key,
    semantic_kind,
)
from ..model import SearchState


class _TransitionsMixin:
    def transition(self, state, action, *, observe=True):
        native = self.native_problem
        if native is not None:
            if not isinstance(action, dict):
                raise ValueError("Illegal state transition")
            handle = native.transition(state._native_handle, action)
            return self.observe_transition(
                state, json.loads(handle.action_json), handle, observe=observe
            )
        if not isinstance(action, dict) or type(action.get("operator_id")) is not int:
            raise ValueError("Illegal state transition")
        oid = action["operator_id"]
        if oid not in self.all_ops:
            raise ValueError("Illegal state transition")
        base = {k: v for k, v in action.items() if k not in ("inputs", "pair_for")}
        choices = self.input_choices(state, oid)
        inputs = action.get("inputs")
        if inputs is None and len(choices) == 1:
            inputs = list(choices[0])
        if inputs is None:
            raise ValueError(
                "Illegal state transition: ambiguous or unavailable inputs"
            )
        if (
            canonical(base) not in self.domain_keys[oid]
            or tuple(inputs or ()) not in choices
            or not self.feasible_action(state, base)
        ):
            raise ValueError("Illegal state transition")
        from planning.state.trees import pair_target

        target = pair_target(self, state, oid, inputs)
        if action.get("pair_for") != target:
            raise ValueError("Illegal candidate-pair target")
        action = dict(base, inputs=inputs)
        if target is not None:
            action["pair_for"] = target
        nodes = {n.node_id: n for n in state.partial_plans.nodes}
        children = [nodes[i] for i in inputs]
        op = self.all_ops[oid]
        preserves_pairs = op.get("type", op.get("kind", "")).upper() in (
            "FILTER",
            "SEMANTIC_FILTER",
        )
        node = PartialPlanNode(
            op["node_id"],
            tuple(inputs),
            1 + max(n.height for n in children),
            oid,
            tuple(self._output_fields(op, children)),
            frozenset().union(*(n.relations for n in children)),
            (
                target
                if target is not None
                else (
                    children[0].pending_join
                    if len(children) == 1 and preserves_pairs
                    else None
                )
            ),
            (
                tuple(
                    tuple(
                        f.get("name", field_key(f)) if isinstance(f, dict) else f
                        for f in n.fields
                    )
                    for n in children
                )
                if target is not None
                else (
                    children[0].pair_columns
                    if len(children) == 1 and preserves_pairs
                    else ()
                )
            ),
        )
        roots = tuple(
            sorted(
                (
                    set(state.partial_plans.roots)
                    - (set() if op.get("type") == "CTE_SCAN" else set(inputs))
                )
                | {node.node_id}
            )
        )
        partial_plans = PartialPlanSet(
            state.partial_plans.nodes + (node,),
            roots,
            tuple(n for n in state.partial_plans.remaining if n != node.node_id),
            state.partial_plans.inputs + ((node.node_id, tuple(inputs)),),
        )
        history = state.history + (canonical(action),)
        stats = self._transition_statistics(
            state, action, partial_plans, history, observe
        )
        child = SearchState(
            history,
            stats,
            copy.copy(state.sample),
            partial_plans,
            dict(state.sample_observations),
        )
        return self.observe(child) if observe and self.sampler else child

    def observe_transition(self, state, action, native_child, *, observe=True):
        """Annotate a native structural child without repeating its transition.

        The native search session calls this synchronous boundary exactly once
        per expansion; mutable evidence remains independent in every branch.
        """
        owner = self._native_owner
        if owner is None:
            raise ValueError("Native transition requires the native planning backend")
        partial_plans = owner.partial_plans(native_child)
        history = state.history + (native_child.action_json,)
        stats = self._transition_statistics(
            state, action, partial_plans, history, observe
        )
        from planning.observations.evidence import freeze_evidence

        child = SearchState(
            history,
            stats,
            copy.copy(state.sample),
            partial_plans,
            dict(state.sample_observations),
            native_child,
            state.action_views + (freeze_evidence(action),),
        )
        return self.observe(child) if observe and self.sampler else child

    def _transition_statistics(self, state, action, partial_plans, history, observe):
        stats = copy_statistics(state.statistics)
        stats[action_key(action)] = self.estimate(state, action, partial_plans)
        return stats

    def observe(self, child):
        return (
            self.sampler.selectivity(self, child) if self.sampler is not None else child
        )

    def execution_operator(self, state, oid) -> dict[str, Any]:
        """Carry the chosen input partition to execution without changing the AST."""
        op = copy.deepcopy(self.all_ops[oid])
        nodes = state._nodes_by_id
        node = state._nodes_by_operator[oid]
        op["input_columns"] = [
            [
                f.get("name", field_key(f)) if isinstance(f, dict) else f
                for f in nodes[c].fields
            ]
            for c in node.children
        ]
        if (
            semantic_kind(op)
            and "join" in op["kind"].lower()
            and len(node.children) == 1
        ):
            partition = nodes[node.children[0]].pair_columns
            if partition:
                op["pair_columns"] = partition
        return op

    def update_accuracy(self, state):
        return self.sampler.accuracy(self, state) if self.sampler else state

    def semantic_actions(self, state):
        return [a for a in state.action_views if a["operator_id"] in self.ops]

    def cache_sources(self, state, action, partial_plans=None):
        partial_plans = (
            partial_plans or self.transition(state, action, observe=False).partial_plans
        )
        current = self.all_ops[action["operator_id"]]["node_id"]
        paths = (
            state._descendants
            if partial_plans is state.partial_plans
            else descendants(partial_plans.inputs)
        )
        return {i for i, op in self.ops.items() if op["node_id"] in paths[current]}

    def estimate(
        self, state, action, partial_plans=None, *, copy_observation=True
    ) -> dict[str, Any]:
        if action["operator_id"] in self.ops:
            from planning.observations.statistics import estimate as estimate_cache
            from planning.observations.statistics import input_statistics

            result = input_statistics(
                self,
                state,
                action,
                copy_observation=copy_observation,
            )
            from planning.observations.accuracy import value as accuracy_value

            quality = accuracy_value(
                self.profile,
                action["model_id"],
                self.ops[action["operator_id"]]["kind"],
                action["layout"],
            )
            if self._number(quality):
                result["accuracy"] = quality
            else:
                result.pop("accuracy", None)
        else:
            op = self.all_ops[action["operator_id"]]
            node_evidence = self.profile.get("node_statistics", {}).get(
                str(op["node_id"]), {}
            )
            result = copy_statistics({"node": node_evidence})["node"]
        op = self.all_ops[action["operator_id"]]
        independent = self.profile.get("sampling_strategy") == "independent_base"
        if independent:
            operator_evidence = self.profile.get("operator_statistics", {}).get(
                str(action["operator_id"]), {}
            )
            if action["operator_id"] not in self.ops:
                result.update(
                    copy_statistics({"operator": operator_evidence})["operator"]
                )
            if "selectivity" in operator_evidence:
                result["selectivity"] = operator_evidence["selectivity"]
                for name in (
                    "estimated_selectivity",
                    "selectivity_fusion",
                    "survival_estimated",
                ):
                    result.pop(name, None)
        if self._number(op.get("duckdb_selectivity")):
            result.setdefault("duckdb_selectivity", op["duckdb_selectivity"])
        if "inputs" in action:
            inputs = action["inputs"]
        else:
            choices = self.input_choices(state, action["operator_id"])
            inputs = list(choices[0]) if len(choices) == 1 else []
        by_node = {
            self.all_ops[old_oid]["node_id"]: state.statistics.get(old_key, {})
            for old_oid, old_key in state._action_keys_by_operator.items()
        }
        child_stats = [
            by_node.get(
                nid,
                {
                    **self.profile.get("node_statistics", {}).get(str(nid), {}),
                    **state.statistics.get(f"scan:{nid}", {}),
                },
            )
            for nid in inputs
        ]
        sizes = [
            stat.get("output_rows", self.logical[nid].get("estimated_rows"))
            for nid, stat in zip(inputs, child_stats)
        ]
        result.pop("conditional_observation", None)
        if sizes and all(self._number(v) and v >= 0 for v in sizes):
            from planning.cost.estimates import propagate, row_width

            observed_selectivity = result.get("selectivity")
            result = propagate(op, result, sizes)
            if result.get("survival_estimated"):
                result["selectivity"] = observed_selectivity
            widths = [info.get("output_width") for info in child_stats]
            if all(self._number(w) for w in widths):
                result["input_width"] = sum(widths)
                result["input_side_widths"] = widths
            field_stats = {}
            for info in child_stats:
                field_stats.update(info.get("output_field_statistics", {}))
            result.setdefault("input_field_statistics", field_stats)
            if "projection" not in op.get("kind", op.get("type", "")).lower():
                result.setdefault("output_field_statistics", field_stats)
            if partial_plans is not None:
                node = next(
                    n
                    for n in partial_plans.nodes
                    if n.operator_id == action["operator_id"]
                )
                width = row_width(
                    node.fields, result.get("output_field_statistics", {})
                )
                if width is not None:
                    result["output_width"] = width
        elif action["operator_id"] not in self.ops:
            for key in (
                "input_rows",
                "output_rows",
                "relational_cost",
                "relational_seconds",
            ):
                result.pop(key, None)
        if independent:
            from dataclasses import replace
            from types import SimpleNamespace

            from planning.sampling.evidence import (
                generated_field_identities,
                generated_survivals,
            )
            from planning.sampling.survival import (
                propagate_depth_statistics,
                restore_cache_fields,
            )

            structure = (
                replace(state, partial_plans=partial_plans)
                if partial_plans is not None
                else state
            )
            node = structure._nodes_by_operator.get(action["operator_id"])
            result["generated_field_identities"] = generated_field_identities(op)
            if node is not None:
                observations = {
                    current.node_id: SimpleNamespace(
                        statistics=by_node.get(
                            current.node_id,
                            state.statistics.get(f"scan:{current.node_id}", {}),
                        )
                    )
                    for current in structure.partial_plans.nodes
                    if current.node_id != node.node_id
                }
                result["generated_input_survival"] = generated_survivals(
                    structure,
                    observations,
                    node,
                    op,
                    result.get(
                        "prefix_selectivity_statistics",
                        result.get("depth_selectivity_statistics", {}),
                    ),
                )
            incoming = propagate_depth_statistics(result, child_stats)
            if action["operator_id"] in self.ops:
                restore_cache_fields(self, action, result, incoming)
        if action["operator_id"] in self.ops:
            return estimate_cache(
                self, state, action, statistics=result, partial_plans=partial_plans
            )
        return result
