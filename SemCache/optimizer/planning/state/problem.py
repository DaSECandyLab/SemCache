"""Legal mixed-plan actions and QueryIR/LogicalQuery adaptation."""

import copy
import itertools
import json
import math
import re
from collections.abc import Callable
from typing import Any, TypeGuard

from planning.prompt_layouts import BLOCK_ORDERS
from planning.state.partial_plans import PartialPlanNode, PartialPlanSet, descendants

from .copying import copy_profile, copy_statistics
from .identity import (
    action_key,
    canonical,
    field_key,
    first_field_orders,
    query_key,
    semantic_kind,
)
from .model import ActionBudgetExceeded, SearchState


class SearchProblem:
    """LogicalQuery exposes fields/scopes/dependencies; QueryIR keeps native legality.

    LogicalQuery operators contain operator_id, node_id, kind, required_columns,
    output_schema (projection), scope, and depends_on. Joins additionally declare
    left_scope/right_scope. No relational physical algorithm is searched.
    """

    def __init__(
        self, query, profile=None, sampler=None, max_actions=8192, *, backend="native"
    ):
        if backend not in ("native", "python"):
            raise ValueError("Unknown planning backend")
        self.backend = backend
        self._native_owner = None
        if query.get("version") != 1 or query.get("kind") not in (
            "QueryIR",
            "LogicalQuery",
        ):
            raise ValueError("Unsupported query representation")
        if type(max_actions) is not int or max_actions < 1:
            raise ValueError("max_actions must be positive")
        self.query, self.profile = copy.deepcopy(query), copy_profile(profile or {})
        # Only independent observations seed search; conditional survivor rates describe a different population.
        self.profile.pop("conditional_inputs", None)
        self.profile.pop("prefix_statistics", None)
        if self.profile.get("sampling_strategy") != "independent_base":
            self.profile["statistics"] = {
                key: row
                for key, row in self.profile.get("statistics", {}).items()
                if key.startswith("scan:")
                or row.get("sampling_strategy") == "independent_base"
            }
            self.profile.pop("operator_statistics", None)
        # Sampling annotates logical/semantic operator dictionaries. Keep those
        # working copies separate from both the caller's bound IR and self.query;
        # query identity must remain stable when freezing or reoptimizing.
        query = copy.deepcopy(self.query)
        if query["kind"] == "QueryIR":
            from .cte import connect_references

            connect_references(query)
        self.key, self.sampler, self.max_actions = (
            query_key(query),
            sampler,
            max_actions,
        )
        self.native = query["kind"] == "QueryIR"
        self.history_provider: Callable[[], list[dict[str, Any]]] | None = None
        if self.native:
            for relation in query.get("relations", []):
                statistics = self.profile.setdefault("node_statistics", {}).setdefault(
                    str(relation["node_id"]), {}
                )
                if self._number(relation.get("estimated_rows")):
                    statistics.setdefault("output_rows", relation["estimated_rows"])
                columns = statistics.setdefault("output_field_statistics", {})
                for column in relation.get("columns", []):
                    if "binding" not in column:
                        continue
                    info = columns.setdefault(field_key(column), {})
                    for name in ("distinct_count", "null_fraction"):
                        if self._number(column.get(name)):
                            info.setdefault(name, column[name])
        if query.get("global_system_prefix"):
            self.profile.setdefault("system_prefix", query["global_system_prefix"])
        self.profile.setdefault("operator_specifications", {}).update(
            {
                o["kind"]: o["operator_specification"]
                for o in query.get("semantic_operators", [])
                if "operator_specification" in o
            }
        )
        self.catalog = self.profile.get("models", {})
        self.constraints = self.profile.get("constraints", {})
        if self.profile.get("query_key", self.key) != self.key:
            raise ValueError("Statistics profile belongs to another query")
        if set(self.constraints) - {
            "allowed_models",
            "gpu_memory_bytes",
            "cpu_memory_bytes",
            "a_min",
        }:
            raise ValueError("Unsupported hard constraints")
        for name, value in self.constraints.items():
            if name.endswith("memory_bytes") and (not self._number(value) or value < 0):
                raise ValueError("Memory budget must be finite and nonnegative")
        minimum = self.constraints.get("a_min")
        if minimum is not None and (not self._number(minimum) or not 0 <= minimum <= 1):
            raise ValueError("a_min must be in [0,1]")
        self.logical = {n["node_id"]: n for n in query["operators"]}
        if len(self.logical) != len(query["operators"]) or not self.logical:
            raise ValueError("Logical node IDs must be unique and nonempty")
        if self.native:
            from planning.cost.estimates import relational_prior

            for node in self.logical.values():
                if "semantic_operator_id" not in node:
                    info = self.profile.setdefault("node_statistics", {}).setdefault(
                        str(node["node_id"]), {}
                    )
                    for key, value in relational_prior(node, self.logical).items():
                        info.setdefault(key, value)
        self.ops, self.all_ops, self.groups = {}, {}, query.get("order_groups", [])
        if not self.native:
            allowed = {
                "scan",
                "filter",
                "projection",
                "join",
                "semantic_filter",
                "semantic_projection",
                "semantic_join",
                "semantic_nested_loop_join",
            }
            if any(
                n.get("kind", "").lower() not in allowed for n in self.logical.values()
            ):
                raise ValueError("LogicalQuery supports SPJ operators only")
            leaves = [n for n in self.logical.values() if n["kind"].lower() == "scan"]
            relations = [n.get("relation_id", n["node_id"]) for n in leaves]
            fields = [field_key(f) for n in leaves for f in n.get("output_schema", [])]
            if len(set(relations)) != len(relations) or len(set(fields)) != len(fields):
                raise ValueError("Base relations and field identities must be unique")
            if any(n.get("children") for n in self.logical.values()):
                raise ValueError(
                    "LogicalQuery inputs are selected by actions, not fixed children"
                )
        if self.native:
            self.ops = {o["operator_id"]: o for o in query["semantic_operators"]}
            semantic_nodes = {o["node_id"]: o for o in self.ops.values()}
            ids = [i for g in self.groups for i in g]
            if (
                len(ids) != len(set(ids))
                or set(ids) != set(self.ops)
                or any(not g for g in self.groups)
            ):
                raise ValueError("Invalid reorder regions")
            # Negative relational IDs cannot collide with native semantic IDs.
            for node_id, node in self.logical.items():
                if node.get("children"):
                    op = copy.deepcopy(semantic_nodes.get(node_id, node))
                    op["operator_id"] = op.get("operator_id", -node_id - 1)
                    op["node_id"] = node_id
                    self.all_ops[op["operator_id"]] = op
                    # Keep the original bound ratio separate from observations.
                    # Rebase this weak prior on each candidate's actual inputs.
                    prior = relational_prior(node, self.logical)
                    if self._number(prior.get("duckdb_selectivity")):
                        op["duckdb_selectivity"] = prior["duckdb_selectivity"]
            self.fixed_inputs = tuple(
                (k, tuple(n["children"])) for k, n in self.logical.items()
            )
            descendants(self.fixed_inputs)
        else:
            for node_id, node in self.logical.items():
                if node.get("kind", "").lower() == "scan":
                    continue
                op = copy.deepcopy(node)
                self.all_ops[op["operator_id"]] = op
                if semantic_kind(op):
                    self.ops[op["operator_id"]] = op
            self.fixed_inputs = ()
        if len(self.all_ops) != sum(
            1
            for n in self.logical.values()
            if (
                n.get("children")
                if self.native
                else n.get("kind", "").lower() != "scan"
            )
        ):
            raise ValueError("Operator IDs must be unique")
        for oid, op in self.all_ops.items():
            if type(oid) is not int:
                raise ValueError("Operator IDs must be integers")
            if set(op.get("depends_on", [])) - set(self.all_ops) or oid in op.get(
                "depends_on", []
            ):
                raise ValueError("Invalid logical dependency")
            targets = op.get("pair_targets", [])
            if targets and (
                op.get("type", op.get("kind", "")).upper() != "FILTER"
                or any(
                    type(n) is not int
                    or n not in self.logical
                    or "join"
                    not in self.logical[n]
                    .get("type", self.logical[n].get("kind", ""))
                    .lower()
                    or not semantic_kind(self.logical[n])
                    for n in targets
                )
            ):
                raise ValueError("Invalid candidate-pair targets")
        dep_graph = [
            (i, tuple(o.get("depends_on", []))) for i, o in self.all_ops.items()
        ]
        descendants(dep_graph)
        self.domain = {
            i: self._domain(o) if i in self.ops else [{"operator_id": i}]
            for i, o in self.all_ops.items()
        }
        from .duplicates import duplicate_groups, implementation

        self.duplicate_groups = duplicate_groups(self)
        for group in set(self.duplicate_groups.values()):
            common = set.intersection(*[
                {implementation(a) for a in self.domain[oid]} for oid in group
            ])
            if not common:
                raise ValueError("Repeated semantic operators have no common model/layout")
            for oid in group:
                self.domain[oid] = [a for a in self.domain[oid]
                                    if implementation(a) in common]
        self.domain_keys = {
            i: {canonical(a) for a in actions} for i, actions in self.domain.items()
        }
        self.tree_regions = query.get("tree_regions", []) if self.native else []
        self.tree_nodes = {}
        for region in self.tree_regions:
            members, bases = region["nodes"], region["base_nodes"]
            if (
                not members
                or region["region_id"] not in members
                or len(set(members)) != len(members)
                or len(set(bases)) != len(bases)
                or set(members) & set(bases)
            ):
                raise ValueError("Invalid model tree region")
            if any(n not in self.logical for n in members + bases):
                raise ValueError("Unknown model tree node")
            for nid in members:
                node = self.logical[nid]
                if (
                    nid in self.tree_nodes
                    or not 1 <= len(node["children"]) <= 2
                    or not set(node["children"]) <= set(members + bases)
                    or node["type"]
                    not in (
                        "FILTER",
                        "SEMANTIC_FILTER",
                        "SEMANTIC_PROJECTION",
                        "SEMANTIC_JOIN",
                        "CROSS_PRODUCT",
                        "COMPARISON_JOIN",
                        "ANY_JOIN",
                    )
                ):
                    raise ValueError("Overlapping or disconnected model tree region")
                self.tree_nodes[nid] = region
            graph_roots = set(members) - {
                c for nid in members for c in self.logical[nid]["children"]
            }
            if graph_roots != {region["region_id"]}:
                raise ValueError("Model tree region requires one original root")
        self.predicate_regions = (
            query.get("predicate_regions", []) if self.native else []
        )
        self.predicate_nodes = {}
        for region in self.predicate_regions:
            members = region["nodes"]
            if (
                not members
                or region["region_id"] != members[-1]
                or region["base_node_id"] not in self.logical
            ):
                raise ValueError("Invalid predicate region")
            for nid in members:
                if nid in self.predicate_nodes or nid not in self.logical:
                    raise ValueError("Overlapping or unknown predicate region node")
                node = self.logical[nid]
                if len(node.get("children", [])) != 1 or node["type"] not in (
                    "FILTER",
                    "SEMANTIC_FILTER",
                    "SEMANTIC_PROJECTION",
                ):
                    raise ValueError("Unsupported predicate region node")
                self.predicate_nodes[nid] = region
            previous = region["base_node_id"]
            for nid in members:
                if self.logical[nid]["children"] != [previous]:
                    raise ValueError(
                        "Predicate region must be a contiguous unary chain"
                    )
                previous = nid
            if not any(self.logical[n]["type"] == "FILTER" for n in members) or not any(
                self.logical[n]["type"].startswith("SEMANTIC_") for n in members
            ):
                raise ValueError(
                    "Predicate region needs ordinary and semantic operators"
                )
        self.cache_history = tuple(
            copy_profile({"history": self.profile.get("history", [])})["history"]
        )
        leaves = []
        for nid, node in self.logical.items():
            if (
                (not node.get("children"))
                if self.native
                else node.get("kind", "").lower() == "scan"
            ):
                relation = node.get("relation_id", nid)
                leaves.append(
                    PartialPlanNode(
                        nid,
                        fields=tuple(node.get("output_schema", [])),
                        relations=frozenset([relation]),
                    )
                )
        if not leaves:
            raise ValueError("At least one base relation is required")
        partial_plans = PartialPlanSet(
            tuple(leaves),
            tuple(n.node_id for n in leaves),
            tuple(sorted(o["node_id"] for o in self.all_ops.values())),
            tuple((n.node_id, ()) for n in leaves),
        )
        initial_statistics = copy_statistics(self.profile.get("statistics", {}))
        if self.profile.get("sampling_strategy") == "independent_base":
            for leaf in leaves:
                source = self.profile.get("source_statistics", {}).get(
                    str(leaf.node_id)
                )
                if source is None:
                    grids = {}
                    for evidence in self.profile.get(
                        "operator_statistics", {}
                    ).values():
                        for field, depths in evidence.get(
                            "prefix_selectivity_statistics",
                            evidence.get("depth_selectivity_statistics", {}),
                        ).items():
                            if any(
                                field.startswith(f"{relation}:")
                                for relation in leaf.relations
                            ):
                                grids.setdefault(field, {}).update(
                                    dict.fromkeys(depths, 1.0)
                                )
                    source = {
                        **self.profile.get("node_statistics", {}).get(
                            str(leaf.node_id), {}
                        ),
                        "estimated_output_survival": grids,
                    }
                    source.setdefault(
                        "output_rows", self.logical[leaf.node_id].get("estimated_rows")
                    )
                if source is not None:
                    initial_statistics[f"scan:{leaf.node_id}"] = copy_statistics(
                        {"source": source}
                    )["source"]
        self.initial = SearchState(
            statistics=initial_statistics,
            sample={},
            partial_plans=partial_plans,
        )
        if backend == "native":
            from ..kernels.state import NativeStateOwner

            self._native_owner = NativeStateOwner(self)
            handle = self._native_owner.problem.initial()
            self.initial = SearchState(
                statistics=self.initial.statistics,
                sample=self.initial.sample,
                partial_plans=self._native_owner.partial_plans(handle),
                _native_handle=handle,
            )

    @property
    def native_problem(self):
        return (
            self._native_owner.sync_resources(self)
            if self._native_owner is not None
            else None
        )

    @staticmethod
    def _number(value) -> TypeGuard[int | float]:
        return type(value) in (int, float) and math.isfinite(value)

    def complete(self, state):
        native = self.native_problem
        if native is not None:
            return native.complete(state._native_handle)
        return not state.partial_plans.remaining and len(state.partial_plans.roots) == 1

    def _domain(self, op):
        default = op["default_layout"]
        blocks = {"DATA_BLOCK", "TASK_DESCRIPTION", "OPERATOR_SPECIFICATION"}
        if (
            len(default.get("block_order", [])) != 3
            or set(default["block_order"]) != blocks
        ):
            raise ValueError("Layout must contain the three O/T/D blocks")
        keys = set(default) - {"block_order"}
        expected = (
            {"left_field_order", "right_field_order"}
            if "join" in op["kind"].lower()
            else {"field_order"}
        )
        if keys != expected or any(
            not default[k] or len(default[k]) != len(set(default[k])) for k in keys
        ):
            raise ValueError("Invalid field layout domain")
        names = op.get("input_fields") or op.get("left_input_fields", []) + op.get(
            "right_input_fields", []
        )
        if any(
            int(index) >= len(names) for index in re.findall(r"\{(\d+)\}", op["task"])
        ):
            raise ValueError("Task placeholder is outside the input field domain")
        if not op["models"] or len(set(op["models"])) != len(op["models"]):
            raise ValueError("Semantic candidate models must be nonempty and unique")
        curated = self.profile.get("layouts", {}).get(str(op["operator_id"]))
        if curated is None:
            size = len(BLOCK_ORDERS) * math.prod(
                max(1, len(v)) for k, v in default.items() if k != "block_order"
            )
            if size * len(op["models"]) > self.max_actions:
                raise ValueError(
                    "Layout domain exceeds max_actions; supply explicit legal layouts"
                )
            keys = list(default)
            curated = [
                dict(zip(keys, map(list, values)))
                for values in itertools.product(
                    *(
                        (
                            BLOCK_ORDERS
                            if k == "block_order"
                            else first_field_orders(default[k])
                        )
                        for k in keys
                    )
                )
            ]
        result = []
        for layout in curated:
            if set(layout) != set(default) or any(
                sorted(layout[k]) != sorted(default[k]) for k in default
            ):
                raise ValueError("Invalid layout; join sides cannot be swapped")
            if tuple(layout["block_order"]) not in BLOCK_ORDERS:
                raise ValueError("Prompt block order must be OTD, ODT or DOT")
            if any(
                layout[k][1:] != [f for f in default[k] if f != layout[k][0]]
                for k in default
                if k != "block_order"
            ):
                raise ValueError(
                    "Only the first field may vary; remaining fields retain binding order"
                )
            for model in op["models"]:
                meta = self.catalog.get(model, {})
                if (
                    "supported_kinds" in meta
                    and op.get("kind") not in meta["supported_kinds"]
                ):
                    continue
                if model not in self.constraints.get("allowed_models", op["models"]):
                    continue
                result.append(
                    {
                        "operator_id": op["operator_id"],
                        "model_id": model,
                        "layout": copy.deepcopy(layout),
                    }
                )
        return sorted({canonical(a): a for a in result}.values(), key=canonical)

    def feasible_action(self, state, action):
        native = self.native_problem
        if native is not None:
            return native.feasible_action(state._native_handle, action)
        from .duplicates import matching_implementation

        if not matching_implementation(self, state, action):
            return False
        if not any(
            device + "_memory_bytes" in self.constraints for device in ("gpu", "cpu")
        ):
            return True
        used = {a["model_id"] for a in state.action_views if "model_id" in a}
        used |= {action["model_id"]} if "model_id" in action else set()
        used |= {m for m, meta in self.catalog.items() if meta.get("loaded")}
        for device in ("gpu", "cpu"):
            limit = self.constraints.get(device + "_memory_bytes")
            if limit is None:
                continue
            total = 0
            for model in used:
                for suffix in ("weight_bytes", "kv_reserve_bytes"):
                    value = self.catalog.get(model, {}).get(device + "_" + suffix)
                    if not self._number(value) or value < 0:
                        return False
                    total += value
            if total > limit:
                return False
        return True

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
            from ..kernels.state import core

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
        from ..sampling import SamplingCoordinator

        # The synchronous observer publishes isolated mutable statistics. Borrow
        # only until it returns; unobserved and custom paths still own copies.
        stats = (
            dict(state.statistics)
            if (
                observe
                and type(self.sampler) is SamplingCoordinator
                and getattr(self.observe, "__func__", None) is SearchProblem.observe
            )
            else copy_statistics(state.statistics)
        )
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
            result = copy.deepcopy(
                self.profile.get("node_statistics", {}).get(str(op["node_id"]), {})
            )
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

            from planning.sampling.independent import (
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
