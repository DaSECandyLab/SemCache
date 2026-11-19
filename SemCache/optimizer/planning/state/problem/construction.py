"""SearchProblem initialization, native lowering and domain validation."""

import copy
import itertools
import math
import re
from collections.abc import Callable
from typing import Any, TypeGuard

from planning.prompt_layouts import BLOCK_ORDERS
from planning.state.partial_plans import PartialPlanNode, PartialPlanSet, descendants

from ..copying import copy_profile, copy_statistics
from ..identity import (
    canonical,
    field_key,
    first_field_orders,
    query_key,
    semantic_kind,
)
from ..model import SearchState


class _ConstructionMixin:
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
            from ..cte import connect_references

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
        from ..duplicates import duplicate_groups, implementation

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
            from ...kernels.state import NativeStateOwner

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
            full_size = len(BLOCK_ORDERS) * math.prod(
                math.factorial(len(v))
                for k, v in default.items()
                if k != "block_order"
            )
            use_full = full_size * len(op["models"]) <= self.max_actions
            if not use_full:
                first_size = len(BLOCK_ORDERS) * math.prod(
                    max(1, len(v)) for k, v in default.items() if k != "block_order"
                )
                if first_size * len(op["models"]) > self.max_actions:
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
                            else (
                                itertools.permutations(default[k])
                                if use_full
                                else first_field_orders(default[k])
                            )
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
        from ..duplicates import matching_implementation

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
