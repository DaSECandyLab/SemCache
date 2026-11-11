"""One-time typed configuration and read-only views of resident native states.

The production resident engine imports sampling configuration once and owns
observations and sample populations in C++. These Python views are used at
startup, manual compatibility boundaries and after the search loop returns.
"""

import copy
import json
import weakref
from collections.abc import Sequence
from functools import cached_property
from typing import Any

from planning.observations.evidence import freeze_evidence


def core():
    try:
        from . import _search_core
    except ImportError as exc:
        raise RuntimeError(
            "Native planner is required; run make planning-native at the SemCache root with the optimizer "
            'Python interpreter (or explicitly select backend="python" for comparison)'
        ) from exc
    return _search_core


class NativePlanNode:
    __slots__ = ("_handle", "_fields", "_relations", "_pair_columns", "__weakref__")

    def __init__(self, handle, fields):
        object.__setattr__(self, "_handle", handle)
        object.__setattr__(self, "_fields", tuple(fields[i] for i in handle.field_ids))
        object.__setattr__(
            self, "_relations", frozenset(json.loads(k) for k in handle.relation_keys)
        )
        object.__setattr__(
            self,
            "_pair_columns",
            tuple(
                tuple(
                    fields[i].get("name", self._field_key(fields[i]))
                    if isinstance(fields[i], dict)
                    else fields[i]
                    for i in side
                )
                for side in handle.pair_field_ids
            ),
        )

    @staticmethod
    def _field_key(value):
        from ..state import field_key

        return field_key(value)

    def __setattr__(self, name, value):
        raise AttributeError("Native partial-plan nodes are read-only")

    node_id = property(lambda self: self._handle.node_id)
    children = property(lambda self: tuple(self._handle.children))
    height = property(lambda self: self._handle.height)
    operator_id = property(lambda self: self._handle.operator_id)
    fields = property(lambda self: self._fields)
    relations = property(lambda self: self._relations)
    pending_join = property(lambda self: self._handle.pending_join)
    pair_columns = property(lambda self: self._pair_columns)

    def __eq__(self, other):
        names = (
            "node_id",
            "children",
            "height",
            "operator_id",
            "fields",
            "relations",
            "pending_join",
            "pair_columns",
        )
        if not all(hasattr(other, name) for name in names):
            return NotImplemented
        return all(getattr(self, name) == getattr(other, name) for name in names)

    __hash__: Any = None

    def __deepcopy__(self, memo):
        return self


class NativeNodes(Sequence):
    """Materialize only requested node adapters, sharing their immutable handles."""

    def __init__(self, handle, owner):
        self._handle, self._owner = handle, owner
        self._cache = {}

    def __len__(self):
        return self._handle.node_count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index not in self._cache:
            self._cache[index] = self._owner.node(self._handle.node_at(index))
        return self._cache[index]


class NativePartialPlanSet:
    _handle: Any
    _owner: "NativeStateOwner"

    def __init__(self, handle, owner):
        object.__setattr__(self, "_handle", handle)
        object.__setattr__(self, "_owner", owner)

    def __setattr__(self, name, value):
        raise AttributeError("Native partial-plan sets are read-only")

    @cached_property
    def nodes(self):
        return NativeNodes(self._handle, self._owner)

    @property
    def roots(self):
        return tuple(self._handle.roots)

    @property
    def remaining(self):
        return tuple(self._handle.remaining)

    @cached_property
    def inputs(self):
        return tuple((node.node_id, node.children) for node in self.nodes)

    def __eq__(self, other):
        if not all(
            hasattr(other, name) for name in ("nodes", "roots", "remaining", "inputs")
        ):
            return NotImplemented
        return (
            self.roots == other.roots
            and self.remaining == other.remaining
            and self.inputs == other.inputs
            and tuple(self.nodes) == tuple(other.nodes)
        )

    __hash__: Any = None

    def __deepcopy__(self, memo):
        return self


class NativeStateOwner:
    def __init__(self, problem):
        from ..state import canonical, field_key

        fields, field_ids = [], {}

        def field_id(value):
            key = canonical(value)
            if key not in field_ids:
                field_ids[key] = len(fields)
                fields.append(copy.deepcopy(value))
            return field_ids[key]

        def field_list(values):
            return [field_id(f) for f in values]

        def keys(values):
            return sorted({field_key(f) for f in values})

        def scope(values):
            return sorted({canonical(value) for value in values})

        nodes = []
        for nid, node in problem.logical.items():
            nodes.append(
                {
                    "node_id": nid,
                    "type": node.get("type", node.get("kind", "")).upper(),
                    "children": node.get("children", []),
                    "fields": field_list(node.get("output_schema", [])),
                    "required": keys(node.get("required_columns", [])),
                    "left_required": keys(node.get("left_required_columns", [])),
                    "right_required": keys(node.get("right_required_columns", [])),
                    "pair_required": [
                        keys(side) for side in node.get("candidate_pair_columns", [])
                    ],
                    "left_scope": scope(node.get("left_scope", [])),
                    "right_scope": scope(node.get("right_scope", [])),
                    "commutative": bool(node.get("join_commutative", False)),
                    "join_type": node.get("join_type", ""),
                    "pair_targets": node.get("pair_targets", []),
                }
            )
        from planning.search.heuristics import fixed_layout, operator_prerequisites

        priority_dependencies = operator_prerequisites(problem)

        operators = []
        for oid, op in problem.all_ops.items():
            operators.append(
                {
                    "operator_id": oid,
                    "node_id": op["node_id"],
                    "kind": op.get("kind", op.get("type", "")).lower(),
                    "ordinary_filter": op.get("type", op.get("kind", "")).upper()
                    == "FILTER",
                    "semantic": oid in problem.ops,
                    "duplicates": list(problem.duplicate_groups.get(oid, ())),
                    "depends_on": op.get("depends_on", []),
                    "priority_dependencies": sorted(priority_dependencies[oid]),
                    "required": keys(op.get("required_columns", [])),
                    "scope": scope(op.get("scope", [])),
                    "left_scope": scope(op.get("left_scope", [])),
                    "right_scope": scope(op.get("right_scope", [])),
                    "produced": field_list(op.get("produced_fields", [])),
                    "output": field_list(op.get("output_schema", [])),
                    "pair_targets": op.get("pair_targets", []),
                    "domain": [canonical(a) for a in problem.domain[oid]],
                    "fixed_layout": fixed_layout(op) if oid in problem.ops else None,
                }
            )
        leaves = [
            {
                "node_id": n.node_id,
                "fields": field_list(n.fields),
                "relations": scope(n.relations),
            }
            for n in problem.initial.partial_plans.nodes
        ]
        config = {
            "native": problem.native,
            "max_actions": problem.max_actions,
            "nodes": nodes,
            "operators": operators,
            "leaves": leaves,
            "groups": problem.groups,
            "tree_regions": problem.tree_regions,
            "predicate_regions": problem.predicate_regions,
            "output_required": keys(problem.query.get("output_schema", [])),
            "field_keys": [field_key(f) for f in fields],
            "models": problem.catalog,
            "constraints": problem.constraints,
        }
        self.fields = tuple(freeze_evidence(value) for value in fields)
        self._nodes = weakref.WeakValueDictionary()
        self._resources = None
        self.problem = core().StateProblem(config)

    def sync_resources(self, problem):
        keys = (
            "loaded",
            "gpu_weight_bytes",
            "gpu_kv_reserve_bytes",
            "cpu_weight_bytes",
            "cpu_kv_reserve_bytes",
        )

        def scalar(value):
            # bool equals 0/1 in Python but is invalid numeric memory evidence.
            return type(value), value

        signature = (
            scalar(problem.constraints.get("gpu_memory_bytes")),
            scalar(problem.constraints.get("cpu_memory_bytes")),
            tuple(
                (name, tuple(scalar(meta.get(key)) for key in keys))
                for name, meta in problem.catalog.items()
            ),
        )
        if signature != self._resources:
            self.problem.update_resources(problem.catalog, problem.constraints)
            self._resources = signature
        return self.problem

    def node(self, handle):
        cached = self._nodes.get(handle.handle_id)
        if cached is None:
            cached = self._nodes[handle.handle_id] = NativePlanNode(handle, self.fields)
        return cached

    def partial_plans(self, handle):
        return NativePartialPlanSet(handle, self)
