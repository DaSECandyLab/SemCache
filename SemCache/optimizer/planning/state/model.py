"""Immutable search-state views and cached structural lookups."""

import json
from dataclasses import dataclass, field
from functools import cached_property
from types import MappingProxyType
from typing import Any

from planning.state.partial_plans import descendants

from .identity import action_key, digest


@dataclass(frozen=True)
class SearchState:
    history: tuple = ()
    statistics: dict = field(default_factory=dict, compare=False)
    sample: Any = field(default=None, compare=False, repr=False)
    partial_plans: Any = field(default=None, compare=False)
    sample_observations: dict = field(default_factory=dict, compare=False, repr=False)
    _native_handle: Any = field(default=None, compare=False, repr=False)
    _action_views: Any = field(default=None, compare=False, repr=False)

    @property
    def native_handle(self):
        return self._native_handle

    @cached_property
    def action_views(self):
        """Shared immutable actions for estimation and sampling hot paths."""
        if self._action_views is not None:
            return self._action_views
        from planning.observations.evidence import freeze_evidence

        return tuple(freeze_evidence(json.loads(a)) for a in self.history)

    @cached_property
    def _action_keys(self):
        """Canonical keys belong to immutable history, never its mutable exports."""
        return tuple(action_key(action) for action in self.action_views)

    @cached_property
    def _action_by_operator(self):
        return MappingProxyType(
            {action["operator_id"]: action for action in self.action_views}
        )

    @cached_property
    def _action_keys_by_operator(self):
        return MappingProxyType(
            {
                action["operator_id"]: key
                for action, key in zip(self.action_views, self._action_keys)
            }
        )

    @cached_property
    def _history_key(self):
        return digest(self.history)

    @cached_property
    def _history_prefix_keys(self):
        return tuple(digest(self.history[:index]) for index in range(len(self.history)))

    @cached_property
    def _nodes_by_id(self):
        return MappingProxyType(
            {node.node_id: node for node in self.partial_plans.nodes}
        )

    @cached_property
    def _nodes_by_operator(self):
        return MappingProxyType(
            {
                node.operator_id: node
                for node in self._nodes_by_id.values()
                if node.operator_id is not None
            }
        )

    @cached_property
    def _descendants(self):
        # dataclasses.replace creates a fresh cache when a plan/history changes.
        return MappingProxyType(descendants(self.partial_plans.inputs))

    @property
    def actions(self):
        return [json.loads(a) for a in self.history]

    @cached_property
    def done(self):
        if self._native_handle is not None:
            return frozenset(self._native_handle.done)
        return frozenset(a["operator_id"] for a in self.action_views)


class ActionBudgetExceeded(ValueError):
    """A valid state has more ready actions than its bounded enumeration allows."""
