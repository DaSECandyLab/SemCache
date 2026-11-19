"""Legal mixed-plan actions and QueryIR/LogicalQuery adaptation."""

from .actions import _ActionsMixin
from .construction import _ConstructionMixin
from .replay import _ReplayMixin
from .transitions import _TransitionsMixin


class SearchProblem(
    _ConstructionMixin,
    _ActionsMixin,
    _TransitionsMixin,
    _ReplayMixin,
):
    """LogicalQuery exposes fields/scopes/dependencies; QueryIR keeps native legality.

    LogicalQuery operators contain operator_id, node_id, kind, required_columns,
    output_schema (projection), scope, and depends_on. Joins additionally declare
    left_scope/right_scope. No relational physical algorithm is searched.
    """
