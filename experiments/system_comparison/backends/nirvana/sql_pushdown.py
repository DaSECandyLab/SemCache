"""Keep deterministic SQL filters ahead of eligible relational joins."""

from experiments.system_comparison.execution.operators import Predicate, SQLJoin
from experiments.validation.runtime.native_graph import Scalar


def _tool(node):
    tool = getattr(node.operator, "tool", None)
    return getattr(tool, "func", tool)


def push_sql_filters(root):
    """Push only SQL predicates across safe, unlimited relational operators.

    The native optimizer still chooses the plan. This subsequent relational
    equivalence avoids materializing a large join before a side-only SQL
    predicate. Semantic operators, outer joins, limits and ambiguous field
    ownership are barriers.
    """
    moves = []

    def move(predicate, child):
        dependencies = set(predicate.operator.dependencies)
        if not dependencies or getattr(child.operator, "limit", None) is not None:
            predicate.set_left_child(child)
            return predicate

        if child.op_name == "join" and isinstance(_tool(child), SQLJoin):
            if child.operator.how == "inner":
                left = set(child.node_fields.left_input_fields)
                right = set(child.node_fields.right_input_fields)
                left_only = dependencies <= left and not dependencies & right
                right_only = dependencies <= right and not dependencies & left
                if left_only or right_only:
                    branch = child.left_child if left_only else child.right_child
                    if dependencies <= set(branch.node_fields.output_fields):
                        fields = list(branch.node_fields.output_fields)
                        predicate.node_fields.left_input_fields = fields.copy()
                        predicate.node_fields.output_fields = fields.copy()
                        moved = move(predicate, branch)
                        if left_only:
                            child.set_left_child(moved)
                        else:
                            child.set_right_child(moved)
                        moves.append({"predicate": predicate.operator.user_instruction,
                                      "across": "sql_inner_join"})
                        return child

        if child.op_name in ("filter", "map"):
            child_tool = _tool(child)
            safe = (child.op_name == "filter" and isinstance(child_tool, Predicate)) or (
                child.op_name == "map" and isinstance(child_tool, Scalar)
            )
            branch = child.left_child
            if safe and branch is not None and dependencies <= set(branch.node_fields.output_fields):
                # Do not move through a map which overwrites an input field.
                if child.op_name != "map" or not dependencies.intersection(child.operator.generated_fields):
                    fields = list(branch.node_fields.output_fields)
                    predicate.node_fields.left_input_fields = fields.copy()
                    predicate.node_fields.output_fields = fields.copy()
                    child.set_left_child(move(predicate, branch))
                    moves.append({"predicate": predicate.operator.user_instruction,
                                  "across": "sql_" + child.op_name})
                    return child
        predicate.set_left_child(child)
        return predicate

    def visit(node):
        if node.left_child is not None:
            node.set_left_child(visit(node.left_child))
        if node.right_child is not None:
            node.set_right_child(visit(node.right_child))
        if (node.op_name == "filter" and isinstance(_tool(node), Predicate)
                and getattr(node.operator, "limit", None) is None
                and node.left_child is not None
                and node.node_fields.output_fields == node.left_child.node_fields.output_fields):
            return move(node, node.left_child)
        return node

    return visit(root), moves
