"""Field identity and generated survival helpers for depth-based estimates."""

import math

from ..state import canonical, field_key
from .prefix_rates import prefix_rate


def field_tokens(fields):
    result = set()
    for field in fields:
        result.add(field_key(field))
        if isinstance(field, dict):
            if field.get("name"):
                result.add(field["name"])
        elif isinstance(field, str):
            result.add(field)
    return result


def generated_field_identities(op):
    return {
        "generated:" + descriptor.get("value_key", descriptor["name"]): canonical(
            [
                descriptor.get(
                    "sample_expression",
                    descriptor.get(
                        "binding", descriptor.get("value_key", descriptor["name"])
                    ),
                ),
                descriptor.get("type"),
            ]
        )
        for descriptor in op.get("input_field_schema", [])
    }


def generated_survivals(state, observations, node, op, local):
    """Start generated populations at their producer, then multiply ancestors.

    A preceding predicate need not read this generated value. Its independent
    scalar selectivity is the fallback when it has no coordinate for the field.
    """
    result = {}
    dependencies = field_tokens(op.get("required_columns", []))
    identities = generated_field_identities(op)
    for field, table in local.items():
        if not field.startswith("generated:"):
            continue
        name = field[len("generated:") :]

        def at(nid, depth):
            current = state._nodes_by_id[nid]
            if current.operator_id is None:
                return 1.0
            child_fields = field_tokens(
                [
                    column
                    for child in current.children
                    for column in state._nodes_by_id[child].fields
                ]
            )
            produced = field_tokens(current.fields) - child_fields
            if (
                name in produced
                or name.startswith("__semantic_input:")
                and dependencies & produced
            ):
                return 1.0
            children = [
                child
                for child in current.children
                if name in field_tokens(state._nodes_by_id[child].fields)
                or dependencies & field_tokens(state._nodes_by_id[child].fields)
            ]
            if not children:
                children = list(current.children)
            values = [at(child, depth) for child in children]
            info = observations[nid].statistics
            previous_field = field
            if field in identities:
                previous_field = next(
                    (
                        name
                        for name, identity in info.get(
                            "generated_field_identities", {}
                        ).items()
                        if identity == identities[field]
                    ),
                    field,
                )
            factor = prefix_rate(info, previous_field, depth)
            return (
                0.0
                if factor == 0 or 0 in values
                else math.prod(values) * factor
                if factor is not None and all(value is not None for value in values)
                else None
            )

        children = [
            child
            for child in node.children
            if name in field_tokens(state._nodes_by_id[child].fields)
            or dependencies & field_tokens(state._nodes_by_id[child].fields)
        ]
        if not children:
            children = list(node.children)
        result[field] = {}
        for depth in table:
            values = [at(child, depth) for child in children]
            result[field][depth] = (
                0.0
                if 0 in values
                else math.prod(values)
                if all(value is not None for value in values)
                else None
            )
    return result
