"""Canonical query, action, task and field identities."""

import hashlib
import json


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def query_key(query):
    return digest(
        {
            k: v
            for k, v in query.items()
            if k not in ("query_id", "environment", "training_group", "training_split")
        }
    )


def action_key(action):
    # Input roots are part of the key: observations at different boundaries differ.
    return digest(action)


def task_key(op):
    """Logical semantic task, independent of the current query's column bindings."""
    return digest(
        [
            op["kind"],
            op["task"],
            [(f["name"], f.get("type")) for f in op.get("input_field_schema", [])],
            op.get("output_schema", []),
        ]
    )


def first_field_orders(fields):
    return [[first] + [f for f in fields if f != first] for first in fields] or [[]]


def field_key(column):
    if isinstance(column, dict):
        column = column.get(
            "binding",
            column.get("identity", column.get("name", column.get("expression"))),
        )
    return canonical(column)


def semantic_kind(op):
    return op.get("kind", op.get("type", "")).upper().startswith("SEMANTIC")
