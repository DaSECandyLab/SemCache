"""Feature dimensions and deterministic logical/task/field identities."""

import hashlib
import math
import re
import unicodedata
from typing import Any, TypeGuard

from planning.prompt_layouts import ENCODED_BLOCK_ORDERS
from planning.state import canonical

FEATURE_VERSION = 18

BUCKETS = 4096
MODEL_SLOTS = 64
KINDS = (
    "scan",
    "filter",
    "projection",
    "join",
    "semantic_filter",
    "semantic_projection",
    "semantic_join",
    "aggregate",
    "order_by",
    "limit",
    "distinct",
    "window",
    "union",
    "unnest",
    "top_n",
    "other",
)
STATISTICS = (
    "input_rows",
    "input_width",
    "selectivity",
    "accuracy",
    "seconds_per_input",
)
IMPLEMENTATION_SIZE = MODEL_SLOTS + len(ENCODED_BLOCK_ORDERS) + 1
NODE_NUMERIC = len(KINDS) + IMPLEMENTATION_SIZE + 3 * len(STATISTICS) + 3
HISTORY_NUMERIC = 18
WORK_MODEL_TIME_OFFSET = len(KINDS) + 6
WORK_NUMERIC = WORK_MODEL_TIME_OFFSET + MODEL_SLOTS * 3
FEATURE_GROUPS = (
    "task",
    "implementation",
    "statistics",
    "cache",
    "history",
    "structure",
)


def field_id(value):
    return (
        int.from_bytes(hashlib.sha256(canonical(value).encode()).digest()[:8], "big")
        % BUCKETS
    )


def text_ids(value):
    text = unicodedata.normalize(
        "NFKC", value if isinstance(value, str) else canonical(value)
    )
    words = re.findall(r"[\u3400-\u9fff]|[^\W\u3400-\u9fff]+|[^\w\s]", text)
    if len(words) > 256:
        words = words[:128] + words[-128:]
    return [field_id(word) for word in words]


def known(value: Any) -> TypeGuard[int | float]:
    return type(value) in (int, float) and math.isfinite(value)


def log_value(value):
    return math.log1p(max(value, 0)) if known(value) else 0.0


def kind(node):
    name = node.get("kind", node.get("type", "")).lower()
    if name == "cross_product":
        return "join"
    for suffix in ("join", "filter", "projection"):
        if suffix in name:
            return ("semantic_" if name.startswith("semantic") else "") + suffix
    for label in (
        "aggregate",
        "order_by",
        "limit",
        "distinct",
        "window",
        "union",
        "unnest",
        "top_n",
    ):
        if label in name:
            return label
    return "scan" if not node.get("children") else "other"


def description(logical, semantic):
    if semantic:
        return 2, text_ids([semantic["task"], semantic.get("input_field_schema", [])])
    keys = (
        "table",
        "table_name",
        "alias",
        "relation_id",
        "source_sql",
        "required_columns",
        "expressions",
        "parameters",
        "output_schema",
    )
    return 1, text_ids({k: logical[k] for k in keys if k in logical})


def field_identity(value):
    if not isinstance(value, dict):
        return field_id(value)
    return field_id(
        value.get(
            "binding",
            value.get("value_key", value.get("expression", value.get("name"))),
        )
    )


def fields(logical, semantic=None):
    incoming = [c for schema in logical.get("input_schema", []) for c in schema]
    incoming = incoming or logical.get("required_columns", [])
    if semantic:
        incoming = semantic.get("input_field_schema") or (
            semantic["input_fields"]
            or semantic["left_input_fields"] + semantic["right_input_fields"]
        )
    return sorted({field_identity(v) for v in incoming}), sorted(
        {field_identity(v) for v in logical.get("output_schema", [])}
    )


def first_fields(op, action):
    if not action or "model_id" not in action:
        return [[], []]
    layout = action["layout"]
    schema = {f["name"]: f for f in op.get("input_field_schema", [])}
    result = []
    for order in (
        layout.get("field_order", layout.get("left_field_order")),
        layout.get("right_field_order"),
    ):
        result.append([field_identity(schema.get(order[0], order[0]))] if order else [])
    return result


def implementation(action, pre_length=None):
    result = [0.0] * IMPLEMENTATION_SIZE
    if action and "model_id" in action:
        result[field_id(("model", action["model_id"])) % MODEL_SLOTS] = 1.0
        result[
            MODEL_SLOTS
            + ENCODED_BLOCK_ORDERS.index(tuple(action["layout"]["block_order"]))
        ] = 1.0
    if action and "model_id" in action:
        result[-1] = log_value(pre_length)
    return result
