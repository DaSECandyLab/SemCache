"""Execution-independent operator inputs shared by cache fitting and runtime."""

import math

from planning.learning.features.schema import HISTORY_NUMERIC, NODE_NUMERIC

PREFIX_FEATURES = ("l_self", "l_previous", "l_existing")
OPERATOR_INPUT_SIZE = NODE_NUMERIC + HISTORY_NUMERIC + 5
FEATURE_SCHEMA = "gat-operator-numeric-history-previous-v1"
FIELD_CONTEXT_SIZE = 52
INPUT_SIZES = {
    "prefix": 3,
    "operator": OPERATOR_INPUT_SIZE,
    "operator_fields": OPERATOR_INPUT_SIZE + FIELD_CONTEXT_SIZE,
    "operator_text": OPERATOR_INPUT_SIZE + 384,
    "operator_full": OPERATOR_INPUT_SIZE + 384 + FIELD_CONTEXT_SIZE,
    "operator_tokens": OPERATOR_INPUT_SIZE + FIELD_CONTEXT_SIZE + 16,
}


def decode(value, valid=True):
    if (
        not valid
        or type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 <= value <= 700
    ):
        return None
    return math.expm1(value)


def operator_context(node):
    numeric, history = node["numeric"], node.get("l_existing")
    if len(numeric) != NODE_NUMERIC:
        raise ValueError("Cache predictor operator feature schema mismatch")
    if history is None:
        history = [0.0] * HISTORY_NUMERIC
    if len(history) != HISTORY_NUMERIC:
        raise ValueError("Cache predictor history feature schema mismatch")
    previous = [decode(v) for v in node.get("l_previous", {}).values()]
    known = [v for v in previous if v is not None]
    return {
        "operator_numeric": list(numeric),
        "history_numeric": list(history),
        "previous_summary": [
            math.log1p(len(previous)),
            math.log1p(math.fsum(known)),
            math.log1p(max(known, default=0.0)),
            math.log1p(math.fsum(known) / len(known)) if known else 0.0,
            float(len(known) == len(previous)),
        ],
        "description_ids": list(node.get("description", [0, []])[1]),
        "field_ids": list(node.get("fields", [[], []])),
        "first_field_ids": list(node.get("first_fields", [[], []])),
    }


def semantic_context(op, action, profile):
    """Use sampled semantic argument distributions, rather than whole-row width."""
    stats = profile.get("operator_statistics", {}).get(str(op["operator_id"]), {})
    columns = stats.get("input_field_statistics", {})
    schema = {field["name"]: field for field in op.get("input_field_schema", [])}
    layout = action["layout"]
    left = layout.get("field_order", layout.get("left_field_order", []))
    right = layout.get("right_field_order", [])
    keys = (
        "average_length",
        "average_bytes",
        "sample_distinct_fraction",
        "sample_top_frequency",
        "null_fraction",
        "sample_effective_samples",
    )

    def field(name):
        value_key = schema.get(name, {}).get(
            "value_key", f"__semantic_input:{op['operator_id']}:{name}"
        )
        values, masks = [], []
        for index, key in enumerate(keys):
            value = columns.get(value_key, {}).get(key)
            valid = type(value) in (float, int) and math.isfinite(value) and value >= 0
            values.append(
                math.log1p(value)
                if valid and index in (0, 1, 5)
                else float(value)
                if valid
                else 0.0
            )
            masks.append(float(valid))
        return values + masks

    groups = [field(name) for name in left + right]
    first = field(left[0]) if left else [0.0] * 12
    second = field(right[0]) if right else [0.0] * 12
    means = (
        [sum(row[i] for row in groups) / len(groups) for i in range(12)]
        if groups
        else [0.0] * 12
    )
    maxima = (
        [max(row[i] for row in groups) for i in range(12)] if groups else [0.0] * 12
    )
    return {
        "semantic_field_context": first
        + second
        + means
        + maxima
        + [
            math.log1p(len(op.get("task", ""))),
            math.log1p(len(op.get("operator_specification", ""))),
            math.log1p(len(profile.get("system_prefix", ""))),
            math.log1p(len(groups)),
        ]
    }


def reuse_features(node, op=None, action=None, profile=None, *, include_tokens=False):
    numeric, history = node["numeric"], node.get("l_existing")
    previous = [decode(v) for v in node.get("l_previous", {}).values()]
    result = {
        "l_self": decode(numeric[-3], numeric[-1] == 1),
        "l_previous": max(previous, default=0.0)
        if all(v is not None for v in previous)
        else None,
        "l_existing": decode(history[1], history[13] == 1)
        if isinstance(history, list) and len(history) == HISTORY_NUMERIC
        else None,
        **operator_context(node),
        **(semantic_context(op, action, profile) if op is not None else {}),
    }
    if include_tokens:
        from planning.execution.cache_prefix import prefix_context

        result.update(prefix_context(node, op, action, profile, result))
    return result


def hashed_bag(ids, size):
    values = [0.0] * size
    for identity in ids:
        values[identity % size] += 1
    return [math.log1p(v) for v in values]


def feature_values(row, mode):
    if mode == "prefix":
        values = [row[name] for name in PREFIX_FEATURES]
    elif mode in INPUT_SIZES:
        values = (
            row["operator_numeric"] + row["history_numeric"] + row["previous_summary"]
        )
        if mode in {"operator_text", "operator_full"}:
            values += hashed_bag(row["description_ids"], 256)
            for fields in row["field_ids"] + row["first_field_ids"]:
                values += hashed_bag(fields, 32)
        if mode in {"operator_fields", "operator_full", "operator_tokens"}:
            values += row["semantic_field_context"]
        if mode == "operator_tokens":
            values += row["prompt_token_context"]
        if len(values) != INPUT_SIZES[mode]:
            raise ValueError("Invalid operator feature vector")
    else:
        raise ValueError("Unknown cache predictor feature mode")
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values
    ):
        raise ValueError("Cache predictor features must be finite and nonnegative")
    return values
