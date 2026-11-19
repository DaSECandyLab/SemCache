"""Prompt namespace serialization and field input extraction."""

import json
import math
import re
from functools import lru_cache

from planning.state import canonical

FIELD_SERIALIZER = "prompt-v4-first-field-value-v1"

def prefix_length(namespace):
    """Characters of system/user content before the first Data Block.

    The namespace also encodes serializer/version and JSON escapes. Those
    identity bytes are not part of the prompt length feature.
    """
    if namespace is None:
        return None
    try:
        parts = json.loads(namespace)
    except (TypeError, ValueError):
        return None
    if (
        not isinstance(parts, list)
        or len(parts) != 3
        or not all(isinstance(p, str) for p in parts)
    ):
        return None
    return len(parts[1]) + len(parts[2].split("<DATA_BLOCK>", 1)[0])


def prefix(op, action, system, operator_spec):
    layout = action["layout"]
    fields = layout.get(
        "field_order",
        layout.get("left_field_order", []) + layout.get("right_field_order", []),
    )
    inputs = (
        op["input_fields"]
        if op["input_fields"]
        else op["left_input_fields"] + op["right_input_fields"]
    )
    inverse = {inputs.index(f): i for i, f in enumerate(fields)}
    task = re.sub(
        r"\{(\d+)\}", lambda m: "{" + str(inverse[int(m[1])]) + "}", op["task"]
    )
    blocks = {"OPERATOR_SPECIFICATION": operator_spec, "TASK_DESCRIPTION": task}
    parts = []
    for block in layout["block_order"]:
        if block == "DATA_BLOCK":
            break
        parts.append(f"<{block}>\n{blocks[block]}\n</{block}>")
    # Global system role/content and v4 serialization remain in the namespace.
    # Do not include the full prompt length: it changes with each data value.
    return canonical(
        [
            "prompt-v4",
            system,
            "\n\n".join(parts) + ("\n\n" if parts else "") + "<DATA_BLOCK>\n" + "{0}: ",
        ]
    )


def request_inputs(op, rows):
    """Recover original input-expression identities from dense display numbering."""
    recovered = []
    expected = len(
        op["input_fields"] or op["left_input_fields"] + op["right_input_fields"]
    )
    for row in rows:
        # 原始绑定顺序与 prompt 展示顺序不同；结构化字段避免任务文本中的标签
        # 被误读成 DATA_BLOCK，也让字段重排后的统计仍对应同一输入表达式。
        values = row.get("input_values")
        if (
            not isinstance(values, list)
            or len(values) != expected
            or not all(isinstance(v, str) for v in values)
        ):
            raise ValueError(
                "Structured telemetry inputs do not match bound fields; regenerate old telemetry"
            )
        recovered.append(dict(enumerate(values)))
    return recovered


def _field_row_indices(op, rows):
    """A join profiles each distinct left/right tuple once, not every pair."""
    if op["input_fields"]:
        return {field: range(len(rows)) for field in op["input_fields"]}
    indices = {}
    for side in ("left", "right"):
        seen, selected = set(), []
        for index, row in enumerate(rows):
            identity = row[f"{side}_index"]
            if identity not in seen:
                seen.add(identity)
                selected.append(index)
        indices.update(dict.fromkeys(op[f"{side}_input_fields"], selected))
    return indices

def serialize_value(value):
    """Match the prompt JSON-scalar character domain (including escaped tags)."""
    if type(value) is str and len(value) <= 4096:
        return _serialize_text(value)
    # Exact built-in scalars have the same representation as json.dumps. Keep
    # subclasses and compound values on the encoder path, and reject nonfinite
    # floats there so the prompt serializer's error behavior stays unchanged.
    if value is None:
        return "null"
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int or (type(value) is float and math.isfinite(value)):
        return repr(value)
    return _serialize_uncached(value)


@lru_cache(maxsize=1024)
def _serialize_text(value):
    """Bounded immutable text reuse across sampling replay; no compound aliasing."""
    return _serialize_uncached(value)


def _serialize_uncached(value):
    import json

    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        .replace("<", r"\u003c")
        .replace(">", r"\u003e")
    )
