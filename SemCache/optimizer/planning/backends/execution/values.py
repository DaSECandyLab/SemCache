"""Typed SQL values and JSON result cells for bounded samples."""

import json
import math


def parse_json_cell(value: str | None):
    if value is None:
        raise ValueError("DuckDB returned NULL for a required JSON cell")
    return json.loads(value)


def identifier(value):
    return '"' + value.replace('"', '""') + '"'


def literal(value):
    if value is None:
        return "NULL"
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) in (int, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite sample value")
        return str(value)
    if isinstance(value, str):
        return "decode(from_hex('" + value.encode().hex() + "'))"
    if isinstance(value, dict):
        return (
            "{"
            + ",".join(identifier(k) + ":" + literal(v) for k, v in value.items())
            + "}"
        )
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(literal(v) for v in value) + "]"
    raise ValueError("Sample SQL values must be scalar")
