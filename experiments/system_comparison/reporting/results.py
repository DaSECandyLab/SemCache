"""Lossless result contracts and strong-model reference agreement."""

import datetime
import decimal
import json
import math
from collections import Counter


def value_for_type(value, kind):
    if value is None:
        return None
    kind = kind.upper()
    if kind in {
        "BIGINT",
        "INTEGER",
        "SMALLINT",
        "TINYINT",
        "HUGEINT",
        "UBIGINT",
        "UINTEGER",
        "USMALLINT",
        "UTINYINT",
    }:
        if isinstance(value, bool):
            raise ValueError("Boolean in an integer result column")
        number = decimal.Decimal(str(value))
        if not number.is_finite() or number != number.to_integral_value():
            raise ValueError(f"Invalid integer result: {value!r}")
        return int(number)
    if kind.startswith("DECIMAL"):
        number = decimal.Decimal(str(value))
        if not number.is_finite():
            raise ValueError("Non-finite decimal")
        # normalize() rounds through the process-wide 28-digit context.
        text = format(number, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return "0" if text in {"-0", ""} else text
    if kind in {"DOUBLE", "FLOAT", "REAL"}:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Non-finite number")
        return number if number else 0.0
    if kind == "BOOLEAN":
        if isinstance(value, bool):
            return value
        if str(value).lower() in {"true", "false"}:
            return str(value).lower() == "true"
        raise ValueError(f"Invalid boolean: {value!r}")
    if kind in {"VARCHAR", "TEXT"}:
        if not isinstance(value, str):
            raise ValueError(f"Invalid text result: {value!r}")
        return value
    if isinstance(value, (datetime.date, datetime.time, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return str(value)
    if isinstance(value, (dict, list, str, int, float, bool)):
        return value
    raise ValueError(f"Unsupported {kind} result value: {type(value).__name__}")


def normalize(result):
    columns = result["columns"]
    if not all(
        isinstance(column, (list, tuple)) and len(column) == 2 for column in columns
    ):
        raise ValueError("Result columns must be ordered name/type pairs")
    rows = []
    for row in result["rows"]:
        if not isinstance(row, (list, tuple)) or len(row) != len(columns):
            raise ValueError("Result row does not match its ordered schema")
        try:
            rows.append(
                [
                    value_for_type(value, kind)
                    for value, (_, kind) in zip(row, columns, strict=True)
                ]
            )
        except (decimal.InvalidOperation, TypeError, OverflowError) as error:
            raise ValueError(f"Invalid typed result: {error}") from error
    return {"columns": [list(column) for column in columns], "rows": rows}


def score(result, reference, *, count=False):
    if reference is None or result is None:
        return {
            "accuracy": None,
            "status": "reference_pending"
            if reference is None
            else "result_unavailable",
        }
    actual, expected = normalize(result), normalize(reference)
    if actual["columns"] != expected["columns"]:
        return {
            "accuracy": None,
            "status": "schema_mismatch",
            "actual_schema": actual["columns"],
            "reference_schema": expected["columns"],
        }

    def encode(row):
        return json.dumps(
            row,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )

    left, right = (
        Counter(map(encode, actual["rows"])),
        Counter(map(encode, expected["rows"])),
    )
    overlap = sum((left & right).values())
    n, m = len(actual["rows"]), len(expected["rows"])
    precision = overlap / n if n else float(not m)
    recall = overlap / m if m else float(not n)
    result = {
        "status": "scored",
        "accuracy": 2 * overlap / (n + m) if n + m else 1.0,
        "precision": precision,
        "recall": recall,
        "exact_match": left == right,
        "result_rows": n,
        "reference_rows": m,
        "metric": "Gemma4-12B reference bag F1",
    }
    if n == m == 1 and len(actual["columns"]) == 1:
        a, b = actual["rows"][0][0], expected["rows"][0][0]
        if a is not None and b is not None:
            try:
                a, b = float(a), float(b)
            except (TypeError, ValueError):
                return result
            absolute = abs(a - b)
            result.update(
                absolute_error=absolute,
                relative_error=absolute / abs(b)
                if b
                else (0.0 if not absolute else None),
                numeric_agreement=a == b
                if count
                else math.isclose(a, b, rel_tol=1e-3, abs_tol=1e-6),
                absolute_tolerance=0.0 if count else 1e-6,
                relative_tolerance=0.0 if count else 1e-3,
            )
    return result
