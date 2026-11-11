"""Validate declared result types and explicit label domains without changing answers."""

import math
from collections.abc import Mapping

OUTPUT_VALIDATION_PROTOCOL = "typed-output-validation-v2"


def _fail(message, context):
    raise ValueError(f"{context}: {message}" if context else message)


def _same(first, second):
    # Python considers True == 1; JSON labels must retain their scalar type.
    return type(first) is type(second) and first == second


def _member(value, values):
    return any(_same(value, allowed) for allowed in values)


def _typed_value(value, declared):
    if declared in {"VARCHAR", "TEXT"}:
        return isinstance(value, str)
    if declared == "BOOLEAN":
        return type(value) is bool
    if declared in {"INTEGER", "BIGINT"}:
        lower, upper = (
            (-(2**31), 2**31 - 1) if declared == "INTEGER" else (-(2**63), 2**63 - 1)
        )
        return type(value) is int and lower <= value <= upper
    if declared in {"DOUBLE", "FLOAT", "REAL"}:
        return type(value) in (int, float) and math.isfinite(value)
    return False


def validate_output_contract(output_contract, *, context=""):
    """Check typed outputs and any explicitly requested output vocabulary."""
    if output_contract is None:
        return
    if not isinstance(output_contract, Mapping):
        _fail("Output contract must be an object", context)
    generated = output_contract.get("generated_columns", {})
    if not isinstance(generated, Mapping):
        _fail("Output contract generated_columns must be an object", context)
    for name, specification in generated.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(specification, Mapping)
        ):
            _fail("Invalid generated column specification", context)
        values = specification.get("values")
        if values is None:
            declared = specification.get("type")
            if declared not in {
                "VARCHAR",
                "TEXT",
                "BOOLEAN",
                "INTEGER",
                "BIGINT",
                "DOUBLE",
                "FLOAT",
                "REAL",
            }:
                _fail(f"Generated column {name!r} has no supported type", context)
            if type(specification.get("nullable", True)) is not bool:
                _fail(f"Generated column {name!r} nullable must be boolean", context)
            if "missing_value" in specification and not _typed_value(
                specification["missing_value"], declared
            ):
                _fail(
                    f"Generated column {name!r} missing_value has the wrong type",
                    context,
                )
            continue
        if not isinstance(values, list) or not values:
            _fail(f"Generated column {name!r} has no finite values", context)
        for index, value in enumerate(values):
            if value is not None and type(value) not in (str, int, float, bool):
                _fail(f"Generated column {name!r} has a non-scalar label", context)
            if _member(value, values[:index]):
                _fail(f"Generated column {name!r} has duplicate labels", context)
        if "missing_value" in specification and not _member(
            specification["missing_value"], values
        ):
            _fail(
                f"Generated column {name!r} missing_value is outside its values",
                context,
            )
        if (
            specification.get("nullable") is False
            or specification.get("nonnull") is True
        ):
            if _member(None, values):
                _fail(
                    f"Generated column {name!r} has contradictory NULL rules", context
                )


def validate_result_labels(result, output_contract, *, context=""):
    """Check every contracted cell against its type and explicit vocabulary.

    NULL is accepted only when declared in values or nullable=True. A declared
    missing_value is a label, not permission to replace NULL or an empty string.
    Empty strings remain valid when explicitly included in values.
    """
    validate_output_contract(output_contract, context=context)
    generated = (output_contract or {}).get("generated_columns", {})
    if not generated:
        return
    if not isinstance(result, Mapping):
        _fail("Result must be an object", context)
    columns, rows = result.get("columns"), result.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        _fail("Result must declare columns and rows", context)
    names = []
    for column in columns:
        name = (
            column
            if isinstance(column, str)
            else column[0]
            if isinstance(column, (list, tuple)) and column
            else None
        )
        if not isinstance(name, str):
            _fail("Invalid result column schema", context)
        if name in names:
            _fail(f"Ambiguous duplicate result column {name!r}", context)
        names.append(name)
    missing = set(generated) - set(names)
    if missing:
        _fail(f"Contracted result columns are missing: {sorted(missing)}", context)
    indexed = [
        (names.index(name), name, specification)
        for name, specification in generated.items()
    ]
    for number, row in enumerate(rows, 1):
        if not isinstance(row, (list, tuple)) or len(row) != len(names):
            _fail(f"Result row {number} does not match its column schema", context)
        for index, name, specification in indexed:
            value = row[index]
            nullable = (
                specification.get("nullable") is True
                and specification.get("nonnull") is not True
            )
            if value is None and nullable:
                continue
            values = specification.get("values")
            if values is None:
                if not _typed_value(value, specification["type"]):
                    _fail(
                        f"Result row {number} column {name!r} has an invalid {specification['type']} value {value!r}",
                        context,
                    )
            elif not _member(value, values):
                _fail(
                    f"Result row {number} column {name!r} has illegal label {value!r}",
                    context,
                )
