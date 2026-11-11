"""Typed ordinary SQL tools for DocETL; semantic operations stay upstream."""

from __future__ import annotations

import json

import pandas as pd

from experiments.system_comparison.reporting.results import normalize


def records(frame, fields):
    clean = frame[list(fields)].astype(object).where(pd.notna(frame[list(fields)]), None)
    result = normalize({"columns": list(fields.items()), "rows": list(clean.itertuples(index=False, name=None))})
    return [dict(zip(fields, row, strict=True)) for row in result["rows"]]


def ordinary_rows(rows, spec):
    from experiments.system_comparison.execution.operators import context_frame
    from experiments.validation.runtime.native_graph import Scalar

    fields = spec["input_fields"]
    frame = pd.DataFrame(rows, columns=list(fields), dtype=object)
    kind = spec["op"]
    if kind == "map":
        scalar = Scalar(spec["options"]["expressions"], fields)
        frame = frame.copy()
        frame[list(scalar.expressions)] = scalar.frame(frame)
    elif kind == "filter":
        scalar = Scalar({"keep": spec["options"]["expression"]}, fields)
        frame = frame.loc[scalar.frame(frame)["keep"].eq(True)].copy()
    elif kind == "context":
        frame = context_frame(frame, spec["options"])
    elif kind != "project":
        raise ValueError("Unknown ordinary operation: " + kind)
    return {"__sql_rows": records(frame, spec["fields"])}


def install_relational_join():
    """Supply ordinary hash joins at DocETL's existing binary DAG boundary.

    Only operations bearing our explicit SQL-key contract take this path;
    native semantic equijoins retain their upstream implementation.
    """
    from docetl.operations.equijoin import EquijoinOperation
    from experiments.system_comparison.execution.operators import SQLJoin

    if getattr(EquijoinOperation.execute, "_comparison_sql", False):
        return
    original = EquijoinOperation.execute

    def execute(operation, left, right):
        spec = operation.config.get("comparison_sql_join")
        if spec is None:
            return original(operation, left, right)
        left_frame = pd.DataFrame(left, columns=list(spec["left_fields"]), dtype=object)
        right_frame = pd.DataFrame(right, columns=list(spec["right_fields"]), dtype=object)
        result = SQLJoin(spec["keys"]).frames(left_frame, right_frame)
        return records(result, {**spec["left_fields"], **spec["right_fields"]}), 0.0

    execute._comparison_sql = True
    EquijoinOperation.execute = execute
    from docetl.containers import StepBoundary

    original_boundary = StepBoundary.next

    def boundary_next(boundary, is_build=False, sample_size_needed=None):
        # A shared SQL CTE is one result within a pipeline execution. Native
        # binary scans can otherwise pull their preceding step a second time.
        # This is per-runner reuse only: no data crosses plans or queries.
        cache = getattr(boundary.runner, "_comparison_sql_steps", None)
        if cache is None:
            cache = boundary.runner._comparison_sql_steps = {}
        if not is_build and sample_size_needed is None and boundary.step_name in cache:
            return cache[boundary.step_name], 0.0, "Using shared SQL step\n"
        result = original_boundary(boundary, is_build, sample_size_needed)
        if not is_build and sample_size_needed is None:
            cache[boundary.step_name] = result[0]
        return result

    StepBoundary.next = boundary_next
    from docetl.operations.unnest import UnnestOperation

    original_unnest = UnnestOperation.execute

    def expand_rows(operation, rows):
        if not operation.config.get("comparison_sql_expand"):
            return original_unnest(operation, rows)
        # These envelopes contain an entire ordinary SQL relation. Native
        # unnest deep-copies its parent for every child, which would copy an
        # N-row relation N times. Only its child records are part of the SQL
        # output, so expand them directly in O(N).
        return [dict(row) for envelope in rows for row in envelope["__sql_rows"]], 0.0

    UnnestOperation.execute = expand_rows


def load_config(path):
    import yaml

    return yaml.safe_load(path.read_text())


def install_nullable_numbers():
    """Represent SQL nullable numbers faithfully in native structured output."""
    import sys
    from docetl.operations.utils import validation

    original_convert = validation.convert_val
    original_validate = validation._validate_value_against_schema

    def convert(value, model="gpt-4o-mini"):
        if value == "number?":
            return {"type": ["number", "null"]}
        return original_convert(value, model)

    def validate(value, schema):
        if schema.get("type") == ["number", "null"]:
            return value is None or original_validate(value, {"type": "number"})
        return original_validate(value, schema)

    for name, module in list(sys.modules.items()):
        if name.startswith("docetl.") and getattr(module, "convert_val", None) is original_convert:
            module.convert_val = convert
    validation._validate_value_against_schema = validate
