"""Shared, portable mathematical contracts for the adapted selected19 workload.

SQL engines execute the expressions directly. Python adapters evaluate the same
AST expression in ordinary DuckDB; they do not reimplement numeric parsing.
This intentionally specifies mathematical aggregation, not native LLM Reduce.
"""

from __future__ import annotations

import ast
import threading
from functools import lru_cache
from pathlib import Path

from experiments.core.sql import identifier, literal

CONTRACT_VERSION = "selected19-math-v2"


def _let(name, value, body):
    # Portable scalar binding supported by all three SQL engines. No subquery
    # or user-defined SQL function needs to cross a native semantic boundary.
    return f"list_extract(list_transform([{value}], {name} -> {body}), 1)"


def price_expression(column="Price"):
    number = r"[+-]?(?:[0-9][0-9,]*(?:\.[0-9]+)?|\.[0-9]+)(?:e[+-]?[0-9]+)?"
    valid = r"[+-]?(?:(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?|\.[0-9]+)(?:e[+-]?[0-9]+)?"
    units = r"(billion|bn|million|mn|thousand|b|m|k)\b"
    token = (
        r"(?:₦\s*|ngn\s*|naira\s*|\bn\s*)?"
        + number
        + r"(?:\s*"
        + units
        + r")?(?:\s*(?:naira|ngn)\b)?"
    )
    marked = r"₦|ngn|naira|^n\s*[+\.0-9-]|" + units
    foreign = r"[$€£¥]|\b(usd|eur|gbp|dollars?|euros?|pounds?|yen)\b"
    ranges = (
        r"\b(or|to|between)\s*(₦|ngn|n)?\s*[0-9]|[0-9kmb]\s*[-–—/]\s*(₦|ngn|n)?\s*[0-9]"
    )
    amount = f"regexp_extract(price_token, {literal(number)}, 0)"
    unit = f"lower(regexp_extract(price_token, {literal(units)}, 1))"
    multiplier = (
        f"CASE {unit} WHEN 'billion' THEN 1e9 WHEN 'bn' THEN 1e9 WHEN 'b' THEN 1e9 "
        "WHEN 'million' THEN 1e6 WHEN 'mn' THEN 1e6 WHEN 'm' THEN 1e6 "
        "WHEN 'thousand' THEN 1e3 WHEN 'k' THEN 1e3 ELSE 1 END"
    )
    parsed = (
        f"CASE WHEN regexp_full_match({amount}, {literal(valid)}) "
        f"THEN TRY_CAST(replace({amount}, ',', '') AS DOUBLE) * ({multiplier}) ELSE NULL END"
    )
    numbers = f"list_transform(price_selected, price_token -> {parsed})"
    result = (
        "CASE WHEN len(price_numbers) > 0 AND "
        "len(list_filter(price_numbers, price_value -> price_value IS NULL "
        "OR NOT isfinite(price_value) OR price_value < 0)) = 0 "
        "AND list_unique(price_numbers) = 1 THEN list_min(price_numbers) ELSE NULL END"
    )
    selected = (
        f"CASE WHEN len(price_marked) > 0 THEN price_marked "
        f"WHEN regexp_full_match(price_text, {literal(valid)}) THEN price_tokens ELSE [] END"
    )
    body = _let(
        "price_tokens",
        f"regexp_extract_all(price_text, {literal(token)})",
        _let(
            "price_marked",
            f"list_filter(price_tokens, price_token -> regexp_matches(price_token, {literal(marked)}))",
            _let("price_selected", selected, _let("price_numbers", numbers, result)),
        ),
    )
    body = (
        f"CASE WHEN price_text IS NULL OR regexp_matches(price_text, {literal(foreign)}) "
        f"OR regexp_matches(price_text, {literal(ranges)}) THEN NULL ELSE {body} END"
    )
    return _let(
        "price_text", f"lower(trim(CAST({identifier(column)} AS VARCHAR)))", body
    )


def runtime_expression(column="Runtime"):
    number = r"[0-9]+(?:\.[0-9]+)?"
    minutes = number + r"\s*(m|min|mins|minute|minutes)?"
    hours = (
        "("
        + number
        + r")\s*(?:hours?|hrs?|h)(?:\s*("
        + number
        + r")\s*(?:minutes?|mins?|m))?"
    )
    result = (
        f"CASE WHEN regexp_full_match(runtime_text, {literal(minutes)}) "
        f"THEN TRY_CAST(regexp_extract(runtime_text, {literal(number)}, 0) AS DOUBLE) "
        f"WHEN regexp_full_match(runtime_text, {literal(hours)}) THEN "
        f"TRY_CAST(regexp_extract(runtime_text, {literal(hours)}, 1) AS DOUBLE) * 60 "
        f"+ COALESCE(TRY_CAST(NULLIF(regexp_extract(runtime_text, {literal(hours)}, 2), '') AS DOUBLE), 0) "
        "ELSE NULL END"
    )
    return _let(
        "runtime_text", f"lower(trim(CAST({identifier(column)} AS VARCHAR)))", result
    )


def numeric_expression(column):
    if column.casefold() == "price":
        return price_expression(column)
    if column.casefold() == "runtime":
        return runtime_expression(column)
    value = f"TRY_CAST(trim(CAST({identifier(column)} AS VARCHAR)) AS DOUBLE)"
    return f"CASE WHEN isfinite({value}) THEN {value} ELSE NULL END"


def aggregate_spec(sql):
    """Read the actual outer aggregate, never a regex prefix or nested query."""
    import sqlglot
    from sqlglot import exp

    tree = sqlglot.parse_one(sql, read="duckdb")
    if not isinstance(tree, exp.Select):
        raise ValueError("A selected query must be a SELECT")
    if len(tree.expressions) != 1:
        return None
    output = tree.expressions[0]
    aggregate = output.this if isinstance(output, exp.Alias) else output
    if not isinstance(aggregate, exp.AggFunc):
        return None
    function = aggregate.sql_name()
    if function not in {"AVG", "MIN", "MAX", "COUNT"}:
        raise ValueError("Unsupported common aggregate")
    if aggregate.args.get("distinct") or isinstance(aggregate.this, exp.Distinct):
        raise ValueError("DISTINCT aggregates need an explicit contract")
    if function == "COUNT":
        if not isinstance(aggregate.this, exp.Star):
            raise ValueError("The selected COUNT contract is COUNT(*)")
        column = None
    else:
        columns = {c.name for c in aggregate.this.find_all(exp.Column)}
        if len(columns) != 1:
            raise ValueError("A numeric aggregate must reference one input column")
        column = next(iter(columns))
    return {
        "version": CONTRACT_VERSION,
        "function": function,
        "column": column,
        "alias": output.alias_or_name or "result",
        "expression": aggregate.sql(dialect="duckdb"),
        "value_expression": aggregate.this.sql(dialect="duckdb") if column else None,
    }


def verify_linear_translation(sql, original, dataset):
    """Reject stale native lineages instead of executing a different SQL task."""
    import sqlglot
    from sqlglot import exp

    expected = []
    for call in ast.walk(ast.parse(Path(original).read_text())):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "df"
            and call.func.attr in {"semantic_map", "semantic_filter"}
        ):
            kw = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
            expected.append(
                (
                    call.lineno,
                    {
                        "kind": "semantic_project"
                        if call.func.attr == "semantic_map"
                        else "semantic_filter",
                        "task": kw["user_instruction"],
                        "inputs": kw["input_columns"],
                        "outputs": kw.get("output_columns", []),
                    },
                )
            )
    expected = [s for _, s in sorted(expected)]
    tree = sqlglot.parse_one(sql, read="duckdb")
    actual = []

    def visit(select):
        source = select.args.get("from_")
        if source and isinstance(source.this, exp.Subquery):
            visit(source.this.this)
        for key in (
            "joins",
            "with_",
            "distinct",
            "group",
            "having",
            "qualify",
            "limit",
            "offset",
            "order",
        ):
            if select.args.get(key):
                raise ValueError(
                    "Canonical linear translation contains unsupported relational stages"
                )
        where = select.args.get("where")
        if where and not (
            isinstance(where.this, exp.Anonymous)
            and where.this.name.lower() == "semantic_filter"
        ):
            raise ValueError("Canonical linear filter differs from original lineage")
        expressions = list(select.expressions) + ([where] if where else [])
        for expression in expressions:
            for fn in expression.find_all(exp.Anonymous):
                kind = fn.name.lower()
                if kind not in {"semantic_filter", "semantic_project"}:
                    continue
                args = fn.expressions
                values = args[1:-1] if kind == "semantic_project" else args[1:]
                inputs = []
                for value in values:
                    if not isinstance(value, exp.Cast) or not isinstance(
                        value.this, exp.Column
                    ):
                        raise ValueError(
                            "Canonical lineage requires explicit column inputs"
                        )
                    inputs.append(value.this.name)
                outputs = (
                    [f.name for f in args[-1].args["to"].expressions]
                    if kind == "semantic_project"
                    else []
                )
                task = args[0].this
                instruction, sep, labels = task.partition("\nInputs:")
                if not sep or labels.strip() != ", ".join(
                    f"{name}: {{{i}}}" for i, name in enumerate(inputs)
                ):
                    raise ValueError(
                        "Canonical input labels differ from column bindings"
                    )
                actual.append(
                    {
                        "kind": kind,
                        "task": instruction,
                        "inputs": inputs,
                        "outputs": outputs,
                    }
                )

    visit(tree)
    if expected != actual or {t.name for t in tree.find_all(exp.Table)} != {
        "nirvana_" + dataset
    }:
        raise ValueError("Canonical SQL stages differ from original native lineage")
    return actual


_local = threading.local()


@lru_cache(maxsize=8192)
def _evaluate_numeric(expression, column, value):
    import duckdb

    if not hasattr(_local, "connection"):
        _local.connection = duckdb.connect()
        _local.connection.execute("SET threads=1")
    return _local.connection.execute(
        f"SELECT {expression} FROM (SELECT ?::VARCHAR AS {identifier(column)})", [value]
    ).fetchone()[0]


def numeric_value(value, spec):
    column = spec["column"]
    expression = spec.get("value_expression") or numeric_expression(column)
    return _evaluate_numeric(expression, column, None if value is None else str(value))


def aggregate_frame(frame, spec):
    """Run the exact SQL aggregate, including SQL NULL and empty-input behavior."""
    import duckdb

    with duckdb.connect() as con:
        con.register("aggregate_input", frame)
        return con.execute(
            f"SELECT {spec['expression']} FROM aggregate_input"
        ).fetchone()[0]
