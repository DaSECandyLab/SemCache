"""Read-only source decoders and SQL expectations for independent query audits."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import sqlglot
from sqlglot import exp

from experiments.core.paths import ROOT

QUERY_ROOT = ROOT / "benchmarks/queries"
SYSTEMS = ("semcache", "sema", "plop", "abacus", "nirvana", "lotus")
CONTRACT = "selected19-source-faithful-v3"


def ident(name):
    return '"' + name.replace('"', '""') + '"'


def literal(value):
    return exp.Literal.string(value).sql(dialect="duckdb")


def py_operations(path):
    operations = []
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "df"
            and node.func.attr.startswith("semantic_")
        ):
            if node.args or any(k.arg is None for k in node.keywords):
                raise ValueError("Dynamic source operation is not supported")
            operations.append(
                {
                    "method": node.func.attr,
                    "line": node.lineno,
                    "kwargs": {k.arg: ast.literal_eval(k.value) for k in node.keywords},
                }
            )
    return sorted(operations, key=lambda op: op["line"])


def sql_operations(sql):
    operations = []
    tree = sqlglot.parse_one(sql, read="duckdb")
    for node in tree.find_all(exp.Anonymous):
        name = node.name.casefold()
        if name not in {"semantic_filter", "semantic_join", "semantic_project"}:
            continue
        args = node.expressions
        task = args[0].this
        if name == "semantic_project":
            values = args[1:-1]
            schema = args[-1].args.get("to")
            outputs = [field.name for field in schema.expressions] if schema else []
            method = "semantic_map"
        else:
            values = args[1:]
            outputs = []
            method = "semantic_join" if name == "semantic_join" else "semantic_filter"
        operations.append(
            {
                "method": method,
                "kwargs": {
                    "user_instruction": task,
                    "input_columns": [
                        column.sql(dialect="duckdb")
                        for value in values
                        for column in value.find_all(exp.Column)
                    ],
                    **({"output_columns": outputs} if outputs else {}),
                },
            }
        )
    return operations


def source_relational_sql(query):
    """Lower original Sema/PLOP syntax, preserving all relational scopes."""
    text = (QUERY_ROOT / query["original"]).read_text()
    # This is syntax recognition, not data parsing. Handle escaped SQL quotes.
    text = re.sub(
        r"\bs'((?:''|[^'])*)'",
        lambda m: "SEMANTIC(" + literal(m[1].replace("''", "'")) + ")",
        text,
    )
    trees = sqlglot.parse(text, read="duckdb")
    final = next(t.copy() for t in reversed(trees) if isinstance(t, exp.Select))
    initial = [
        exp.CTE(
            this=t.expression.copy(),
            alias=exp.TableAlias(this=exp.to_identifier(t.this.name)),
            materialized=True,
        )
        for t in trees
        if isinstance(t, exp.Create)
    ]
    if initial:
        existing = final.args.get("with_")
        final.set(
            "with_",
            exp.With(expressions=initial + (existing.expressions if existing else [])),
        )

    def lower(node):
        node.comments = None
        if not isinstance(node, exp.Anonymous) or node.name.casefold() != "semantic":
            return node
        task = node.expressions[0].this
        inputs = []

        def bind(match):
            column = sqlglot.parse_one(match[1], read="duckdb")
            if not isinstance(column, exp.Column):
                raise ValueError("Source template must bind a column")
            inputs.append(column)
            return "{" + str(len(inputs) - 1) + "}"

        prompt = re.sub(r"\{([^{}]+)\}", bind, task)
        owner = node.parent
        while owner is not None and not isinstance(owner, (exp.Join, exp.Select)):
            owner = owner.parent
        binary = (
            isinstance(owner, exp.Join)
            and len({c.table.casefold() for c in inputs}) > 1
        )
        return exp.Anonymous(
            this="semantic_join" if binary else "semantic_filter",
            expressions=[
                exp.Literal.string(prompt),
                *[exp.Cast(this=c, to=exp.DataType.build("VARCHAR")) for c in inputs],
            ],
        )

    return (
        final.transform(lower).sql(dialect="duckdb", pretty=True, comments=False)
        + ";\n"
    )


def reduction_input_sql(column, relation):
    # LIST preserves duplicates, NULL and empty strings; the native list literal needs no JSON extension.
    return (
        f"SELECT CAST(COALESCE(list({ident(column)}), []) AS VARCHAR) AS reduce_values, "
        f"COUNT(*) AS reduce_count FROM {ident(relation)}"
    )


def linear_sql(query, *, operations=None, aggregate=None, numeric_columns=()):
    """Rebuild Python source operations as readable ordered CTEs."""
    operations = (
        py_operations(QUERY_ROOT / query["original"])
        if operations is None
        else operations
    )
    ctes = [f"stage_0 AS (SELECT * FROM nirvana_{query['dataset']})"]
    previous = "stage_0"
    reduction = None
    for i, op in enumerate(operations, 1):
        method, args = op["method"], op["kwargs"]
        if method == "semantic_reduce":
            if i != len(operations):
                raise ValueError("Only a terminal Reduce is supported")
            reduction = args
            break
        inputs = args["input_columns"]
        task = (
            args["user_instruction"]
            + "\nInputs: "
            + ", ".join(f"{c}: {{{j}}}" for j, c in enumerate(inputs))
        )
        values = ", ".join(f"CAST({ident(c)} AS VARCHAR)" for c in inputs)
        stage = f"stage_{i}"
        if method == "semantic_map":
            schema = ", ".join(
                f"{ident(c)} {'DOUBLE' if c in numeric_columns else 'VARCHAR'}"
                for c in args["output_columns"]
            )
            ctes.append(
                f"project_{i} AS (SELECT *, semantic_project({literal(task)}, {values}, NULL::STRUCT({schema})) AS __mapped FROM {previous})"
            )
            fields = ", ".join(
                f"__mapped.{ident(c)} AS {ident(c)}" for c in args["output_columns"]
            )
            ctes.append(
                f"{stage} AS (SELECT * EXCLUDE (__mapped), {fields} FROM project_{i})"
            )
        elif method == "semantic_filter":
            ctes.append(
                f"{stage} AS (SELECT * FROM {previous} WHERE semantic_filter({literal(task)}, {values}))"
            )
        else:
            raise ValueError(method)
        previous = stage
    prefix = "WITH\n" + ",\n".join(ctes)
    rows_sql = prefix + f"\nSELECT * FROM {previous}"
    if aggregate:
        from experiments.validation.faithful_runtime import aggregate_sql

        return (
            prefix + "\n" + aggregate_sql(aggregate, previous) + ";\n",
            operations,
            None,
        )
    if not reduction:
        return rows_sql + ";\n", operations, None
    ctes.append(
        "__reduce_input AS ("
        + reduction_input_sql(reduction["input_column"], previous)
        + ")"
    )
    task = (
        reduction["user_instruction"]
        + "\nInputs: "
        + reduction["input_column"]
        + " values as a DuckDB list literal (one element per row, including duplicates and NULL; text values are quoted): {0}"
    )
    ctes.append(
        "__reduce_output AS (SELECT semantic_project("
        + literal(task)
        + ", CAST(reduce_values AS VARCHAR), NULL::STRUCT(result VARCHAR)) AS __reduced "
        + "FROM __reduce_input WHERE reduce_count > 0)"
    )
    # Nirvana's source Reduce explicitly returns this sentinel on zero rows.
    # Preserve it instead of silently returning mathematical AVG(NULL)/COUNT(0).
    result = (
        "WITH\n"
        + ",\n".join(ctes)
        + "\nSELECT __reduced.result AS result FROM __reduce_output\nUNION ALL\n"
        + "SELECT 'No data to process.' AS result FROM __reduce_input WHERE reduce_count = 0;\n"
    )
    return result, operations, reduction
