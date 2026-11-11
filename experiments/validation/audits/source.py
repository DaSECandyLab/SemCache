"""Decode actual source/API/SQL operations without trusting query metadata."""

from __future__ import annotations

import ast
import re

import sqlglot
from sqlglot import exp

from experiments.core.io import digest as digest
from experiments.core.paths import ROOT

QUERIES = ROOT / "benchmarks/queries"
SYSTEMS = ("semcache", "sema", "plop", "abacus", "nirvana", "lotus")
NATIVE_NAMES = {"semantic", "semantic_string", "audit_native"}


def source_operations(path):
    result = []
    for node in sorted(
        ast.walk(ast.parse(path.read_text())), key=lambda n: getattr(n, "lineno", 0)
    ):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "df"
            and node.func.attr.startswith("semantic_")
        ):
            continue
        kw = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
        result.append(
            {
                "method": node.func.attr,
                "task": kw["user_instruction"],
                "inputs": kw.get("input_columns", [kw.get("input_column")]),
                "outputs": kw.get("output_columns", []),
            }
        )
    return result


def named_task(task):
    refs = []

    def bind(match):
        refs.append(sqlglot.parse_one(match[1], read="duckdb"))
        return "{" + str(len(refs) - 1) + "}"

    numbered = re.sub(r"\{([^{}]+)\}", bind, task)
    return numbered, refs


def native_tree(text, system):
    if system == "sema":
        text = re.sub(
            r"\bs'((?:''|[^'])*)'",
            lambda m: (
                "audit_native("
                + exp.Literal.string(m[1].replace("''", "'")).sql(dialect="duckdb")
                + ")"
            ),
            text,
        )
    tree = sqlglot.parse_one(text, read="duckdb")
    if system == "semcache":
        return tree

    def decode(fn, field=None, output_type="VARCHAR"):
        task = fn.expressions[0].this.split("\nOutput protocol:", 1)[0]
        task, refs = named_task(task)
        args = [
            exp.Literal.string(task),
            *[exp.Cast(this=r, to=exp.DataType.build("VARCHAR")) for r in refs],
        ]
        if field:
            args.append(
                sqlglot.parse_one(
                    f'NULL::STRUCT("{field}" {output_type})', read="duckdb"
                )
            )
            return exp.Anonymous(this="semantic_project", expressions=args)
        owner = fn.parent
        while owner is not None and not isinstance(owner, (exp.Select, exp.Join)):
            owner = owner.parent
        kind = (
            "semantic_join"
            if isinstance(owner, exp.Join) and len({r.table for r in refs}) > 1
            else "semantic_filter"
        )
        return exp.Anonymous(this=kind, expressions=args)

    def visit(node):
        # Scalar string APIs wrap their output in a nullable one-field struct.
        if (
            isinstance(node, exp.Struct)
            or isinstance(node, exp.Anonymous)
            and node.name.casefold() == "struct_pack"
        ):
            fields = node.expressions
            if len(fields) == 1 and isinstance(fields[0], exp.PropertyEQ):
                value = fields[0].expression
                output_type = "VARCHAR"
                if isinstance(value, (exp.Cast, exp.TryCast)):
                    output_type = value.args["to"].sql(dialect="duckdb")
                    value = value.this
                if (
                    isinstance(value, exp.Nullif)
                    and isinstance(value.this, exp.Anonymous)
                    and value.this.name.casefold() in NATIVE_NAMES
                ):
                    if value.expression.this != "__SQL_NULL__":
                        raise ValueError("Changed NULL output protocol")
                    return decode(value.this, fields[0].this.name, output_type)
        if isinstance(node, exp.Anonymous) and node.name.casefold() in NATIVE_NAMES:
            return decode(node)
        return node

    return tree.transform(visit)


def source_tree(path):
    text = re.sub(r"^\s*(LOAD|CALL)\b[^;]*;", "", path.read_text(), flags=re.I | re.M)
    text = re.sub(
        r"\bs'((?:''|[^'])*)'",
        lambda m: (
            "SEMANTIC("
            + exp.Literal.string(m[1].replace("''", "'")).sql(dialect="duckdb")
            + ")"
        ),
        text,
    )
    statements = sqlglot.parse(text, read="duckdb")
    final = next(t.copy() for t in reversed(statements) if isinstance(t, exp.Select))
    ctes = [
        exp.CTE(
            this=s.expression.copy(),
            alias=exp.TableAlias(this=s.this.this.copy()),
            materialized=True,
        )
        for s in statements
        if isinstance(s, exp.Create)
    ]
    if ctes:
        existing = final.args.get("with_")
        final.set(
            "with_",
            exp.With(expressions=ctes + (existing.expressions if existing else [])),
        )

    def lower(node):
        if isinstance(node, exp.Anonymous) and node.name.casefold() == "semantic":
            task, refs = named_task(node.expressions[0].this)
            owner = node.parent
            while owner is not None and not isinstance(owner, (exp.Select, exp.Join)):
                owner = owner.parent
            kind = (
                "semantic_join"
                if isinstance(owner, exp.Join) and len({r.table for r in refs}) > 1
                else "semantic_filter"
            )
            return exp.Anonymous(
                this=kind,
                expressions=[
                    exp.Literal.string(task),
                    *[exp.Cast(this=r, to=exp.DataType.build("VARCHAR")) for r in refs],
                ],
            )
        return node

    return final.transform(lower)


def inline_appstore(tree):
    """Undo the disclosed PLOP input-binding CTE for relational comparison."""
    with_ = tree.args.get("with_")
    candidate = tree.args.get("from_")
    candidate = candidate.this if candidate is not None else None
    if (
        not with_
        and isinstance(candidate, exp.Subquery)
        and candidate.alias_or_name == "__appstore_candidates"
    ):
        tree = tree.copy()
        base = tree.args["from_"].this.this.copy()
        outer = tree.args["where"].this.copy()
        for column in outer.find_all(exp.Column):
            if column.name.casefold() == "translated_review":
                column.set("table", exp.to_identifier("ur"))
        base.set("expressions", [base.expressions[0]])
        base.set("where", exp.Where(this=exp.and_(base.args["where"].this, outer)))
        return base
    if (
        not with_
        or len(with_.expressions) != 1
        or with_.expressions[0].alias_or_name != "__appstore_candidates"
    ):
        return tree
    base = with_.expressions[0].this.copy()
    outer = tree.args["where"].this.copy()

    def qualify(node):
        if isinstance(node, exp.Column) and node.name.casefold() == "translated_review":
            node.set("table", exp.to_identifier("ur"))
        return node

    outer = outer.transform(qualify)
    base.set("expressions", [base.expressions[0]])
    base.set("where", exp.Where(this=exp.and_(base.args["where"].this, outer)))
    return base


def canonical_sql(tree):
    from experiments.validation.workload.cte import inline_single_use

    # CTE aliases and their single-use derived-table form are equivalent.
    # The helper retains shared and explicitly materialized relations.
    normalized, _ = inline_single_use(tree.sql(dialect="duckdb"))
    tree = sqlglot.parse_one(normalized, read="duckdb")
    appstore = {t.name.casefold() for t in tree.find_all(exp.Table)} == {
        "user_reviews",
        "playstore",
    }
    for node in tree.walk():
        node.comments = None
        if isinstance(node, exp.Identifier):
            node.set("this", node.this.casefold())
            node.set("quoted", False)
        if (
            appstore
            and isinstance(node, exp.Column)
            and not node.table
            and node.name.casefold() == "translated_review"
        ):
            node.set("table", exp.to_identifier("ur"))

    # Conjunction order changes in PLOP's binding CTE; scopes stay intact.
    def conjunction(node):
        if isinstance(node, exp.Paren) and isinstance(node.this, exp.And):
            return conjunction(node.this)
        if isinstance(node, exp.And):
            terms = []

            def flatten(term):
                if isinstance(term, exp.Paren) and isinstance(term.this, exp.And):
                    flatten(term.this)
                elif isinstance(term, exp.And):
                    flatten(term.this)
                    flatten(term.expression)
                else:
                    terms.append(term)

            flatten(node)
            return exp.and_(
                *[
                    t.copy()
                    for t in sorted(terms, key=lambda t: t.sql(dialect="duckdb"))
                ]
            )
        return node

    tree = tree.transform(conjunction)
    return sqlglot.parse_one(
        tree.sql(dialect="duckdb", comments=False), read="duckdb"
    ).sql(dialect="duckdb")


def sql_operations(tree):
    result = []
    for cte in tree.args["with_"].expressions:
        for node in cte.find_all(exp.Anonymous):
            if node.name.casefold() not in {"semantic_filter", "semantic_project"}:
                continue
            task = node.expressions[0].this
            inputs = (
                [
                    c.name
                    for v in node.expressions[1:-1]
                    if node.name.casefold() == "semantic_project"
                    for c in v.find_all(exp.Column)
                ]
                if node.name.casefold() == "semantic_project"
                else [
                    c.name for v in node.expressions[1:] for c in v.find_all(exp.Column)
                ]
            )
            method = (
                "semantic_map"
                if node.name.casefold() == "semantic_project"
                else "semantic_filter"
            )
            outputs = (
                [f.name for f in node.expressions[-1].args["to"].expressions]
                if method == "semantic_map"
                else []
            )
            if cte.alias_or_name == "__reduce_output":
                collect = next(
                    c
                    for c in tree.args["with_"].expressions
                    if c.alias_or_name == "__reduce_input"
                )
                lists = [
                    n for n in collect.walk() if isinstance(n, (exp.ArrayAgg, exp.List))
                ]
                if len(lists) != 1 or lists[0].args.get("distinct"):
                    raise ValueError("Reduce bridge must retain the complete row bag")
                inputs = [c.name for c in lists[0].find_all(exp.Column)]
                method, outputs = "semantic_reduce", []
                if "values as a DuckDB list literal" not in task:
                    raise ValueError(
                        "Reduce input serialization is not declared accurately"
                    )
            result.append(
                {
                    "method": method,
                    "task": task.split("\nInputs:", 1)[0],
                    "inputs": inputs,
                    "outputs": outputs,
                }
            )
    return result


def python_operations(path, system):
    result = []
    tree = ast.parse(path.read_text())

    def literal(node):
        return ast.literal_eval(node)

    for call in sorted(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        ),
        key=lambda n: n.lineno,
    ):
        method = call.func.attr
        if method not in {
            "semantic_map",
            "semantic_filter",
            "semantic_reduce",
            "sem_map",
            "sem_filter",
            "sem_agg",
        }:
            continue
        kw = {k.arg: k.value for k in call.keywords}
        if system == "nirvana":
            result.append(
                {
                    "method": method,
                    "task": literal(kw["user_instruction"]),
                    "inputs": literal(
                        kw.get(
                            "input_columns",
                            ast.List(elts=[kw.get("input_column")], ctx=ast.Load()),
                        )
                    ),
                    "outputs": literal(kw["output_columns"])
                    if "output_columns" in kw
                    else [],
                }
            )
        elif system == "lotus":
            task = literal(call.args[0])
            inputs = re.findall(r"\{([^{}]+)\}", task)
            if not inputs:
                raise ValueError(
                    "Lotus semantic operation does not bind any input columns"
                )
            result.append(
                {
                    "method": {
                        "sem_map": "semantic_map",
                        "sem_filter": "semantic_filter",
                        "sem_agg": "semantic_reduce",
                    }[method],
                    "task": task.split("\nInputs:", 1)[0],
                    "inputs": inputs,
                    "outputs": [literal(kw["suffix"])] if method == "sem_map" else [],
                }
            )
        else:
            task = (
                literal(kw["desc"])
                if method == "sem_map"
                else literal(kw["agg"])
                if method == "sem_agg"
                else literal(call.args[0])
            )
            outputs = []
            if method == "sem_map":
                for field in kw["cols"].elts:
                    values = dict(zip((literal(k) for k in field.keys), field.values))
                    outputs.append(literal(values["name"]))
                    if literal(values["desc"]) != task:
                        raise ValueError("Abacus output-field instruction changed")
            result.append(
                {
                    "method": {
                        "sem_map": "semantic_map",
                        "sem_filter": "semantic_filter",
                        "sem_agg": "semantic_reduce",
                    }[method],
                    "task": task,
                    "inputs": literal(kw["depends_on"]),
                    "outputs": outputs,
                }
            )
    return result
