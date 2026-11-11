"""Inline singly referenced CTEs without duplicating shared computations."""

from __future__ import annotations

from collections import Counter
import re

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import Scope, build_scope


def invariants(tree):
    """Work and row-set boundaries must survive an alias-only rewrite."""
    root = build_scope(tree)
    base_tables = Counter(
        source.name.casefold()
        for scope in root.traverse()
        for _, source in scope.selected_sources.values()
        if isinstance(source, exp.Table)
    ) if root else Counter()
    return {
        "semantic_tasks": sorted(
            (node.name.casefold(), node.expressions[0].this)
            for node in tree.find_all(exp.Anonymous)
            if node.name.casefold().startswith(("semantic", "__cte_semantic_literal"))
        ),
        "tables": dict(base_tables),
        "limits": sorted(node.sql(dialect="duckdb") for node in tree.find_all(exp.Limit)),
        "orders": sorted(node.sql(dialect="duckdb") for node in tree.find_all(exp.Order)),
        "distincts": sum(1 for _ in tree.find_all(exp.Distinct)),
        "joins": sum(1 for _ in tree.find_all(exp.Join)),
    }


def inline_single_use(sql: str, *, system: str | None = None) -> tuple[str, dict]:
    encoded = sql
    if system == "sema":
        # Sema's s'...' syntax is not a DuckDB/SQLGlot string literal. Wrap
        # it temporarily without touching its field placeholders or content.
        encoded = re.sub(r"\bs'((?:''|[^'])*)'", lambda m: "__cte_semantic_literal('" + m[1] + "')", sql)
    tree = sqlglot.parse_one(encoded, read="duckdb")
    before = invariants(tree)
    original_ctes = list(tree.find_all(exp.CTE))
    removed = []
    while True:
        root = build_scope(tree)
        if root is None:
            break
        counts = root.ref_count()
        scopes = list(root.traverse())
        chosen = None
        for scope in scopes:
            if not scope.is_cte or counts[id(scope)] != 1:
                continue
            cte = scope.expression.parent
            if not isinstance(cte, exp.CTE) or cte.args.get("materialized") is not None:
                continue
            if cte.parent.args.get("recursive"):
                continue
            references = [
                node
                for owner in scopes
                for node, source in owner.selected_sources.values()
                if isinstance(source, Scope) and source is scope and isinstance(node, exp.Table)
            ]
            if len(references) == 1:
                chosen = (cte, references[0])
                break
        if chosen is None:
            break
        cte, reference = chosen
        alias = reference.args.get("alias")
        alias = alias.copy() if alias else exp.TableAlias(this=exp.to_identifier(reference.name, quoted=True))
        if cte.alias_column_names and not alias.args.get("columns"):
            alias.set("columns", [node.copy() for node in cte.args["alias"].args["columns"]])
        reference.replace(exp.Subquery(this=cte.this.copy(), alias=alias))
        removed.append(cte.alias)
        with_clause = cte.parent
        cte.pop()
        if not with_clause.expressions:
            with_clause.pop()
    after = invariants(tree)
    if before != after:
        raise ValueError("CTE inlining changed semantic work or SQL row-set boundaries")
    root = build_scope(tree)
    counts = root.ref_count() if root else {}
    kept = [
        {"name": scope.expression.parent.alias,
         "references": counts[id(scope)],
         "materialized": scope.expression.parent.args.get("materialized")}
        for scope in root.traverse() if scope.is_cte
    ] if root else []
    report = {"before": len(original_ctes), "after": len(kept), "inlined": removed, "retained": kept, "invariants": after}
    result = tree.sql(dialect="duckdb", pretty=True) + ";\n" if removed else sql
    if removed and system == "sema":
        result = re.sub(r"__CTE_SEMANTIC_LITERAL\(\s*'((?:''|[^'])*)'\s*\)", lambda m: "s'" + m[1] + "'", result, flags=re.I)
        if "__cte_semantic_literal(" in result.lower():
            raise ValueError("Unrestored Sema semantic literal")
    return result, report
