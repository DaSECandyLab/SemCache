"""Inspect semantic calls in SELECT scopes in dependency order."""

import re

import sqlglot
from sqlglot import exp


def _owned_anonymous(root):
    """Anonymous functions in one SELECT expression, excluding nested SELECTs."""
    out = []

    def walk(node):
        if isinstance(node, (exp.Subquery, exp.Select, exp.CTE)) and node is not root:
            return
        if isinstance(node, exp.Anonymous):
            out.append(node)
        for child in node.args.values():
            children = child if isinstance(child, list) else [child]
            for c in children:
                if isinstance(c, exp.Expression):
                    walk(c)

    walk(root)
    return out


def sql_functions(sql):
    """Extract semantic calls once, in dependency order, from every SELECT."""
    tree = sqlglot.parse_one(sql, read="duckdb") if isinstance(sql, str) else sql
    out = []
    seen = set()

    def visit(node):
        if isinstance(node, exp.Subquery):
            node = node.this
        if not isinstance(node, exp.Select) or id(node) in seen:
            return
        seen.add(id(node))
        with_ = node.args.get("with_")
        if with_:
            for cte in with_.expressions:
                visit(cte.this)
        frm = node.args.get("from_")
        if frm and isinstance(frm.this, exp.Subquery):
            visit(frm.this)
        for j in node.args.get("joins") or []:
            if isinstance(j.this, exp.Subquery):
                visit(j.this)
        roots = list(node.expressions)
        if node.args.get("where"):
            roots.append(node.args["where"])
        if node.args.get("having"):
            roots.append(node.args["having"])
        for j in node.args.get("joins") or []:
            if j.args.get("on"):
                roots.append(j.args["on"])
        nodes = []
        for root in roots:
            nodes.extend(_owned_anonymous(root))
        nodes = sorted(nodes, key=lambda x: x.meta.get("start", 0))
        for n in nodes:
            name = n.name.lower()
            if name not in {
                "semantic",
                "semantic_filter",
                "semantic_project",
                "semantic_join",
            }:
                continue
            a = n.expressions
            if not a:
                continue
            if name == "semantic_project":
                task = a[0].this
                vals = a[1:-1]
                fields = [f.name for f in a[-1].args["to"].expressions]
                kind = "semantic_project"
            else:
                task = a[0].this
                vals = a[1:]
                fields = []
                kind = "semantic_join" if name == "semantic_join" else "semantic_filter"
                if name == "semantic":
                    owner = n
                    while owner.parent is not None and not isinstance(
                        owner.parent, (exp.Select, exp.Join)
                    ):
                        owner = owner.parent
                    aliases = {
                        p.split(".")[0]
                        for p in re.findall(r"\{([^{}]+)\}", task)
                        if "." in p
                    }
                    if isinstance(owner.parent, exp.Join) and len(aliases) > 1:
                        kind = "semantic_join"
            out.append(
                {
                    "kind": kind,
                    "task": task,
                    "inputs": [x.sql(dialect="duckdb") for x in vals],
                    "fields": fields,
                }
            )

    visit(tree)
    return out
