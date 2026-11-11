"""Inline linear source stages for read-only contract verification."""

from sqlglot import exp


def inline_ctes(tree):
    """Inline single-use linear stages, avoiding unsupported SemCache CTE branches.

    Used only for Python-origin linear pipelines. SQL-origin shared CTEs retain
    their original scopes and materialization boundaries.
    """
    definitions = {}

    def inline(node):
        if isinstance(node, exp.Table) and node.name in definitions:
            return exp.Subquery(
                this=definitions[node.name].copy(),
                alias=exp.TableAlias(this=exp.to_identifier(node.alias_or_name)),
            )
        return node

    for cte in tree.args["with_"].expressions:
        definitions[cte.alias_or_name] = cte.this.copy().transform(inline)
    result = tree.copy()
    result.set("with_", None)
    return result.transform(inline)
