"""Expose adjacent filters as one conjunction without crossing SQL boundaries."""

from sqlglot import exp


def merge_filter_stages(tree):
    def simple_filter(node):
        return (
            isinstance(node, exp.Select)
            and node.args.get("where") is not None
            and len(node.expressions) == 1
            and isinstance(node.expressions[0], exp.Star)
            and not any(node.expressions[0].args.values())
            and all(
                not value
                for key, value in node.args.items()
                if key not in {"expressions", "from_", "where"}
            )
        )

    def merge(node):
        if not simple_filter(node):
            return node
        while isinstance(node.args["from_"].this, exp.Subquery):
            source = node.args["from_"].this
            inner = source.this
            if not simple_filter(inner):
                break
            old_alias = source.alias_or_name
            new_alias = inner.args["from_"].this.alias_or_name

            def rebind(column):
                if isinstance(column, exp.Column) and column.table == old_alias:
                    column.set(
                        "table", exp.to_identifier(new_alias) if new_alias else None
                    )
                return column

            condition = node.args["where"].this.copy().transform(rebind)
            node.set(
                "where",
                exp.Where(this=exp.and_(inner.args["where"].this.copy(), condition)),
            )
            node.set("from_", inner.args["from_"].copy())
        return node

    return tree.copy().transform(merge)
