"""Capture the complete input relation of a final deterministic SQL aggregate."""

import sqlglot
from sqlglot import Dialect, exp
from sqlglot.tokens import TokenType


def capture_plan(sql, table="__reference_aggregate_input"):
    # Mask native s'...' expressions using token spans, so escaped quotes and
    # comments never get rewritten as if they were semantic expressions.
    tokens = Dialect.get_or_raise("duckdb").tokenize(sql)
    replacements, spans = {}, []
    for left, right in zip(tokens, tokens[1:]):
        if (
            left.token_type == TokenType.VAR
            and left.text.lower() == "s"
            and right.token_type == TokenType.STRING
            and left.end + 1 == right.start
        ):
            marker = f"__reference_snapshot_semantic_{len(spans)}__"
            while marker in sql:
                marker += "_"
            replacements[marker] = sql[left.start : right.end + 1]
            spans.append((left.start, right.end + 1, marker))
    masked = sql
    for start, end, marker in reversed(spans):
        masked = masked[:start] + marker + masked[end:]
    tree = sqlglot.parse_one(masked, read="duckdb")
    if not isinstance(tree, exp.Select):
        return None
    aggregates = [
        a for projection in tree.expressions for a in projection.find_all(exp.AggFunc)
    ]
    if not aggregates:
        return None
    group = tree.args.get("group")
    keys = (
        {column.sql(dialect="duckdb") for column in group.expressions}
        if group
        else set()
    )
    for projection in tree.expressions:
        value = projection.this if isinstance(projection, exp.Alias) else projection
        if not (
            isinstance(value, (exp.Avg, exp.Sum, exp.Min, exp.Max, exp.Count))
            or isinstance(value, exp.Column)
            and value.sql(dialect="duckdb") in keys
        ):
            raise ValueError(
                "Aggregate capture requires aggregate outputs and explicit group columns"
            )
        if any(
            column.name in replacements for column in projection.find_all(exp.Column)
        ):
            raise ValueError(
                "Semantic aggregate arguments must be projected before the final aggregate"
            )
    if any(
        tree.args.get(key)
        for key in ("having", "qualify", "distinct", "limit", "offset")
    ):
        raise ValueError(
            "Aggregate capture does not support post-aggregate row filtering"
        )
    before = tree.copy()
    before.set("expressions", [exp.Star()])
    before.set("order", None)
    before.set("group", None)
    final = tree.copy()
    final.set("from_", exp.From(this=exp.to_table(table)))
    for key in ("where", "joins", "with_"):
        final.set(key, None)
    for column in final.find_all(exp.Column):
        for key in ("table", "db", "catalog"):
            column.set(key, None)

    def render(value):
        text = value.sql(dialect="duckdb")
        for marker, original in replacements.items():
            text = text.replace(marker, original)
        return text

    return {"input_sql": render(before), "aggregate_sql": render(final), "table": table}
