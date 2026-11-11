"""Bounded startup probes; generated training queries still read full tables."""

from planning.backends.execution import identifier, quote

from training.source_data import CORPORA

FILTER_TASK = "Does text {0} describe a concrete action, activity, or event? Metadata {1} only provides context."
PROJECTION_TASK = "Return flag=true if text {0} is written from a first-person perspective; otherwise flag=false. Metadata: {1}."
JOIN_TASK = "Do texts {0} and {2} describe a shared concrete activity, difficulty, or purpose? {1} and {3} are metadata, not sufficient evidence alone."


def calibration_table(dataset, connection):
    """Use rowid on a table or a verified SELECT-* alias of that table."""
    from sqlglot import exp, parse_one

    table = CORPORA[dataset][0]

    def physical(name):
        return bool(
            connection.execute(
                "SELECT 1 FROM duckdb_tables() WHERE schema_name='main' "
                "AND table_name=" + quote(name)
            )
        )

    if physical(table):
        return table
    views = connection.execute(
        "SELECT sql FROM duckdb_views() WHERE schema_name='main' AND view_name="
        + quote(table)
    )
    if views:
        select = parse_one(views[0][0], read="duckdb").expression
        source = select.args.get("from_") if isinstance(select, exp.Select) else None
        underlying = source.this if source is not None else None
        if (
            isinstance(select, exp.Select)
            and len(select.expressions) == 1
            and isinstance(select.expressions[0], exp.Star)
            and not any(
                value
                for key, value in select.args.items()
                if key not in {"expressions", "from_"}
            )
            and isinstance(underlying, exp.Table)
            and not underlying.db
            and not underlying.catalog
            and physical(underlying.name)
        ):
            return underlying.name
    raise ValueError(
        f"Calibration requires a physical table or a plain table alias: {table}"
    )


def calibration_input_sql(dataset, limit, seed, *, source_table=None):
    """Select a bounded calibration cohort; workload inputs remain full tables."""
    if type(limit) is not int or limit < 1:
        raise ValueError("Calibration sample limit must be a positive integer")
    table, text, aux, _ = CORPORA[dataset]
    table = source_table or table
    return f"""SELECT rowid::BIGINT AS doc_id,
        CAST({text} AS VARCHAR) AS text, CAST({aux} AS VARCHAR) AS aux,
        CAST({aux} AS VARCHAR) AS pair_bucket
        FROM {identifier(table)} ORDER BY hash(rowid,{int(seed)}),rowid LIMIT {limit}"""
