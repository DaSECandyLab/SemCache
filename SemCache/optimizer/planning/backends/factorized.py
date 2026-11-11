"""Lower exact product samples inside DuckDB, carrying private leaf identities."""

from contextlib import ExitStack, contextmanager

from ..sampling.rows import CartesianRows
from ..state import digest


def fingerprint(rows):
    if isinstance(rows, CartesianRows):
        return ["product", fingerprint(rows.left), fingerprint(rows.right)]
    return [(row.identity, row.values, row.lineage) for row in rows]


@contextmanager
def relation_scope(backend, inputs, *, normalized=False):
    from planning.observations.evidence import freeze_evidence

    from ..sampling.rows import SampleRow
    from .execution import identifier, parse_json_cell

    leaves, tables, metadata = [], [], []
    with ExitStack() as stack:

        def compile_rows(rows):
            if isinstance(rows, CartesianRows):
                left, right = compile_rows(rows.left), compile_rows(rows.right)
                return (
                    "(SELECT sha256(semantic_sample_json(list_value(l.__sample_id,r.__sample_id))) __sample_id,"
                    "l.* EXCLUDE (__sample_id),r.* EXCLUDE (__sample_id) "
                    f"FROM {left} l CROSS JOIN {right} r)"
                )
            table = stack.enter_context(backend._table_scope((rows,), ([],)))[0]
            key = f"__factor_leaf_{len(leaves)}"
            index = {row.identity: row for row in rows}
            if normalized:
                entry = backend._input_table_names.get(table)
                typed = entry.normalized if entry is not None else None
                if typed is None:
                    typed = {}
                    # Read each physical source row once after DuckDB casts it.
                    # Cached immutable leaf tables retain this small typed bank.
                    for record in backend.connection.execute(
                        "SELECT semantic_sample_json(t) FROM " + table + " t"
                    ):
                        values = parse_json_cell(record[0])
                        identity = values.pop("__sample_id")
                        typed[identity] = SampleRow(
                            identity,
                            freeze_evidence(
                                {
                                    name: backend._value(name, value)
                                    for name, value in values.items()
                                }
                            ),
                            index[identity].lineage,
                        )
                    if entry is not None:
                        entry.normalized = typed
                index = typed
            leaves.append((key, index))
            tables.append(table)
            metadata.append(key)
            return f"(SELECT *,__sample_id AS {identifier(key)} FROM {table})"

        expressions = [compile_rows(rows) for rows in inputs]
        yield expressions, leaves, tables, metadata


def bound_sql(op, expressions, metadata=()):
    from .execution import identifier

    sql = (
        op.get("candidate_sql")
        if len(expressions) == 2 and op["kind"].lower() == "filter"
        else op.get("sql", op.get("sampling_sql"))
    )
    if not sql:
        raise ValueError("Relational sampling requires bound stage SQL")
    if len(expressions) == 1:
        sql = sql.replace("$INPUT", expressions[0])
        if (
            metadata
            and "AGGREGATE" not in op["kind"].upper()
            and not sql.startswith("SELECT *")
        ):
            prefix = (
                "SELECT DISTINCT " if sql.startswith("SELECT DISTINCT ") else "SELECT "
            )
            sql = sql.replace(
                prefix, prefix + ",".join(identifier(k) for k in metadata) + ",", 1
            )
    else:
        sql = sql.replace("$LEFT", expressions[0]).replace("$RIGHT", expressions[1])
    return sql


def execute(backend, op, inputs):
    from planning.observations.evidence import freeze_evidence

    from ..sampling.rows import SampleRow
    from .execution import identifier, parse_json_cell

    # These bound operators preserve every source field. Fetch their small
    # identity columns rather than serializing repeated text/compound values.
    source_only = op["kind"].upper() in {"FILTER", "ORDER_BY", "LIMIT"} and op.get(
        "sql", ""
    ).startswith("SELECT * FROM ")
    with relation_scope(backend, inputs, normalized=source_only) as (
        expressions,
        leaves,
        _,
        metadata,
    ):
        sql = bound_sql(op, expressions, metadata)
        identity_columns = (
            ["__sample_id"]
            if len(inputs) == 1
            else ["__left_sample_id", "__right_sample_id"]
        )
        returned_columns = identity_columns + metadata
        records = backend.connection.execute(
            (
                "SELECT " + ",".join(identifier(k) for k in returned_columns)
                if source_only
                else "SELECT semantic_sample_json(t)"
            )
            + " FROM ("
            + sql
            + ") t"
        )
        output = []
        aggregate = "AGGREGATE" in op["kind"].upper()
        for record in records:
            values = (
                dict(zip(returned_columns, record))
                if source_only
                else parse_json_cell(record[0])
            )
            chosen = {}
            origin = set()
            if not aggregate:
                for key, index in leaves:
                    identity = values.pop(key, None)
                    if identity not in index:
                        raise ValueError("Factorized SQL lost a source identity")
                    source = index[identity]
                    chosen[key] = source
                    origin.update(source.lineage)
            if source_only:
                source_values = {
                    key: value
                    for source in chosen.values()
                    for key, value in source.values.items()
                }
                source_values.update(values)
                values = source_values
            for key in list(values):
                if key.startswith("__factor_leaf_") or (
                    key.startswith("__semantic_input:")
                    and not op.get("_preserve_sample_inputs")
                ):
                    values.pop(key)
            if aggregate:
                identity = digest([op["operator_id"], values])
                values.pop("__sample_id", None)
            elif len(inputs) == 1:
                identity = values.pop("__sample_id", None)
            else:
                identity = digest(
                    [values.pop("__left_sample_id"), values.pop("__right_sample_id")]
                )
            row = SampleRow(
                identity,
                freeze_evidence(
                    {key: backend._value(key, value) for key, value in values.items()}
                ),
                tuple(sorted(origin, key=repr)),
            )
            if (
                not aggregate
                and len(inputs) == 1
                and isinstance(inputs[0], CartesianRows)
                and not inputs[0].contains(row)
            ):
                raise ValueError("Factorized SQL changed the sample identity tree")
            output.append(row)
        return output
