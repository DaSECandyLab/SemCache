"""Runners for the common numeric workload and explicit source translations.

The generated files keep query construction separate from model routing. This
module deliberately does not invent a reference answer or numeric parser.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from experiments.core.resources import CONCURRENCY
from experiments.core.sql import identifier as quote


def aggregate_sql(aggregate, relation):
    """One shared finite-double, bag-preserving mathematical contract."""
    function = aggregate["function"]
    if function not in {"AVG", "MIN", "MAX", "COUNT", "SUM"}:
        raise ValueError("Unsupported deterministic aggregate")
    if function == "COUNT":
        argument = "*"
    else:
        value = f"TRY_CAST({quote(aggregate['column'])} AS DOUBLE)"
        argument = f"CASE WHEN isfinite({value}) THEN {value} ELSE NULL END"
    return f"SELECT {function}({argument}) AS result FROM {quote(relation)}"


def aggregate_frame(frame, aggregate):
    """Ordinary relational host aggregation; it never calls an LLM or parses units."""
    import duckdb
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(
            [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in frame]
        )
    # Native engines can return no records and no schema after a false filter.
    if frame.empty:
        return pd.DataFrame(
            {"result": [0 if aggregate["function"] == "COUNT" else None]}
        )
    with duckdb.connect(config={"threads": CONCURRENCY}) as con:
        con.register("common_rows", frame)
        return con.execute(aggregate_sql(aggregate, "common_rows")).fetchdf()


def run_nirvana(dataframe, job, *, aggregate=None):
    if job.get("native_mechanisms"):
        from experiments.system_comparison.backends.nirvana.optimizer import (
            optimize_and_execute,
        )

        output, cost, runtime, native_timing = optimize_and_execute(dataframe, job)
    else:
        from nirvana.optim import OptimizeConfig

        output, cost, runtime = dataframe.optimize_and_execute(
            optim_config=OptimizeConfig(**job.get("optimizer", {}))
        )
        native_timing = {}
    host_seconds = None
    if aggregate:
        from experiments.system_comparison.reporting.timing import stage

        started = time.monotonic()
        with stage("host", "shared deterministic aggregate"):
            output = aggregate_frame(output, aggregate)
        host_seconds = time.monotonic() - started
        runtime += host_seconds
    output.attrs["native_timing"] = native_timing
    if host_seconds is not None:
        output.attrs["native_timing"]["host_seconds"] = host_seconds
    return output, cost, runtime


def run_abacus(dataset, job, *, reduction=False, aggregate=None):
    from systems.abacus.validate import native_config

    if job.get("native_mechanisms"):
        from experiments.system_comparison.backends.abacus.optimizer import (
            optimize_and_run,
        )

        result = optimize_and_run(dataset, job, config=native_config(job))
    else:
        result = dataset.run(config=native_config(job))
    if aggregate:
        from experiments.system_comparison.reporting.timing import stage

        with stage("host", "shared deterministic aggregate"):
            return aggregate_frame(result, aggregate)
    if reduction:
        import pandas as pd

        rows = [{"result": record["result"]} for record in result]
        # The source Nirvana Reduce emits a sentinel on empty input; Abacus's
        # native aggregate instead produces zero records. Normalize that edge.
        return pd.DataFrame(rows or [{"result": "No data to process."}])
    return result


def run_relational(system, sql, tables, job, semantic=None):
    """Execute ordinary relations in DuckDB and semantic stages in native APIs.

    This disclosed adapter does not exercise the system's whole-query optimizer.
    An explicit semantic callback also allows model-free differential testing.
    """
    import duckdb
    import pandas as pd

    from experiments.validation.runtime.relational_port import SQLPort

    if semantic is None and job.get("native_mechanisms"):
        from experiments.system_comparison.execution.graph import run

        return run(system, sql, tables, job)
    if semantic is None:
        from experiments.validation.native_api import NativeAPI

        Path(job["output"]).mkdir(parents=True, exist_ok=True)
        semantic = NativeAPI(system, job)
    with duckdb.connect(config={"threads": CONCURRENCY}) as con:
        for index, (name, frame) in enumerate(tables.items()):
            view = f"__input_frame_{index}"
            con.register(view, frame)
            quoted = '"' + name.replace('"', '""') + '"'
            schema = job.get("input_schemas", {}).get(name) or frame.attrs.get(
                "duckdb_schema"
            )
            if not schema:
                raise ValueError(
                    f'{name}: SQL host requires input_schemas or DataFrame.attrs["duckdb_schema"] to preserve SQL types'
                )
            columns = ", ".join(
                'CAST("'
                + column.replace('"', '""')
                + '" AS '
                + kind
                + ') AS "'
                + column.replace('"', '""')
                + '"'
                for column, kind in schema
            )
            con.execute(f"CREATE TABLE {quoted} AS SELECT {columns} FROM {view}")
        cursor = SQLPort(con, semantic).execute(sql)
        schema = [(c[0], str(c[1])) for c in cursor.description]
        frame = pd.DataFrame(
            cursor.fetchall(), columns=[c[0] for c in schema], dtype=object
        )
        frame.attrs["duckdb_schema"] = schema
        return frame


def load_common_tables(database, dataset):
    """Read shared inputs without converting DECIMAL to floating point."""
    import duckdb
    import pandas as pd

    table_names = {
        "estate": ("nirvana_estate",),
        "movies": ("nirvana_movies",),
        "appstore": ("user_reviews", "playstore"),
        "movies_sema": (
            "movie",
            "movie_genres",
            "genre",
            "movie_languages",
            "language",
            "production_country",
            "country",
        ),
        "tpch": (
            "part",
            "supplier",
            "customer",
            "orders",
            "lineitem",
            "partsupp",
            "nation",
            "region",
        ),
    }
    tables = {}
    with duckdb.connect(str(database), read_only=True) as con:
        available = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        for name in table_names[dataset]:
            relation = "raw_" + name if "raw_" + name in available else name
            cursor = con.execute('SELECT * FROM "' + relation.replace('"', '""') + '"')
            schema = [(c[0], str(c[1])) for c in cursor.description]
            frame = pd.DataFrame(
                cursor.fetchall(), columns=[c[0] for c in schema], dtype=object
            )
            frame.attrs["duckdb_schema"] = schema
            tables[name] = frame
    return tables


def candidate_frame(tables, sql):
    """Execute an explicit, ordinary SQL prefix; semantic operations stay native."""
    import duckdb
    import pandas as pd
    import sqlglot
    from sqlglot import exp

    tree = sqlglot.parse_one(sql, read="duckdb")
    if not isinstance(tree, exp.Select) or any(
        n.name.lower().startswith("semantic") for n in tree.find_all(exp.Anonymous)
    ):
        raise ValueError("Candidate SQL must be an ordinary SELECT")
    from experiments.system_comparison.reporting.timing import stage

    with stage("host", "ordinary source candidate relation"):
        with duckdb.connect(config={"threads": CONCURRENCY}) as con:
            for i, (name, frame) in enumerate(tables.items()):
                schema = frame.attrs.get("duckdb_schema")
                if not schema:
                    raise ValueError("Candidate inputs require declared SQL types")
                con.register("__frame_" + str(i), frame)
                columns = ", ".join(
                    f"CAST({quote(c)} AS {t}) AS {quote(c)}" for c, t in schema
                )
                con.execute(
                    f"CREATE VIEW {quote(name)} AS SELECT {columns} FROM __frame_{i}"
                )
            cursor = con.execute(sql)
            schema = [(c[0], str(c[1])) for c in cursor.description]
            output = pd.DataFrame(
                cursor.fetchall(), columns=[c[0] for c in schema], dtype=object
            )
            output.attrs["duckdb_schema"] = schema
            return output


def nirvana_frame(frame):
    """Build a native frame using declared SQL types, including nullable IDs."""
    from experiments.system_comparison.backends.nirvana.graph import Nirvana

    return Nirvana({}).source(frame, dict(frame.attrs["duckdb_schema"]))


def output_columns(output, columns):
    """Keep SQL output columns and duplicate rows, including typed empty output."""
    import pandas as pd

    if not isinstance(output, pd.DataFrame):
        output = pd.DataFrame(
            [row.to_dict() if hasattr(row, "to_dict") else dict(row) for row in output]
        )
    if output.empty:
        return output.reindex(columns=columns)
    return output.loc[:, columns].copy()


def query_main(namespace):
    """Print a generated query's immutable source metadata without model calls."""
    print(
        json.dumps(
            {
                "system": namespace.get("SYSTEM"),
                "query_id": namespace.get("QUERY_ID"),
                "source": namespace.get("SOURCE"),
                "source_sha256": namespace.get("SOURCE_SHA256"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_lotus(dataframe, job):
    """LOTUS eager operators execute during DataFrame construction."""
    return dataframe


def run_lotus_relational(sql, tables, job, model=None, semantic=None):
    """Disclose the ordinary-relation boundary for SQL-origin workloads."""
    if semantic is None and job.get("native_mechanisms"):
        from experiments.system_comparison.execution.graph import run

        return run("lotus", sql, tables, job)
    if semantic is None:
        import lotus
        import pandas as pd

        from experiments.validation.native_api import instruction

        if model is not None:
            lotus.settings.configure(lm=model)

        def semantic(kind, task, rows, fields):
            names = [f"arg{i}" for i in range(len(rows[0]))]
            frame = pd.DataFrame(rows, columns=names, dtype=object)
            bound = instruction(task, ["{" + name + "}" for name in names])
            if kind == "projection":
                if len(fields) != 1:
                    raise ValueError("LOTUS host projection requires one output field")
                name = fields[0]["name"]
                output = frame.sem_map(bound, suffix=name)
                return [{name: value} for value in output[name]]
            accepted = set(frame.sem_filter(bound).index)
            return [i in accepted for i in range(len(frame))]

    return run_relational("lotus", sql, tables, job, semantic=semantic)
