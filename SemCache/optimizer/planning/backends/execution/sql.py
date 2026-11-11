"""Python differential sampling backend with typed input-table reuse."""

import copy
import json
import math
import tempfile
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from optimizer_server import OptimizerServer
from query_runner import quote

from planning.observations.evidence import (
    _EvidenceDict,
    freeze_evidence,
    is_frozen_evidence,
)
from planning.settings.configuration import response_cache_settings
from planning.settings.parallelism import positive_parallelism

from ...state import digest, task_key
from .policy import StageActionPolicy
from .values import identifier, literal, parse_json_cell


@dataclass(eq=False)
class _InputTable:
    rows: tuple
    row_columns: tuple
    columns: tuple
    types: tuple
    name: str
    borrowers: int = 1
    committed: bool = False
    normalized: dict | None = None


class SQLBackend:
    """Materialize one logical stage; DuckDB chooses its relational implementation.

    Trusted operator SQL uses $INPUT or $LEFT/$RIGHT. Unary SQL preserves
    __sample_id; binary SQL emits __left_sample_id and __right_sample_id. Only
    the declared model/layout is passed to the single semantic stage socket.
    """

    physical_request_telemetry = True

    def __init__(
        self,
        build,
        models,
        *,
        field_types=None,
        database=None,
        setup="",
        collect_telemetry=False,
        history_sink=None,
        observation_sink=None,
        adaptive_sketch=None,
        parallelism=1,
        cpu_workers=None,
    ):
        from ..duckdb import DuckDBConnection

        self.parallelism = positive_parallelism(parallelism)
        cpu_limit = (
            self.parallelism
            if cpu_workers is None
            else positive_parallelism(cpu_workers)
        )
        if cpu_workers is not None and cpu_limit > 64:
            raise ValueError("cpu_workers must be between 1 and 64")
        sql_workers = min(self.parallelism, cpu_limit)
        self.temporary = tempfile.TemporaryDirectory(prefix="sem-mixed-")
        self.policy = StageActionPolicy()
        self.server = OptimizerServer(
            Path(self.temporary.name) / "stage.sock", self.policy
        )
        self.decisions = []
        self.operator_specifications = {}
        self.system_prefix = None
        self.server.record = self._record
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.connection = None
        self._transaction_active = False
        self.types = field_types or {}
        self.serial = 0
        self.history_sink = history_sink
        self.adaptive_sketch = dict(adaptive_sketch or {})
        self.observation_sink = observation_sink
        self.collect_telemetry = collect_telemetry or history_sink is not None
        self.observations = []
        self.events = []
        self.telemetry = Path(self.temporary.name) / "telemetry.jsonl"
        try:
            build = Path(build).resolve()
            self.connection = DuckDBConnection(build, database, parallelism=sql_workers)
            self.connection.execute(
                f"LOAD {quote(build / 'extension/semantic/semantic.duckdb_extension')};"
            )
            self.connection.execute(f"SET semantic_parallelism={self.parallelism};")
            self.connection.execute(f"SET semantic_models={quote(json.dumps(models))};")
            self.connection.execute(
                "SET semantic_response_cache="
                + quote(json.dumps(response_cache_settings()))
                + ";"
            )
            self.connection.execute(
                f"SET semantic_optimizer_socket={quote(Path(self.temporary.name) / 'stage.sock')};"
            )
            self.connection.execute("SET semantic_optimizer_timeout_ms=120000;")
            # Request budgets require HTTP-boundary events even when callers do
            # not retain diagnostic observations or history.
            self.connection.execute(f"SET semantic_telemetry={quote(self.telemetry)};")
            if setup:
                self.connection.execute(setup)
            self.connection.execute("BEGIN TRANSACTION;")
            self._transaction_active = True
        except BaseException:
            self.close()
            raise

    def _record(self, query, response):
        self.decisions.append({"query_ir": query, "plan_spec": response})
        self.system_prefix = query.get("global_system_prefix")
        self.operator_specifications.update(
            {
                op["kind"]: op["operator_specification"]
                for op in query["semantic_operators"]
                if "operator_specification" in op
            }
        )

    def load(self, relation, sql):
        from ...sampling import SampleRow

        if self.connection is None:
            raise RuntimeError("SQL backend is closed")
        records = self.connection.execute(sql, named=True)
        if any("__sample_id" not in r or r["__sample_id"] is None for r in records):
            raise ValueError("Base SQL must expose a stable non-NULL __sample_id")
        return [
            SampleRow(
                str(r["__sample_id"]),
                freeze_evidence(
                    {k: self._value(k, v) for k, v in r.items() if k != "__sample_id"}
                ),
                ((relation, r["__sample_id"]),),
            )
            for r in records
        ]

    def _value(self, key, value):
        if value is None:
            return None
        kind = self.types.get(key, "VARCHAR").upper()
        if kind in (
            "TINYINT",
            "SMALLINT",
            "INTEGER",
            "BIGINT",
            "HUGEINT",
            "UTINYINT",
            "USMALLINT",
            "UINTEGER",
            "UBIGINT",
            "UHUGEINT",
        ):
            return int(value)
        if kind in ("FLOAT", "REAL", "DOUBLE"):
            return float(value)
        if kind == "BOOLEAN":
            return value in (True, "true")
        if kind.startswith(("STRUCT", "LIST", "MAP")) or kind.endswith("[]"):
            # The C API supplies DuckDB display strings. JSON is requested for
            # compound sample fields in _stage rather than parsing that display.
            return json.loads(value) if isinstance(value, str) else value
        return value

    def _input_cache(self):
        # Lazily initialize to preserve callers constructing lightweight backends.
        if not hasattr(self, "_input_tables"):
            self._input_tables = OrderedDict()
            self._input_table_names = {}
            self._input_table_rows = 0
            self._retired_input_table_names = set()
            self._committed_transient_table_names = set()
        return self._input_tables

    def _release_table(self, table):
        self._input_cache()
        entry = self._input_table_names.get(table)
        if entry is not None:
            entry.borrowers -= 1
        elif self.connection is not None:
            self.connection.execute(f"DROP TABLE IF EXISTS {table};")
            if table in self._committed_transient_table_names:
                self._retired_input_table_names.add(table)
                self._committed_transient_table_names.remove(table)

    def _rollback_input_tables(self, tables):
        """Invalidate before rollback; committed inputs can survive the rollback."""
        cached = list(self._input_cache().values())
        names = list(
            dict.fromkeys(
                [entry.name for entry in cached]
                + list(self._retired_input_table_names)
                + list(self._committed_transient_table_names)
                + list(tables)
            )
        )
        self._input_tables.clear()
        self._input_table_names.clear()
        self._retired_input_table_names.clear()
        self._committed_transient_table_names.clear()
        self._input_table_rows = 0
        connection = self.connection
        if connection is None:
            return
        restart = getattr(self, "_transaction_active", False)
        try:
            if restart:
                connection.execute("ROLLBACK;")
                self._transaction_active = False
            for name in names:
                connection.execute(f"DROP TABLE IF EXISTS {name};")
            if restart:
                connection.execute("BEGIN TRANSACTION;")
                self._transaction_active = True
        except Exception:
            # A failed rollback/drop/restart leaves unknown transactional state.
            # Closing releases every table while preserving the original error.
            self.connection = None
            self._transaction_active = False
            try:
                connection.close()
            except Exception:
                pass

    @contextmanager
    def _table_scope(self, inputs, names):
        tables = []
        try:
            for rows, columns in zip(inputs, names):
                tables.append(self._table(rows, columns))
            yield tables
            for table in tables:
                self._release_table(table)
        except BaseException:
            self._rollback_input_tables(tables)
            raise

    def _table(self, rows, names):
        if self.connection is None:
            raise RuntimeError("SQL backend is closed")
        cache = self._input_cache()
        source = next((entry for entry in cache.values() if entry.rows is rows), None)
        row_columns = (
            source.row_columns
            if source is not None
            else tuple(dict.fromkeys(k for row in rows for k in row.values))
        )
        columns = tuple(
            dict.fromkeys(["__sample_id"] + list(names) + list(row_columns))
        )
        types = tuple(
            "VARCHAR" if c == "__sample_id" else self.types.get(c, "VARCHAR")
            for c in columns
        )
        key = (id(rows), columns, types)
        if key in cache:
            entry = cache[key]
            cache.move_to_end(key)
            entry.borrowers += 1
            return entry.name

        from ...sampling import SampleRow

        eligible = source is not None or (
            type(rows) is tuple
            and len(rows) <= 8192
            and all(
                type(row) is SampleRow
                and type(row.values) is _EvidenceDict
                and is_frozen_evidence(row.values)
                for row in rows
            )
        )
        self.serial += 1
        table = f"mixed_input_{self.serial}"
        definitions = ",".join(
            identifier(c) + " " + kind for c, kind in zip(columns, types)
        )
        try:
            self.connection.execute(f"CREATE TEMP TABLE {table} ({definitions});")
            if rows:
                scalar = all(
                    not isinstance(value, (dict, list, tuple))
                    for row in rows
                    for value in row.values.values()
                )
                if scalar:
                    self.connection.append_scalar_rows(
                        table,
                        (
                            [
                                row.identity
                                if c == "__sample_id"
                                else row.values.get(c)
                                for c in columns
                            ]
                            for row in rows
                        ),
                    )
                else:
                    # Compound values require DuckDB's typed STRUCT/LIST expressions;
                    # their string cast syntax does not preserve JSON escape sequences.
                    values = ",".join(
                        "("
                        + ",".join(
                            literal(
                                row.identity
                                if c == "__sample_id"
                                else row.values.get(c)
                            )
                            for c in columns
                        )
                        + ")"
                        for row in rows
                    )
                    self.connection.execute(f"INSERT INTO {table} VALUES {values};")
            if eligible:
                # Evict only released inputs. Two active join inputs may together
                # exceed the row bound, so the second remains a fresh transient table.
                while cache and (
                    len(cache) >= 32 or self._input_table_rows + len(rows) > 8192
                ):
                    victim = next(
                        ((k, e) for k, e in cache.items() if not e.borrowers), None
                    )
                    if victim is None:
                        eligible = False
                        break
                    old_key, entry = victim
                    self.connection.execute(f"DROP TABLE {entry.name};")
                    if entry.committed:
                        self._retired_input_table_names.add(entry.name)
                    del cache[old_key]
                    del self._input_table_names[entry.name]
                    self._input_table_rows -= len(entry.rows)
                if eligible:
                    entry = _InputTable(rows, row_columns, columns, types, table)
                    cache[key] = entry
                    self._input_table_names[table] = entry
                    self._input_table_rows += len(rows)
        except BaseException:
            self._rollback_input_tables([table])
            raise
        return table

    def __call__(self, op, inputs, model, layout):
        if not self.collect_telemetry and not model:
            return self._stage(op, inputs, model, layout)
        self.telemetry.write_text("")
        started = time.perf_counter()
        stage_failed = False
        try:
            rows = self._stage(op, inputs, model, layout)
        except BaseException:
            stage_failed = True
            raise
        finally:
            seconds = time.perf_counter() - started
            try:
                events = []
                for line in self.telemetry.read_text().splitlines():
                    event = json.loads(line)
                    events.append(event)
                    # Preserve preceding confirmations if a failed writer left
                    # an incomplete final JSON record.
                    self.events.append(event)
                if self.history_sink is not None:
                    from planning.observations.statistics import ingest_history

                    ingest_history(
                        events, self.history_sink, adaptive=self.adaptive_sketch
                    )
            except BaseException:
                # Recover confirmed requests without replacing the stage's error.
                if not stage_failed:
                    raise
        if not self.collect_telemetry:
            return rows
        self.observations.append(
            {
                "operator_id": op["operator_id"],
                "kind": op["kind"],
                "model_id": model,
                "task_key": task_key(op) if model else None,
                "inputs": [len(group) for group in inputs],
                "seconds": seconds,
                "events": events,
            }
        )
        if self.observation_sink is not None:
            self.observation_sink(self.observations[-1])
        return rows

    def join_pairs(self, op, pairs, model, layout):
        """Evaluate only the selected pairs in one materialized semantic stage.

        Pair identities prevent the two temporary inputs from forming an extra
        Cartesian product. Original input groups remain with the sampler for
        population weights and left/right survival statistics.
        """
        from ...sampling import SampleRow

        if (
            op.get("kind", "").lower()
            not in (
                "semantic_join",
                "semantic_nested_loop_join",
            )
            or not model
        ):
            raise ValueError("Pair batching requires a semantic Join and model")
        pairs = tuple(pairs)
        if not pairs:
            return []
        columns = [
            list(dict.fromkeys(key for pair in pairs for key in pair[side].values))
            for side in (0, 1)
        ]
        if not columns[0] or not columns[1] or set(columns[0]) & set(columns[1]):
            raise ValueError("Candidate-pair inputs require disjoint field partitions")
        rows = [
            SampleRow(
                digest([left.identity, right.identity]),
                freeze_evidence(
                    {
                        **{key: left.values.get(key) for key in columns[0]},
                        **{key: right.values.get(key) for key in columns[1]},
                    }
                ),
                tuple(sorted(set(left.lineage) | set(right.lineage), key=repr)),
            )
            for left, right in pairs
        ]
        if len({row.identity for row in rows}) != len(rows):
            raise ValueError("Candidate-pair identities must be unique")
        output = self({**op, "pair_columns": columns}, (rows,), model, layout)
        if self.collect_telemetry:
            self.observations[-1].update(
                candidate_pair_count=len(pairs),
                source_input_rows=[
                    len({pair[side].identity for pair in pairs}) for side in (0, 1)
                ],
            )
        return output

    def reset_observations(self):
        self.observations.clear()
        self.events.clear()

    def prepare_inputs(self, op, inputs):
        """Evaluate bound compound/generated input expressions on the sample."""
        if self.connection is None:
            raise RuntimeError("SQL backend is closed")

        output = []
        for side, rows in enumerate(inputs):
            fields = [
                f
                for f in op.get("input_field_schema", [])
                if f.get("value_key", "").startswith("__semantic_input:")
                and (
                    len(inputs) == 1
                    or (f["name"] in op.get("right_input_fields", [])) == (side == 1)
                )
            ]
            if not fields or not rows:
                output.append(rows)
                continue
            from ...sampling.rows import CartesianRows

            if isinstance(rows, CartesianRows):
                from ...state import field_key

                dependencies = [
                    key if isinstance(key, str) else field_key(key)
                    for key in op.get("required_columns", [])
                ]
                leaf = rows.leaf_for(dependencies) if dependencies else None
                if leaf is not None:
                    output.append(
                        rows.replace_leaf(leaf, self._prepare_rows(leaf, fields))
                    )
                else:
                    expressions = ",".join(
                        field["sample_expression"]
                        + " AS "
                        + identifier(field["value_key"])
                        for field in fields
                    )
                    output.append(
                        tuple(
                            self._stage(
                                {
                                    "kind": "PROJECTION",
                                    "operator_id": op["operator_id"],
                                    "sql": "SELECT *," + expressions + " FROM $INPUT",
                                    "_preserve_sample_inputs": True,
                                },
                                (rows,),
                                None,
                                None,
                            )
                        )
                    )
            else:
                output.append(self._prepare_rows(rows, fields))
        return tuple(output)

    def _prepare_rows(self, rows, fields):
        from ...sampling import SampleRow

        if self.connection is None:
            raise RuntimeError("SQL backend is closed")
        with self._table_scope((rows,), ([],)) as tables:
            expressions = ",".join(
                field["sample_expression"] + " AS " + identifier(field["value_key"])
                for field in fields
            )
            records = self.connection.execute(
                "SELECT semantic_sample_json(t) FROM (SELECT *,"
                + expressions
                + " FROM "
                + tables[0]
                + ") t"
            )
            origins = {row.identity: row.lineage for row in rows}
            result = []
            for record in records:
                values = parse_json_cell(record[0])
                identity = values.pop("__sample_id")
                result.append(
                    SampleRow(identity, freeze_evidence(values), origins[identity])
                )
            return tuple(result)

    def _estimate_population_relation(self, op):
        """Keep real-table key statistics when estimating a relational stage."""
        cache = getattr(self, "relational_estimates", None)
        if cache is None:
            self.relational_estimates = cache = {}
        statements = op["population_input_sql"]
        if not isinstance(statements, list) or not statements:
            raise ValueError("Population estimate requires input SQL")
        key = digest(
            ["population-estimate-v1", op["population_estimate_sql"], statements]
        )
        if key in cache:
            return copy.deepcopy(cache[key])

        def cardinality(node):
            value = node.get("extra_info", {}).get("Estimated Cardinality")
            if value is not None:
                return float(value)
            children = node.get("children", [])
            return cardinality(children[0]) if len(children) == 1 else None

        def proven_empty(node):
            name = node.get("name", "").upper()
            if name == "EMPTY_RESULT":
                return True
            children = node.get("children", [])
            return (
                len(children) == 1
                and name
                in {
                    "PROJECTION",
                    "FILTER",
                    "ORDER_BY",
                    "LIMIT",
                    "STREAMING_LIMIT",
                    "TOP_N",
                }
                and proven_empty(children[0])
            )

        rounded = False

        def explain(sql):
            nonlocal rounded
            identity = digest(["population-cardinality-v1", sql])
            if identity not in cache:
                rows = self.connection.execute("EXPLAIN (FORMAT JSON) " + sql)
                tree = parse_json_cell(rows[0][-1])[0]
                value = 0.0 if proven_empty(tree) else cardinality(tree)
                floor = value == 0 and not proven_empty(tree)
                cache[identity] = {
                    "cardinality": 1.0 if floor else value,
                    "rounded_zero": floor,
                }
            record = cache[identity]
            rounded |= record["rounded_zero"]
            return record["cardinality"]

        sides = [explain(sql) for sql in statements]
        known = all(
            type(n) in (int, float) and math.isfinite(n) and n >= 0 for n in sides
        )
        incoming = math.prod(sides) if known else None
        output = explain(op["population_estimate_sql"])
        rate = output / incoming if incoming and output is not None else None
        kind = op.get("kind", op.get("type", "")).upper()
        contracting = kind in {"FILTER", "DISTINCT"} or (
            kind == "AGGREGATE" and not op.get("scalar_aggregate")
        )
        capped = contracting and rate is not None and rate > 1
        result = {
            "input_rows": incoming,
            "output_rows": incoming if capped else output,
            "selectivity": 1.0 if capped else rate,
            "population_input_rows": sides,
            "population_output_rows": output,
            "population_rounded_zero": rounded,
            "population_contraction_capped": capped,
            "uncertain": rounded or capped,
            "cardinality_basis": "DuckDB EXPLAIN on real-table relational context",
        }
        cache[key] = result
        return copy.deepcopy(result)

    def estimate_relation(self, op, inputs):
        """DuckDB EXPLAIN estimates for the actual bound relational stage.

        Temporary sample inputs provide field distributions. The resulting local
        selectivity is propagated onto population inputs, rather than treating the
        sample's output count as a population cardinality.
        """
        if self.connection is None:
            raise RuntimeError("SQL backend is closed")
        if "population_sample" in op:
            cache = getattr(self, "relational_estimates", None)
            if cache is None:
                self.relational_estimates = cache = {}
            spec = op["population_sample"]
            key = digest(["population-sample-v1", spec["probe_sql"]])
            if key not in cache:
                started = time.perf_counter()
                left, right, n, passed = map(
                    float, self.connection.execute(spec["probe_sql"])[0]
                )
                join, census = spec["kind"] == "join", n == left
                denominator = n * (right if join else 1)
                point = passed / denominator if denominator else 0
                floored = denominator > 0 and passed == 0 and not census
                rate = 1 / denominator if floored else point
                incoming = left * (right if join else 1)
                cache[key] = {
                    "input_rows": incoming, "output_rows": incoming * rate,
                    "selectivity": rate, "sampled_selectivity": point,
                    "sampled_selectivity_complete": True,
                    "population_input_rows": [left, right] if join else [left],
                    "population_output_rows": incoming * rate,
                    "population_sample_rows": n,
                    "population_sample_output_rows": passed,
                    "population_sample_census": census,
                    "population_sample_zero_floor": floored,
                    "population_cardinality_sampling": True,
                    "conditioned_relational_filters": spec["conditioned_filters"],
                    "effective_samples": n, "uncertain": not census,
                    "reliability": {"selectivity": 1 if census else n / (n + 1)},
                    "population_sampling_seconds": time.perf_counter() - started,
                    "cardinality_basis": (
                        "one-sided reservoir join sample against full peer with ready relational filters"
                        if join else "reservoir sample of base-field relational predicate"
                    ),
                }
            sampled = copy.deepcopy(cache[key])
            from planning.kernels import _search_core

            estimated = {}
            try:
                if "population_estimate_sql" in op and "population_input_sql" in op:
                    estimated = self._estimate_population_relation(op)
            except Exception as error:
                sampled["duckdb_estimate_error"] = str(error)
            return _search_core.mean_relational_estimate(sampled, estimated)
        if "population_estimate_sql" in op and "population_input_sql" in op:
            return self._estimate_population_relation(op)
        from ...sampling.rows import CartesianRows
        from ..factorized import bound_sql, fingerprint, relation_scope

        factored = any(isinstance(rows, CartesianRows) for rows in inputs)
        key = digest(
            [
                op.get("sampling_sql", op.get("sql")),
                [fingerprint(rows) for rows in inputs]
                if factored
                else [[r.values for r in rows] for rows in inputs],
            ]
        )
        cache = getattr(self, "relational_estimates", None)
        if cache is None:
            self.relational_estimates = cache = {}
        if key in cache:
            return copy.deepcopy(cache[key])
        names = op.get("input_columns", [[]] * len(inputs))
        scope = (
            relation_scope(self, inputs)
            if factored
            else self._table_scope(inputs, names)
        )
        with scope as bound:
            tables = bound[2] if factored else bound
            expressions = bound[0] if factored else tables
            sql = (
                op.get("candidate_sql")
                if len(inputs) == 2 and op.get("kind", "").lower() == "filter"
                else op.get("sql", op.get("sampling_sql"))
            )
            if not sql:
                raise ValueError("Relational sampling requires bound stage SQL")
            sql = (
                bound_sql(op, expressions)
                if factored
                else (
                    sql.replace("$INPUT", tables[0])
                    if len(inputs) == 1
                    else sql.replace("$LEFT", tables[0]).replace("$RIGHT", tables[1])
                )
            )
            # DuckDB withholds statistics for transaction-local inserts. These
            # private temporary tables must be visible to its estimator first.
            self.connection.execute("COMMIT; BEGIN TRANSACTION;")
            # Only tables present at this commit can reappear on a later rollback;
            # names are bounded by 32 cached tables plus active stage inputs.
            self._retired_input_table_names.clear()
            self._committed_transient_table_names = set(tables).difference(
                self._input_table_names
            )
            for entry in self._input_cache().values():
                entry.committed = True
            profile = self.connection.execute("EXPLAIN (FORMAT JSON) " + sql)
            tree = parse_json_cell(profile[0][-1])

            def cardinality(node):
                value = node.get("extra_info", {}).get("Estimated Cardinality")
                if value is not None:
                    return float(value)
                children = node.get("children", [])
                return cardinality(children[0]) if len(children) == 1 else None

            output = cardinality(tree[0])
            incoming = math.prod(len(rows) for rows in inputs)
            result = {
                "input_rows": incoming,
                "output_rows": None,
                "selectivity": None,
                "cardinality_basis": "DuckDB EXPLAIN of bound relational stage",
            }
            if output is not None:
                result.update(
                    output_rows=output,
                    selectivity=output / incoming if incoming else None,
                )
            cache[key] = result
            return copy.deepcopy(result)

    @staticmethod
    def sample_output_column(op):
        if op["kind"].upper() != "SEMANTIC_PROJECTION":
            return None
        sql = op.get("sql", op.get("sampling_sql", ""))
        target = sql.rsplit(" AS ", 1)[-1].split(" FROM ", 1)[0].strip()
        if target.startswith('"') and target.endswith('"'):
            return target[1:-1].replace('""', '"')
        return target

    @staticmethod
    def sample_request_key(op, inputs, model, layout):
        from planning.observations.statistics import serialize_value

        descriptors = {
            field["name"]: field for field in op.get("input_field_schema", [])
        }
        names = op.get("input_fields") or op.get("left_input_fields", []) + op.get(
            "right_input_fields", []
        )
        values = []
        for name in names:
            key = descriptors.get(name, {}).get("value_key", name)
            side = (
                1
                if len(inputs) == 2 and name in op.get("right_input_fields", [])
                else 0
            )
            row = inputs[side][0]
            if key not in row.values:
                raise ValueError("Semantic sampling needs prepared actual input values")
            values.append(serialize_value(row.values[key]))
        return digest(
            [op["kind"], op["task"], op.get("output_schema"), model, layout, values]
        )

    def cartesian(self, op, inputs, limit):
        """Keep exact products; calibrate only on a bounded real SQL probe."""
        from ...sampling.rows import CartesianRows

        try:
            product = CartesianRows(*inputs)
        except ValueError:
            return tuple(self(op, inputs, None, None))
        if not product:
            return ()
        calibrated = getattr(self, "_product_calibrations", None)
        if calibrated is None:
            self._product_calibrations = calibrated = set()
        if op["operator_id"] not in calibrated:
            count = min(len(inputs[0]), max(1, math.isqrt(limit)))
            probe = (inputs[0][:count], inputs[1][: max(1, limit // count)])
            self(op, probe, None, None)
            calibrated.add(op["operator_id"])
        return product

    def _stage(self, op, inputs, model, layout):
        if self.connection is None:
            raise RuntimeError("SQL backend is closed")
        from ...sampling import SampleRow
        from ...sampling.rows import CartesianRows

        if model is None and any(isinstance(rows, CartesianRows) for rows in inputs):
            if "AGGREGATE" in op.get("kind", op.get("type", "")).upper():
                # Bound aggregates number groups in input encounter order. Keep
                # that ordering and its existing sample IDs via the ordinary path.
                inputs = tuple(
                    tuple(rows) if isinstance(rows, CartesianRows) else rows
                    for rows in inputs
                )
            else:
                from ..factorized import execute

                return execute(self, op, inputs)
        if len(inputs) not in (1, 2) or not op.get("sql"):
            raise ValueError(
                "Mixed execution requires trusted operator SQL and one/two inputs"
            )
        names = op.get("input_columns", [])
        sql = (
            op.get("candidate_sql")
            if len(inputs) == 2 and op["kind"].lower() == "filter"
            else op["sql"]
        )
        paired = len(inputs) == 1 and "join" in op["kind"].lower()
        if paired:
            names = op.get("pair_columns")
            if not names or len(names) != 2 or set(names[0]) & set(names[1]):
                raise ValueError(
                    "Candidate-pair execution requires disjoint input field partitions"
                )
            names = [list(side) for side in names]
            if " FROM $LEFT l JOIN $RIGHT r ON " not in sql:
                raise ValueError(
                    "Candidate-pair SQL requires the trusted left/right join template"
                )
            original = inputs[0]
            # An empty cohort has no row schema and cannot issue a request.
            if not original:
                return ()
            # Bound expressions are prepared after the relational pair fixed
            # its base fields. Carry only declared temporary arguments into
            # their respective side; retain strict validation of all fields.
            present = {key for row in original for key in row.values}
            from ...state import field_key

            required = {
                key if isinstance(key, str) else field_key(key)
                for key in op.get("required_columns", [])
            }
            if not required <= present:
                raise ValueError("Candidate input lost a required source field")
            # Independent projection slices and complete base schemas share
            # the same original column ownership. Drop only absent outputs;
            # undeclared extras still fail the exact row-schema check below.
            names = [[key for key in side if key in present] for side in names]
            fields = set(names[0]) | set(names[1])
            for field in op.get("input_field_schema", []):
                key = field.get("value_key", "")
                if (
                    key.startswith("__semantic_input:")
                    and key in present
                    and key not in fields
                ):
                    side = int(field["name"] in op.get("right_input_fields", []))
                    names[side].append(key)
                    fields.add(key)
            if any(set(r.values) != fields for r in original):
                raise ValueError("Candidate input changed its field partition")
            inputs = tuple(
                [
                    SampleRow(
                        r.identity,
                        freeze_evidence({k: r.values[k] for k in side}),
                        r.lineage,
                    )
                    for r in original
                ]
                for side in names
            )
            sql = sql.replace(" ON ", " ON l.__sample_id=r.__sample_id AND ", 1)
        if not sql:
            raise ValueError("Binary candidate filter requires trusted candidate_sql")
        table_names = [
            names[index] if names and isinstance(names[0], list) else names
            for index in range(len(inputs))
        ]
        with self._table_scope(inputs, table_names) as tables:
            if len(inputs) == 1:
                if "$INPUT" not in sql:
                    raise ValueError("Unary SQL requires $INPUT")
                sql = sql.replace("$INPUT", tables[0])
            else:
                if "$LEFT" not in sql or "$RIGHT" not in sql:
                    raise ValueError("Join SQL requires $LEFT and $RIGHT")
                sql = sql.replace("$LEFT", tables[0]).replace("$RIGHT", tables[1])
            action = {
                "operator_id": op["operator_id"],
                "model_id": model,
                "layout": layout,
            }
            self.policy.action = action if model else None
            self.policy.pairwise = len(inputs) == 2
            self.policy.candidate_pairs = paired
            try:
                # Preserve typed generated values for downstream sketches/statistics.
                result = self.connection.execute(
                    "SELECT semantic_sample_json(t) AS sample FROM (" + sql + ") t"
                )
                result = [parse_json_cell(row[0]) for row in result]
            except RuntimeError as error:
                raise RuntimeError(
                    f"Logical stage {op['operator_id']}: {error}"
                ) from error
            finally:
                self.policy.action = None
                self.policy.candidate_pairs = False
            lineage = [{r.identity: r.lineage for r in rows} for rows in inputs]
            output = []
            for row in result:
                for key in list(row):
                    if key.startswith("__semantic_input:"):
                        row.pop(key)
                aggregate = "AGGREGATE" in op.get("kind", op.get("type", "")).upper()
                if aggregate:
                    identity = digest([op["operator_id"], row])
                    row.pop("__sample_id", None)
                    origin = ()
                elif len(inputs) == 1:
                    identity = row.pop("__sample_id", None)
                    if identity is None or identity not in lineage[0]:
                        raise ValueError("Unary SQL must preserve __sample_id")
                    origin = lineage[0][identity]
                else:
                    left, right = (
                        row.pop("__left_sample_id", None),
                        row.pop("__right_sample_id", None),
                    )
                    if (
                        left is None
                        or right is None
                        or left not in lineage[0]
                        or right not in lineage[1]
                    ):
                        raise ValueError(
                            "Join SQL must preserve both sample identities"
                        )
                    if paired and left != right:
                        raise ValueError("Semantic stage invented a candidate pair")
                    identity = left if paired else digest([left, right])
                    origin = tuple(
                        sorted(set(lineage[0][left]) | set(lineage[1][right]), key=repr)
                    )
                output.append(
                    SampleRow(
                        identity,
                        freeze_evidence({k: self._value(k, v) for k, v in row.items()}),
                        origin,
                    )
                )
            return output

    def close(self):
        if hasattr(self, "_input_tables"):
            self._input_tables.clear()
            self._input_table_names.clear()
            self._retired_input_table_names.clear()
            self._committed_transient_table_names.clear()
            self._input_table_rows = 0
        self._transaction_active = False
        if self.connection:
            self.connection.close()
            self.connection = None
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()
        self.temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
