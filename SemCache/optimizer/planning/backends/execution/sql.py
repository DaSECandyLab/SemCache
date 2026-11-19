"""Python differential sampling backend with typed input-table reuse."""

import json
import tempfile
import threading
import time
from pathlib import Path

from optimizer_server import OptimizerServer
from query_runner import quote

from planning.observations.evidence import (
    freeze_evidence,
)
from planning.settings.configuration import response_cache_settings
from planning.settings.parallelism import positive_parallelism

from ...state import digest, task_key
from .policy import StageActionPolicy
from .sql_estimates import _SQLEstimateMixin
from .sql_tables import _SQLTableMixin
from .values import identifier, parse_json_cell


class SQLBackend(_SQLTableMixin, _SQLEstimateMixin):
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

    def _stage(self, op, inputs, model, layout):
        if self.connection is None:
            raise RuntimeError("SQL backend is closed")
        from ...sampling import SampleRow

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
