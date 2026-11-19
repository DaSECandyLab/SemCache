"""DuckDB cardinality estimation and semantic sample cache keys."""

import copy
import math
import time

from ...state import digest
from .values import parse_json_cell


class _SQLEstimateMixin:
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
        key = digest(
            [
                op.get("sampling_sql", op.get("sql")),
                [[r.values for r in rows] for rows in inputs],
            ]
        )
        cache = getattr(self, "relational_estimates", None)
        if cache is None:
            self.relational_estimates = cache = {}
        if key in cache:
            return copy.deepcopy(cache[key])
        names = op.get("input_columns", [[]] * len(inputs))
        with self._table_scope(inputs, names) as tables:
            sql = (
                op.get("candidate_sql")
                if len(inputs) == 2 and op.get("kind", "").lower() == "filter"
                else op.get("sql", op.get("sampling_sql"))
            )
            if not sql:
                raise ValueError("Relational sampling requires bound stage SQL")
            sql = (
                sql.replace("$INPUT", tables[0])
                if len(inputs) == 1
                else sql.replace("$LEFT", tables[0]).replace("$RIGHT", tables[1])
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
