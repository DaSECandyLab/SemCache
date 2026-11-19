"""Temporary sample table lifecycle, serialization and transaction scopes."""

import json
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass

from planning.observations.evidence import (
    _EvidenceDict,
    is_frozen_evidence,
)

from .values import identifier, literal


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


class _SQLTableMixin:
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
