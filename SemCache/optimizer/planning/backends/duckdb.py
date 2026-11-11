"""Small synchronous C API adapter using the repository's exact DuckDB build.

SQLBackend materializes stage inputs in temporary tables and reads resulting
scalar rows through this adapter, preserving NULLs and Unicode text.
"""

import ctypes as C
import math
from pathlib import Path
from typing import Literal, overload

from planning.settings.parallelism import positive_parallelism


class Result(C.Structure):
    _fields_ = [
        ("deprecated_column_count", C.c_uint64),
        ("deprecated_row_count", C.c_uint64),
        ("deprecated_rows_changed", C.c_uint64),
        ("deprecated_columns", C.c_void_p),
        ("deprecated_error_message", C.c_void_p),
        ("internal_data", C.c_void_p),
    ]


class String(C.Structure):
    _fields_ = [("data", C.c_void_p), ("size", C.c_uint64)]


class VectorString(C.Structure):
    # Pinned duckdb_string_t: 12 inline bytes or prefix + nonowning pointer.
    _fields_ = [("length", C.c_uint32), ("prefix", C.c_ubyte * 4), ("data", C.c_void_p)]


class DuckDBConnection:
    """Own a connection to the exact DuckDB library paired with the extension."""

    def __init__(self, build_dir, database=None, *, readonly=True, parallelism=1):
        parallelism = positive_parallelism(parallelism)
        self.lib = C.CDLL(str(Path(build_dir).resolve() / "src/libduckdb.so"))
        self.database, self.connection = C.c_void_p(), C.c_void_p()
        pointer = C.POINTER(C.c_void_p)
        result = C.POINTER(Result)
        signatures = {
            "duckdb_create_config": ([pointer], C.c_int),
            "duckdb_set_config": ([C.c_void_p, C.c_char_p, C.c_char_p], C.c_int),
            "duckdb_destroy_config": ([pointer], None),
            "duckdb_open_ext": ([C.c_char_p, pointer, C.c_void_p, pointer], C.c_int),
            "duckdb_connect": ([C.c_void_p, pointer], C.c_int),
            "duckdb_disconnect": ([pointer], None),
            "duckdb_close": ([pointer], None),
            "duckdb_query": ([C.c_void_p, C.c_char_p, result], C.c_int),
            "duckdb_destroy_result": ([result], None),
            "duckdb_result_error": ([result], C.c_char_p),
            "duckdb_column_count": ([result], C.c_uint64),
            "duckdb_column_type": ([result, C.c_uint64], C.c_int),
            "duckdb_result_chunk_count": ([Result], C.c_uint64),
            "duckdb_result_get_chunk": ([Result, C.c_uint64], C.c_void_p),
            "duckdb_destroy_data_chunk": ([pointer], None),
            "duckdb_data_chunk_get_size": ([C.c_void_p], C.c_uint64),
            "duckdb_data_chunk_get_vector": ([C.c_void_p, C.c_uint64], C.c_void_p),
            "duckdb_vector_get_data": ([C.c_void_p], C.c_void_p),
            "duckdb_vector_get_validity": ([C.c_void_p], C.POINTER(C.c_uint64)),
            "duckdb_column_name": ([result, C.c_uint64], C.c_char_p),
            "duckdb_row_count": ([result], C.c_uint64),
            "duckdb_value_is_null": ([result, C.c_uint64, C.c_uint64], C.c_bool),
            "duckdb_value_string": ([result, C.c_uint64, C.c_uint64], String),
            "duckdb_free": ([C.c_void_p], None),
            "duckdb_appender_create_ext": (
                [C.c_void_p, C.c_char_p, C.c_char_p, C.c_char_p, pointer],
                C.c_int,
            ),
            "duckdb_appender_column_count": ([C.c_void_p], C.c_uint64),
            "duckdb_appender_column_type": ([C.c_void_p, C.c_uint64], C.c_void_p),
            "duckdb_get_type_id": ([C.c_void_p], C.c_int),
            "duckdb_destroy_logical_type": ([pointer], None),
            "duckdb_appender_error": ([C.c_void_p], C.c_char_p),
            "duckdb_appender_end_row": ([C.c_void_p], C.c_int),
            "duckdb_appender_close": ([C.c_void_p], C.c_int),
            "duckdb_appender_destroy": ([pointer], C.c_int),
            "duckdb_append_null": ([C.c_void_p], C.c_int),
            "duckdb_append_bool": ([C.c_void_p, C.c_bool], C.c_int),
            "duckdb_append_int64": ([C.c_void_p, C.c_int64], C.c_int),
            "duckdb_append_double": ([C.c_void_p, C.c_double], C.c_int),
            "duckdb_append_varchar_length": (
                [C.c_void_p, C.c_char_p, C.c_uint64],
                C.c_int,
            ),
        }
        for name, (arguments, returns) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = arguments, returns
        config, error = C.c_void_p(), C.c_void_p()
        try:
            if self.lib.duckdb_create_config(C.byref(config)):
                raise RuntimeError("Cannot create DuckDB config")
            settings = {
                "allow_unsigned_extensions": "true",
                "threads": str(parallelism),
            }
            if database is not None and readonly:
                settings["access_mode"] = "READ_ONLY"
            for key, value in settings.items():
                if self.lib.duckdb_set_config(config, key.encode(), value.encode()):
                    raise RuntimeError(f"Invalid DuckDB setting: {key}")
            path = (
                str(Path(database).resolve()).encode() if database is not None else None
            )
            if self.lib.duckdb_open_ext(
                path, C.byref(self.database), config, C.byref(error)
            ):
                raise RuntimeError(
                    C.string_at(error).decode() if error.value else "Cannot open DuckDB"
                )
            if self.lib.duckdb_connect(self.database, C.byref(self.connection)):
                raise RuntimeError("Cannot connect to DuckDB")
        except BaseException:
            self.close()
            raise
        finally:
            self.lib.duckdb_destroy_config(C.byref(config))
            if error.value:
                self.lib.duckdb_free(error)

    @overload
    def execute(
        self, sql: str, *, named: Literal[True]
    ) -> list[dict[str, str | None]]: ...

    @overload
    def execute(
        self, sql: str, *, named: Literal[False] = False
    ) -> list[list[str | None]]: ...

    def execute(
        self, sql: str, *, named: bool = False
    ) -> list[dict[str, str | None]] | list[list[str | None]]:
        """Execute trusted stage SQL and return scalar rows with NULLs preserved."""

        if not self.connection.value:
            raise RuntimeError("DuckDB connection is closed")
        if "\0" in sql:
            raise ValueError("SQL cannot contain NUL")
        result = Result()
        try:
            if self.lib.duckdb_query(self.connection, sql.encode(), C.byref(result)):
                raise RuntimeError(
                    self.lib.duckdb_result_error(C.byref(result)).decode()
                )
            return self._read_result(result, named=named)
        finally:
            self.lib.duckdb_destroy_result(C.byref(result))

    def _read_result(self, result, *, named=False):
        columns = range(self.lib.duckdb_column_count(C.byref(result)))
        names = (
            [
                self.lib.duckdb_column_name(C.byref(result), col).decode()
                for col in columns
            ]
            if named
            else []
        )
        if columns and all(
            self.lib.duckdb_column_type(C.byref(result), col) == 17 for col in columns
        ):
            return self._read_strings(result, columns, names)
        rows = []
        for row in range(self.lib.duckdb_row_count(C.byref(result))):
            values = []
            for col in columns:
                if self.lib.duckdb_value_is_null(C.byref(result), col, row):
                    values.append(None)
                    continue
                value = self.lib.duckdb_value_string(C.byref(result), col, row)
                try:
                    values.append(C.string_at(value.data, value.size).decode())
                finally:
                    self.lib.duckdb_free(value.data)
            rows.append(dict(zip(names, values)) if named else values)
        return rows

    def _read_strings(self, result, columns, names):
        """Copy UTF-8 scalars once per chunk without allocating C strings per cell."""
        rows = []
        for index in range(self.lib.duckdb_result_chunk_count(result)):
            chunk = C.c_void_p(self.lib.duckdb_result_get_chunk(result, index))
            if not chunk.value:
                raise RuntimeError("DuckDB lost a materialized result chunk")
            try:
                size = self.lib.duckdb_data_chunk_get_size(chunk)
                values: list[list[str | None]] = [
                    [None] * len(columns) for _ in range(size)
                ]
                for col in columns:
                    vector = self.lib.duckdb_data_chunk_get_vector(chunk, col)
                    validity = self.lib.duckdb_vector_get_validity(vector)
                    data = C.cast(
                        self.lib.duckdb_vector_get_data(vector), C.POINTER(VectorString)
                    )
                    for row in range(size):
                        if validity and not validity[row >> 6] & (1 << (row & 63)):
                            continue
                        value = data[row]
                        address = (
                            C.addressof(value) + 4 if value.length <= 12 else value.data
                        )
                        values[row][col] = C.string_at(address, value.length).decode()
                if names:
                    rows.extend(dict(zip(names, row)) for row in values)
                else:
                    rows.extend(values)
            finally:
                self.lib.duckdb_destroy_data_chunk(C.byref(chunk))
        return rows

    def append_scalar_rows(self, table, rows):
        """Stream temporary-table scalars through DuckDB's DataChunk appender.

        DuckDB casts values to the table's column types. The caller owns the
        transaction; compound values retain the existing typed SQL path.
        """
        if not self.connection.value:
            raise RuntimeError("DuckDB connection is closed")
        if "\0" in table:
            raise ValueError("Table name cannot contain NUL")
        appender = C.c_void_p()

        def checked(state):
            if state:
                error = self.lib.duckdb_appender_error(appender)
                raise RuntimeError(
                    error.decode() if error else "DuckDB appender failed"
                )

        try:
            checked(
                self.lib.duckdb_appender_create_ext(
                    self.connection, b"temp", b"main", table.encode(), C.byref(appender)
                )
            )
            columns = self.lib.duckdb_appender_column_count(appender)
            column_types = []
            for index in range(columns):
                logical = C.c_void_p(
                    self.lib.duckdb_appender_column_type(appender, index)
                )
                try:
                    column_types.append(self.lib.duckdb_get_type_id(logical))
                finally:
                    self.lib.duckdb_destroy_logical_type(C.byref(logical))
            for row in rows:
                if len(row) != columns:
                    raise ValueError("Sample row does not match table columns")
                if any(type(v) not in (bool, int, float, str, type(None)) for v in row):
                    raise ValueError("Native sample loading requires scalar values")
                if any(type(v) is float and not math.isfinite(v) for v in row):
                    raise ValueError("Non-finite sample value")
                for index, value in enumerate(row):
                    if value is None:
                        state = self.lib.duckdb_append_null(appender)
                    elif type(value) is bool:
                        state = self.lib.duckdb_append_bool(appender, value)
                    elif column_types[index] == 1 and type(value) in (int, float):
                        state = self.lib.duckdb_append_bool(appender, bool(value))
                    elif type(value) is int and -(1 << 63) <= value < (1 << 63):
                        state = self.lib.duckdb_append_int64(appender, value)
                    elif type(value) is float:
                        # Plain SQL numeric literals retain decimal precision
                        # and normalize negative zero. Pinned type IDs:
                        # Integers=2..9/16/32, VARCHAR=17, DECIMAL=19.
                        # Exponent literals are DOUBLE.
                        value = value if value else 0.0
                        decimal = str(value)
                        if (
                            column_types[index] in (*range(2, 10), 16, 17, 19, 32)
                            and "e" not in decimal.lower()
                        ):
                            encoded = decimal.encode()
                            state = self.lib.duckdb_append_varchar_length(
                                appender, encoded, len(encoded)
                            )
                        else:
                            state = self.lib.duckdb_append_double(appender, value)
                    else:
                        encoded = str(value).encode()
                        state = self.lib.duckdb_append_varchar_length(
                            appender, encoded, len(encoded)
                        )
                    checked(state)
                checked(self.lib.duckdb_appender_end_row(appender))
            checked(self.lib.duckdb_appender_close(appender))
        finally:
            if appender.value:
                self.lib.duckdb_appender_destroy(C.byref(appender))

    def close(self):
        if self.connection.value:
            self.lib.duckdb_disconnect(C.byref(self.connection))
        if self.database.value:
            self.lib.duckdb_close(C.byref(self.database))
