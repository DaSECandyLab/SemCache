"""Bounded ctypes bridge to the C++ coordinated bottom-k builder."""

import ctypes as C
import itertools
import json
from functools import lru_cache
from pathlib import Path

from . import _search_core

DEFAULT_LIBRARY = Path(_search_core.statistics_path())
FETCH_TYPE = C.CFUNCTYPE(
    C.c_int,
    C.c_size_t,
    C.POINTER(C.c_size_t),
    C.c_size_t,
    C.POINTER(C.c_char_p),
    C.POINTER(C.c_size_t),
    C.c_void_p,
)


@lru_cache(maxsize=8)
def _library(path):
    lib = C.CDLL(str(path))
    lib.sem_prefix_version.restype = C.c_int
    if lib.sem_prefix_version() != 1:
        raise ValueError("Unsupported native prefix ABI")
    lib.sem_prefix_error.restype = C.c_char_p
    signatures = {
        "sem_prefix_create": (
            [C.c_size_t, C.c_size_t, C.POINTER(C.c_size_t), C.c_size_t],
            C.c_void_p,
        ),
        "sem_prefix_add": (
            [C.c_void_p, C.POINTER(C.c_char_p), C.POINTER(C.c_size_t), C.c_size_t],
            C.c_int,
        ),
        "sem_prefix_json": ([C.c_void_p], C.c_char_p),
        "sem_prefix_destroy": ([C.c_void_p], None),
        "sem_adaptive_build": (
            [
                C.c_char_p,
                C.c_char_p,
                C.POINTER(C.c_size_t),
                C.c_size_t,
                FETCH_TYPE,
                C.c_void_p,
            ],
            C.c_void_p,
        ),
        "sem_adaptive_json": ([C.c_void_p], C.c_char_p),
        "sem_adaptive_destroy": ([C.c_void_p], None),
    }
    for name, (arguments, returns) in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = arguments, returns
    lib.sem_table_build.argtypes = lib.sem_adaptive_build.argtypes
    lib.sem_table_build.restype = C.c_void_p
    return lib


def adaptive_build(fields, config=None, library=DEFAULT_LIBRARY, *, table=False):
    """One native operation owns sampling, tuning and full-stream construction.

    Callback buffers remain alive until C++ requests the next batch. Replayable
    field sequences must remain immutable throughout the synchronous call.
    """
    config = config or {}
    names = sorted(fields)
    if not names:
        raise ValueError("Adaptive build requires at least one field")
    lib = _library(library)
    encoded, failures = {}, []

    @FETCH_TYPE
    def fetch(field, indices, count, strings, lengths, context):
        try:
            if field >= len(names) or count > 1024:
                raise ValueError("Invalid native source batch")
            values = fields[names[field]]
            encoded.clear()
            for i in range(count):
                value = values[indices[i]]
                if not isinstance(value, str):
                    raise TypeError("PrefixSummary inputs must be serialized strings")
                data = encoded.get(value)
                if data is None:
                    data = value.encode("utf-8")
                    encoded[value] = data
                strings[i], lengths[i] = data, len(data)
            return 0
        except Exception as exc:  # noqa: BLE001 — ctypes callbacks must capture exceptions
            failures.append(exc)
            return -1

    counts = (C.c_size_t * len(names))(*(len(fields[n]) for n in names))
    build_fields = lib.sem_table_build if table else lib.sem_adaptive_build
    result = build_fields(
        json.dumps(config, allow_nan=False).encode(),
        json.dumps(names, ensure_ascii=False).encode(),
        counts,
        len(names),
        fetch,
        None,
    )
    if not result:
        if failures:
            raise failures[0]
        raise ValueError(lib.sem_prefix_error().decode())
    try:
        serialized = lib.sem_adaptive_json(result)
        if serialized is None:
            raise ValueError(lib.sem_prefix_error().decode())
        return json.loads(serialized)
    finally:
        lib.sem_adaptive_destroy(result)


def build(strings, capacity, max_length, anchors, library=DEFAULT_LIBRARY):
    lib = _library(library)
    handle = lib.sem_prefix_create(
        capacity, max_length, (C.c_size_t * len(anchors))(*anchors), len(anchors)
    )

    def failure():
        return ValueError(lib.sem_prefix_error().decode())

    if not handle:
        raise failure()
    try:
        source = iter(strings)
        while True:
            values = list(itertools.islice(source, 1024))
            if not values:
                break
            if any(not isinstance(s, str) for s in values):
                raise ValueError("PrefixSummary inputs must be serialized strings")
            # encoded 必须存活至 C 调用结束；单独传字节长度以保留字符串内的 NUL。
            encoded = [s.encode("utf-8") for s in values]
            pointers = (C.c_char_p * len(encoded))(*encoded)
            lengths = (C.c_size_t * len(encoded))(*(len(s) for s in encoded))
            if lib.sem_prefix_add(handle, pointers, lengths, len(encoded)):
                raise failure()
        result = lib.sem_prefix_json(handle)
        if result is None:
            raise failure()
        return json.loads(result)
    finally:
        lib.sem_prefix_destroy(handle)
