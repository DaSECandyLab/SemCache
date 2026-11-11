"""Stream sent-request prefixes into bounded bottom-k history in the optimizer."""

import ctypes as C
import json
from pathlib import Path

from planning.execution.adaptation_evidence import RuntimeInputEvidence
from planning.observations.cache import PrefixSummary
from planning.observations.statistics import FIELD_SERIALIZER, prefix
from planning.state import digest


class RuntimeHistoryReader:
    def __init__(self):
        self.offset = 0
        self.path = None
        self.recorded = set()
        self.inputs = RuntimeInputEvidence()

    def close(self):
        self.inputs.close()

    def statistics(self, entries):
        from planning.kernels.sketches import DEFAULT_LIBRARY, _library

        return self.inputs.snapshot(entries, _library(DEFAULT_LIBRARY))

    def consume(self, request, query, history):
        from planning.kernels.sketches import DEFAULT_LIBRARY, _library

        path = Path(request["telemetry_path"]).resolve(strict=True)
        if self.path is not None and self.path != path:
            raise ValueError("Runtime telemetry path changed within a query")
        self.path = path
        lib = _library(DEFAULT_LIBRARY)
        ops = {o["operator_id"]: o for o in query["semantic_operators"]}
        groups = {}
        try:
            with path.open("rb") as stream:
                stream.seek(self.offset)
                while True:
                    position = stream.tell()
                    line = stream.readline()
                    if (
                        not line
                        or not line.endswith(b"\n")
                        or stream.tell() > request.get("telemetry_bytes", float("inf"))
                    ):
                        stream.seek(position)
                        break
                    row = json.loads(line)
                    if row.get("query_id") != query["query_id"]:
                        continue
                    if "telemetry_bytes" not in request and (
                        row.get("event") == "runtime_chunk_completed"
                        and row["chunk_index"] >= request["completed_chunks"]
                    ):
                        stream.seek(position)
                        break
                    if (
                        row.get("event")
                        not in {"semantic_row", "semantic_result_cache_hit"}
                        or row.get("parse_status") != "validated"
                    ):
                        continue
                    oid = row["operator_id"]
                    op = ops[oid]
                    self.inputs.add(op, row, lib)
                    if row["event"] != "semantic_row":
                        continue
                    action = {
                        "operator_id": oid,
                        "model_id": row["model_id"],
                        "layout": {k: row[k] for k in op["default_layout"]},
                    }
                    key = oid, row.get("config_generation", 0), digest(action)
                    if key not in groups:
                        coordinates = PrefixSummary(
                            128, 4096, serializer=FIELD_SERIALIZER
                        )
                        anchors = list(coordinates.sketches)
                        handle = lib.sem_prefix_create(
                            128,
                            4096,
                            (C.c_size_t * len(anchors))(*anchors),
                            len(anchors),
                        )
                        if not handle:
                            raise ValueError(lib.sem_prefix_error().decode())
                        groups[key] = [handle, action, row, 0, 0]
                    group = groups[key]
                    names = (
                        op["input_fields"]
                        or op["left_input_fields"] + op["right_input_fields"]
                    )
                    first = action["layout"].get(
                        "field_order", action["layout"].get("left_field_order")
                    )[0]
                    encoded = row["input_values"][names.index(first)].encode("utf-8")
                    if lib.sem_prefix_add(
                        group[0],
                        (C.c_char_p * 1)(encoded),
                        (C.c_size_t * 1)(len(encoded)),
                        1,
                    ):
                        raise ValueError(lib.sem_prefix_error().decode())
                    group[3] += 1
                    group[4] += len(row["system_text"]) + len(row["user_text"])
                end = stream.tell()
            for key, (handle, action, first, count, characters) in groups.items():
                identity = self.offset, end, key
                if identity in self.recorded:
                    continue
                encoded = lib.sem_prefix_json(handle)
                if not encoded:
                    raise ValueError(lib.sem_prefix_error().decode())
                summary = PrefixSummary.load(json.loads(encoded))
                summary.serializer = FIELD_SERIALIZER
                op = ops[action["operator_id"]]
                field = action["layout"].get(
                    "field_order", action["layout"].get("left_field_order")
                )[0]
                history.record(
                    action["model_id"],
                    prefix(
                        op,
                        action,
                        first["system_text"],
                        first["operator_specification"],
                    ),
                    digest([op["kind"], op["task"], op["output_schema"]]),
                    field,
                    summary,
                    now=first.get("observed_unix_seconds"),
                    requests=count,
                    characters=characters,
                )
                self.recorded.add(identity)
            self.offset = end
            return sum(g[3] for g in groups.values())
        finally:
            for group in groups.values():
                lib.sem_prefix_destroy(group[0])
