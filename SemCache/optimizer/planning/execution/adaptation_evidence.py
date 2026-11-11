"""Bounded sketches and field moments over actual consumed request inputs."""

import ctypes as C
import hashlib
import json
from collections import Counter

from planning.observations.cache import PrefixSummary
from planning.observations.statistics import FIELD_SERIALIZER


class RuntimeInputEvidence:
    def __init__(self):
        self.operators = {}

    def add(self, op, row, lib):
        oid, generation = op["operator_id"], row.get("config_generation", 0)
        current = self.operators.get(oid)
        if current is not None and current["generation"] != generation:
            self._destroy(current, lib)
            current = None
        if current is None:
            current = self.operators[oid] = {
                "generation": generation,
                "fields": {},
                "requests": 0,
            }
        names = op["input_fields"] or op["left_input_fields"] + op["right_input_fields"]
        schema = {
            f["name"]: f.get("value_key", f["name"])
            for f in op.get("input_field_schema", [])
        }
        current["requests"] += 1
        for name, value in zip(names, row["input_values"], strict=True):
            key = schema.get(name, name)
            bank = current["fields"].get(key)
            if bank is None:
                anchors = list(PrefixSummary(128, 4096).sketches)
                handle = lib.sem_prefix_create(
                    128, 4096, (C.c_size_t * len(anchors))(*anchors), len(anchors)
                )
                if not handle:
                    raise ValueError(lib.sem_prefix_error().decode())
                bank = current["fields"][key] = {
                    "handle": handle,
                    "count": 0,
                    "characters": 0,
                    "bytes": 0,
                    "nulls": 0,
                    "values": Counter(),
                    "overflow": False,
                }
            encoded = value.encode("utf-8")
            if lib.sem_prefix_add(
                bank["handle"],
                (C.c_char_p * 1)(encoded),
                (C.c_size_t * 1)(len(encoded)),
                1,
            ):
                raise ValueError(lib.sem_prefix_error().decode())
            bank["count"] += 1
            bank["characters"] += len(value)
            bank["bytes"] += len(encoded)
            bank["nulls"] += value == "null"
            identity = hashlib.blake2b(encoded, digest_size=8).digest()
            if identity in bank["values"] or len(bank["values"]) < 4096:
                bank["values"][identity] += 1
            else:
                bank["overflow"] = True

    def snapshot(self, entries, lib):
        result = {}
        for oid, current in self.operators.items():
            if current["generation"] != entries[oid]["current"]["config_generation"]:
                continue
            summaries, moments = {}, {}
            for name, bank in current["fields"].items():
                raw = lib.sem_prefix_json(bank["handle"])
                if not raw:
                    raise ValueError(lib.sem_prefix_error().decode())
                summary = json.loads(raw)
                summary["metadata"]["serializer"] = FIELD_SERIALIZER
                summaries[name] = {"summary": summary}
                count = bank["count"]
                moments[name] = {
                    "average_length": bank["characters"] / count,
                    "average_bytes": bank["bytes"] / count,
                    "null_fraction": bank["nulls"] / count,
                    "sample_effective_samples": count,
                    "sample_distinct_fraction": len(bank["values"]) / count
                    if not bank["overflow"]
                    else None,
                    "sample_top_frequency": max(bank["values"].values()) / count
                    if not bank["overflow"]
                    else None,
                    "basis": "actual consumed execution request inputs",
                }
            result[str(oid)] = {
                "input_field_summaries": summaries,
                "input_field_statistics": moments,
                "observed_requests": current["requests"],
                "config_generation": current["generation"],
                "basis": "actual consumed execution request inputs",
            }
        return result

    @staticmethod
    def _destroy(current, lib):
        for bank in current["fields"].values():
            lib.sem_prefix_destroy(bank["handle"])

    def close(self):
        if self.operators:
            from planning.kernels.sketches import DEFAULT_LIBRARY, _library

            lib = _library(DEFAULT_LIBRARY)
            for current in self.operators.values():
                self._destroy(current, lib)
            self.operators.clear()
