"""Read-through cache for completed initial probes, never execution history."""

import json
import math
import os
import tempfile
from pathlib import Path

from planning.state import digest


def valid_probe(payload, contract):
    if not isinstance(payload, dict) or digest(payload.get("contract")) != digest(
        contract
    ):
        return False
    try:
        if "layouts" in contract:
            if set(payload["layouts"]) != set(contract["layouts"]):
                return False
            for layout in payload["layouts"].values():
                if set(layout["models"]) != set(contract["models"]):
                    return False
                for item in layout["models"].values():
                    if not (
                        0 <= item["value"] <= 1
                        and item["total"] > 0
                        and item["samples"] > 0
                        and math.isfinite(item["seconds_per_request"])
                        and item["seconds_per_request"] >= 0
                        and all(
                            key in item
                            for key in (
                                "correct",
                                "weighted_correct",
                                "weighted_total",
                                "weighted_sum_squares",
                                "effective_samples",
                                "judge_model",
                                "basis",
                                "latency_cv",
                                "latency_m2",
                                "selectivity",
                            )
                        )
                    ):
                        return False
            return True
        return bool(payload["profile"]) and all(
            math.isfinite(item["seconds_per_input"]) and item["seconds_per_input"] >= 0
            for item in payload["profile"].values()
        )
    except (KeyError, TypeError, AttributeError, ValueError):
        return False


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix=".probe-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_probe(path, contract, shared_cache_dir=None):
    path = Path(path)
    candidates = [path]
    if shared_cache_dir is not None:
        candidates.append(Path(shared_cache_dir) / path.name)
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text())
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not valid_probe(payload, contract):
            continue
        if candidate != path:
            _write(path, payload)
        elif shared_cache_dir is not None:
            _write(Path(shared_cache_dir) / path.name, payload)
        return payload
    return None


def store_probe(path, payload, shared_cache_dir=None):
    if not valid_probe(payload, payload.get("contract", {})):
        raise ValueError("Incomplete initial calibration cannot be cached")
    path = Path(path)
    _write(path, payload)
    if shared_cache_dir is not None:
        _write(Path(shared_cache_dir) / path.name, payload)
