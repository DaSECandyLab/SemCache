"""Native summary representation, reuse estimates and persistent history.

Character-domain sketches describe reuse potential, never physical KV residency.
All sources share SHA-256 and exact serializer/version metadata.
"""

import hashlib
import heapq
import itertools
import json
import math
import re
import sqlite3
import sys
import time
import uuid
from functools import lru_cache, wraps
from threading import RLock
from typing import Any

from planning.state import canonical

HASH_VERSION = "sha256-prefix-v1"
ENTRY_ENCODING = "sha256"
SERIALIZER = "prompt-v4-json-unicode"


def require_hash_entries(value):
    if value.get("entry_encoding") != ENTRY_ENCODING:
        raise ValueError("Prefix entries require sha256 encoding; rebuild statistics")


def decode_prefix_hash(value):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Prefix hash must have 64 lowercase hexadecimal digits")
    return int(value, 16)


def _hash_rank(prefix):
    return int.from_bytes(hashlib.sha256(prefix.encode()).digest(), "big")


_cached_rank = lru_cache(maxsize=8192)(_hash_rank)


def rank(prefix):
    # Bound retained text as well as entry count; mutable subclasses bypass reuse.
    if type(prefix) is str and len(prefix) <= 1024:
        return _cached_rank(prefix)
    return _hash_rank(prefix)


class BottomKSketch:
    def __init__(self, capacity=128):
        if type(capacity) is not int or not 2 <= capacity <= sys.maxsize:
            raise ValueError("Sketch capacity must be at least 2")
        self.capacity = capacity
        self.entries = {}
        self.heap = []
        self.complete = True

    def add(self, prefix, count=1):
        self._add_hash(rank(prefix), count)

    def _add_hash(self, h, count):
        if type(h) is not int or not 0 <= h < 2**256:
            raise ValueError("Prefix hash must be a 256-bit nonnegative integer")
        if type(count) is not int or count < 1:
            raise ValueError("Frequency must be positive")
        if h in self.entries:
            self.entries[h] += count
            return
        if len(self.entries) < self.capacity:
            self.entries[h] = count
            heapq.heappush(self.heap, -h)
        else:
            self.complete = False
            largest = -self.heap[0]
            if h < largest:
                heapq.heapreplace(self.heap, -h)
                del self.entries[largest]
                self.entries[h] = count

    def sample(self):
        if self.complete:
            return 1.0, self.entries
        # 第 k 小哈希只定义抽样阈值，不计入样本；估计器使用严格小于阈值的项。
        threshold = -self.heap[0]
        tau = threshold / (2**256)
        return tau, {h: c for h, c in self.entries.items() if h < threshold}

    def merge(self, other):
        sampled = [s.capacity for s in (self, other) if not s.complete]
        capacity = (
            min(sampled)
            if sampled
            else max(
                self.capacity,
                other.capacity,
                len(self.entries.keys() | other.entries.keys()),
            )
        )
        result = BottomKSketch(capacity)
        # Summed counts are exact for prefixes retained below the final threshold;
        # discarded prefixes can never re-enter as thresholds only decrease.
        for source in (self, other):
            for h, count in source.entries.items():
                result._add_hash(h, count)
        result.complete &= self.complete and other.complete
        return result

    def json(self):
        return {
            "capacity": self.capacity,
            "complete": self.complete,
            "entry_encoding": ENTRY_ENCODING,
            "entries": {f"{h:064x}": count for h, count in self.entries.items()},
        }

    @classmethod
    def load(cls, value):
        require_hash_entries(value)
        result = cls(value["capacity"])
        entries = {decode_prefix_hash(h): count for h, count in value["entries"].items()}
        if len(entries) <= result.capacity:
            if any(type(count) is not int or count < 1 for count in entries.values()):
                raise ValueError("Frequency must be positive")
            # Serialized sketches already contain the retained set. Restore it
            # independently without replaying every insertion into the heap.
            result.entries = dict(entries)
            result.heap = [-h for h in entries]
            heapq.heapify(result.heap)
        else:
            for h, count in entries.items():
                result._add_hash(h, count)
        result.complete = value["complete"]
        return result


class PrefixSummary:
    def __init__(
        self, capacity=128, max_length=4096, anchors=None, serializer=SERIALIZER
    ):
        if type(max_length) is not int or not 1 <= max_length <= sys.maxsize:
            raise ValueError("Maximum tracked length must be positive")
        self.capacity, self.max_length, self.serializer = (
            capacity,
            max_length,
            serializer,
        )
        if anchors is None:
            anchors = sorted(
                {1, max_length}
                | {2**k for k in range(max_length.bit_length()) if 2**k <= max_length}
            )
        if (
            not isinstance(anchors, (list, tuple))
            or not anchors
            or any(type(k) is not int or not 1 <= k <= max_length for k in anchors)
            or list(anchors) != sorted(set(anchors))
            or anchors[0] != 1
            or anchors[-1] != max_length
        ):
            raise ValueError("Anchors must increase from 1 through the maximum length")
        self.sketches = {k: BottomKSketch(capacity) for k in anchors}
        self.count = self.total_length = self.truncated = 0
        self.adaptive = None

    @classmethod
    def build(cls, strings, capacity=128, max_length=4096, anchors=None):
        """Construct with the native library; errors are never silently bypassed."""
        from planning.kernels.sketches import build

        coordinates = cls(capacity, max_length, anchors)
        return cls.load(
            build(strings, capacity, max_length, list(coordinates.sketches))
        )

    @property
    def metadata(self):
        result = {
            "hash_version": HASH_VERSION,
            "serializer": self.serializer,
            "domain": "characters",
            "capacity": self.capacity,
            "max_length": self.max_length,
            "anchors": list(self.sketches),
        }
        if self.adaptive is not None:
            result["adaptive"] = self.adaptive
        return result

    def compatible(self, other):
        return self.serializer == other.serializer and bool(
            self.sketches.keys() & other.sketches.keys()
        )

    def json(self):
        return {
            "metadata": self.metadata,
            "count": self.count,
            "total_length": self.total_length,
            "truncated": self.truncated,
            "sketches": {str(k): s.json() for k, s in self.sketches.items()},
        }

    @classmethod
    def load(cls, value):
        meta = value["metadata"]
        if meta["hash_version"] != HASH_VERSION or meta["domain"] != "characters":
            raise ValueError("Unsupported sketch version/domain")
        result = cls(
            meta["capacity"], meta["max_length"], meta["anchors"], meta["serializer"]
        )
        result.count, result.total_length, result.truncated = (
            value[k] for k in ("count", "total_length", "truncated")
        )
        result.adaptive = meta.get("adaptive")
        result.sketches = {
            int(k): BottomKSketch.load(s)
            for k, s in sorted(value["sketches"].items(), key=lambda item: int(item[0]))
        }
        if list(result.sketches) != meta["anchors"]:
            raise ValueError("Serialized anchor keys do not match metadata")
        return result


def present(alpha, count):
    if not 0 <= alpha <= 1:
        raise ValueError("Survival probability must be in [0,1]")
    return 1.0 if alpha == 1 else -math.expm1(count * math.log1p(-alpha))


def repeated_prefix_hits(alpha, count):
    """§3.1.1 binomial repeats, without cancelling a small positive result."""
    if not alpha or count <= 1:
        return 0.0
    if alpha == 1:
        return count - 1.0
    visits = alpha * count
    if visits >= 0.125:
        return max(0.0, visits + math.expm1(count * math.log1p(-alpha)))
    term = visits * ((count - 1) * alpha) / 2
    total, k = term, 2
    while term:
        term *= -((count - k) * alpha) / (k + 1)
        updated = total + term
        if updated == total:
            break
        total, k = updated, k + 1
    return max(0.0, total)


def integrate(values):
    ordered = sorted(values)
    total = values[ordered[0]]
    for left, right in itertools.pairwise(ordered):
        gap = right - left
        total += (gap - 1) * values[left] / 2 + (gap + 1) * values[right] / 2
    return total


def survival_rate(alpha, depth=None):
    if isinstance(alpha, dict):
        value = (
            alpha["total"]
            if depth is None
            else alpha.get("depths", {}).get(str(depth), alpha["total"])
        )
    else:
        value = alpha
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Survival probability must be finite and in [0,1]")
    if 0 <= value <= 1:
        return value
    # Ratios of independently rounded weighted sums can land on the adjacent
    # float outside an endpoint. Preserve that valid endpoint, not an unknown
    # or an invented zero; genuinely invalid probabilities still fail.
    if math.nextafter(0.0, -math.inf) <= value <= math.nextafter(1.0, math.inf):
        return min(1.0, max(0.0, value))
    raise ValueError("Survival probability must be finite and in [0,1]")


def self_reuse(
    summary, alpha: float | dict[str, Any] = 1.0, *, requests=None
) -> float | None:
    rate = survival_rate(alpha)
    if rate is None:
        return None if summary.count else 0.0
    distinct = rate * summary.count
    denominator = distinct if requests is None else requests
    if denominator == 0:
        return 0.0
    if not distinct:
        return None
    scale = denominator / distinct
    values = {}
    for depth, sketch in summary.sketches.items():
        rate = survival_rate(alpha, depth)
        tau, sample = sketch.sample()
        if rate is None:
            if sample:
                return None
            rate = 0.0
        values[depth] = (
            sum(
                max(0.0, repeated_prefix_hits(rate, c) + (scale - 1) * rate * c)
                for c in sample.values()
            )
            / tau
        )
    return integrate(values) / denominator


def _cross_sample(left, right):
    if (
        left.complete is True
        and right.complete is True
        and all(
            type(sketch.capacity) is int and 2 <= sketch.capacity <= sys.maxsize
            for sketch in (left, right)
        )
        and all(
            type(prefix) is int
            and 0 <= prefix < 2**256
            and type(count) is int
            and count > 0
            for sketch in (left, right)
            for prefix, count in sketch.entries.items()
        )
    ):
        # A complete union has tau=1; mutable or invalid entries use merge validation.
        tau, sample = 1.0, {**left.entries, **right.entries}
    else:
        tau, sample = left.merge(right).sample()
    common = set(sample) & set(left.entries) & set(right.entries)
    return tau, common


def joint_presence(source, target, source_count, target_count, common_count):
    """Nested same-row survivals, with independent rows as an explicit approximation.

    Each depth supplies its own rates. Common rows are measured by shared row
    identity, never inferred from hash-bucket overlap.
    """
    if not 0 <= common_count <= min(source_count, target_count):
        raise ValueError("Common frequency must be between zero and both marginals")
    a, b = present(source, source_count), present(target, target_count)
    correction = (
        (1.0 - a)
        * (
            1.0
            if target_count == common_count
            else 1.0 - present(target, target_count - common_count)
        )
        * (present(target, common_count) if common_count else 0.0)
    )
    return min(a, b, max(max(0.0, a + b - 1.0), a * b + correction))


def cross_reuse(
    source,
    target,
    alpha_source: float | dict[str, Any] = 1.0,
    alpha_target: float | dict[str, Any] = 1.0,
    *,
    coupling=None,
    requests=None,
    _terms=None,
    _reduce=None,
) -> float | None:
    if not source.compatible(target):
        raise ValueError("Cross-source summaries must use compatible coordinates")
    rate = survival_rate(alpha_target)
    count = target.count
    if rate is None:
        return None if count else 0.0
    denominator = rate * count if requests is None else requests
    if denominator == 0:
        return 0.0
    values = {}
    if coupling and coupling.get("kind") == "same_field":
        for depth, sketch in target.sketches.items():
            tau, sample = sketch.sample()
            alpha = survival_rate(alpha_target, depth)
            if alpha is None and sample:
                return None
            values[depth] = sum(present(alpha or 0.0, c) for c in sample.values()) / tau
        return integrate(values) / denominator
    for depth in sorted(source.sketches.keys() & target.sketches.keys()):
        left, right = source.sketches[depth], target.sketches[depth]
        counts, total = None, 0.0
        if _terms is None or coupling:
            tau, common = _cross_sample(left, right)
        else:
            tau, counts = _terms(source, target, depth)
            common = counts
        rate_source, rate_target = (
            survival_rate(alpha_source, depth),
            survival_rate(alpha_target, depth),
        )
        if rate_source is None or rate_target is None:
            if common:
                return None
            values[depth] = 0.0
            continue
        if counts is None:
            overlap = {}
            if coupling and coupling.get("kind") == "same_relation":
                raw_common = coupling["common"]
                require_hash_entries(raw_common)
                overlap = {
                    decode_prefix_hash(h): c
                    for h, c in raw_common.get(str(depth), {}).items()
                }
            total = sum(
                joint_presence(
                    rate_source,
                    rate_target,
                    left.entries[p],
                    right.entries[p],
                    overlap.get(p, 0.0),
                )
                if coupling and coupling.get("kind") == "same_relation"
                else present(rate_source, left.entries[p])
                * present(rate_target, right.entries[p])
                for p in common
            )
        elif _reduce is None:
            total = sum(
                present(rate_source, source_count) * present(rate_target, target_count)
                for source_count, target_count in counts
            )
        values[depth] = (
            _reduce(counts, rate_source, rate_target, tau)
            if _reduce is not None and counts is not None
            else total / tau
        )
    return integrate(values) / denominator


def join_self(
    left,
    right,
    left_total_length=None,
    alpha_left: float | dict[str, Any] = 1.0,
    alpha_right: float | dict[str, Any] = 1.0,
    *,
    left_requests=None,
    right_requests=None,
) -> float | None:
    left_rate, right_rate = survival_rate(alpha_left), survival_rate(alpha_right)
    if left_rate == 0 or right_rate == 0 or not left.count or not right.count:
        return 0.0
    if left_rate is None or right_rate is None:
        return None
    left_count = left.count * left_rate if left_requests is None else left_requests
    right_count = right.count * right_rate if right_requests is None else right_requests
    if not left_count or not right_count:
        return 0.0
    mean_left = (
        left.total_length if left_total_length is None else left_total_length
    ) / left.count
    left_reuse, right_reuse = (
        self_reuse(left, alpha_left, requests=left_requests),
        self_reuse(right, alpha_right, requests=right_requests),
    )
    if left_reuse is None or right_reuse is None:
        return None
    return (
        left_reuse / right_count
        + max(0, right_count - 1) / right_count * mean_left
        + right_reuse
    )


def synchronized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)

    return call


class CacheHistory:
    """Persistent per-model bounded directory of actually executed prefixes.

    This is a research history catalogue, not a shadow KV eviction simulator.
    Updates require actual telemetry; speculative search never writes it.
    """

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __init__(self, path, capacity=256):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("CacheHistory capacity must be positive")
        self.capacity = capacity
        # The embedded socket service runs on a different thread from its owner.
        # Serialize whole transactions, including read/modify/write counters.
        self.lock = RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS history (key TEXT PRIMARY KEY, model TEXT NOT NULL, pre TEXT NOT NULL, "
            "operator TEXT NOT NULL, field TEXT NOT NULL, updated REAL NOT NULL, summary TEXT NOT NULL, "
            "requests_before INTEGER NOT NULL, characters_before INTEGER NOT NULL)"
        )
        columns = {r[1] for r in self.db.execute("PRAGMA table_info(history)")}
        if not {"requests_before", "characters_before"} <= columns:
            self.db.close()
            raise ValueError(
                "Obsolete cache-history schema; use a new history database"
            )
        with self.db:
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS model_load (model TEXT PRIMARY KEY, requests INTEGER NOT NULL, characters INTEGER NOT NULL)"
            )
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS history_source ON history(model,pre)"
            )

    @synchronized
    def load(self, model):
        return self.db.execute(
            "SELECT requests,characters FROM model_load WHERE model=?", (model,)
        ).fetchone() or (0, 0)

    @synchronized
    def advance_load(self, model, requests, characters):
        if any(type(v) is not int or v < 0 for v in (requests, characters)):
            raise ValueError("Model load counters must be nonnegative integers")
        before = self.load(model)
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO model_load VALUES (?,?,?)",
                (model, before[0] + requests, before[1] + characters),
            )

    @synchronized
    def record(
        self,
        model,
        pre,
        operator,
        field,
        summary,
        now=None,
        *,
        requests=None,
        characters=None,
        requests_before=None,
        characters_before=None,
        update_load=True,
    ):
        now = time.time() if now is None else now
        requests = summary.count if requests is None else requests
        characters = summary.total_length if characters is None else characters
        if any(type(x) is not int or x < 0 for x in (requests, characters)):
            raise ValueError("Model load counters must be nonnegative integers")
        key = uuid.uuid4().hex
        with self.db:
            before = self.db.execute(
                "SELECT requests,characters FROM model_load WHERE model=?", (model,)
            ).fetchone() or (0, 0)
            self.db.execute(
                "INSERT INTO history "
                "(key,model,pre,operator,field,updated,summary,requests_before,characters_before) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    key,
                    model,
                    pre,
                    str(operator),
                    field,
                    now,
                    canonical(summary.json()),
                    before[0] if requests_before is None else requests_before,
                    before[1] if characters_before is None else characters_before,
                ),
            )
            current = (
                (before[0] + requests, before[1] + characters)
                if update_load
                else before
            )
            if update_load:
                self.db.execute(
                    "INSERT OR REPLACE INTO model_load VALUES (?,?,?)",
                    (model, *current),
                )
            count = self.db.execute(
                "SELECT COUNT(*) FROM history WHERE model=?", (model,)
            ).fetchone()[0]
            if count > self.capacity:
                rows = self.db.execute(
                    "SELECT key,updated,requests_before,characters_before FROM history WHERE model=?",
                    (model,),
                ).fetchall()
                observed_now = max(now, max(r[1] for r in rows))
                distances = [
                    (
                        k,
                        t,
                        [
                            max(0, observed_now - t),
                            None if n is None else max(0, current[0] - n),
                            None
                            if characters is None
                            else max(0, current[1] - characters),
                        ],
                    )
                    for k, t, n, characters in rows
                ]
                scales = [
                    max((d[i] or 0 for _, _, d in distances), default=0)
                    for i in range(3)
                ]
                ranked = sorted(
                    distances,
                    key=lambda r: (
                        sum((v or 0) / s if s else 0 for v, s in zip(r[2], scales)),
                        -r[1],
                        r[0],
                    ),
                    reverse=True,
                )
                self.db.executemany(
                    "DELETE FROM history WHERE key=?",
                    [(r[0],) for r in ranked[: count - self.capacity]],
                )
        return key

    @synchronized
    def close(self):
        self.db.close()

    @synchronized
    def clear(self):
        """Reset the request catalogue together with an explicit KV-cache reset."""
        with self.db:
            self.db.execute("DELETE FROM history")
            self.db.execute("DELETE FROM model_load")

    @synchronized
    def snapshot(self, now=None, *, model=None, pre=None):
        """Read bounded source entries without claiming physical cache residency."""
        now = time.time() if now is None else now
        loads = {
            model: (requests, characters)
            for model, requests, characters in self.db.execute(
                "SELECT * FROM model_load"
            )
        }
        query, parameters = "SELECT * FROM history", []
        if model is not None:
            query += " WHERE model=?"
            parameters.append(model)
            if pre is not None:
                query += " AND pre=?"
                parameters.append(pre)
        query += " ORDER BY updated,key"
        entries = []
        for (
            key,
            source_model,
            source_pre,
            op,
            field,
            updated,
            raw,
            n,
            characters,
        ) in self.db.execute(query, parameters):
            summary = json.loads(raw)
            current_requests, current_characters = loads.get(
                source_model, (n, characters)
            )
            entries.append(
                {
                    "id": key,
                    "model_id": source_model,
                    "pre": source_pre,
                    "operator": op,
                    "first_field": field,
                    "age_seconds": max(0.0, now - updated),
                    "started_at": updated,
                    "requests_before": n,
                    "characters_before": characters,
                    "requests_since": None if n is None else current_requests - n,
                    "characters_since": None
                    if characters is None
                    else current_characters - characters,
                    "summary": summary,
                    "capacity": summary["metadata"]["capacity"],
                }
            )
        return entries
