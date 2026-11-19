"""Hash-encoded prefix sketches and per-source character summaries."""

import hashlib
import heapq
import re
import sys
from functools import lru_cache

HASH_VERSION = "sha256-prefix-v1"
ENTRY_ENCODING = "sha256"
SERIALIZER = "prompt-v4-json-unicode"
_PREFIX_HEX_RE = re.compile(r"[0-9a-f]{64}\Z")


def require_hash_entries(value):
    if value.get("entry_encoding") != ENTRY_ENCODING:
        raise ValueError("Prefix entries require sha256 encoding; rebuild statistics")


def decode_prefix_hash(value):
    if type(value) is not str or len(value) != 64 or _PREFIX_HEX_RE.match(value) is None:
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
        self.observed_max_length = None
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
        result = {
            "metadata": self.metadata,
            "count": self.count,
            "total_length": self.total_length,
            "truncated": self.truncated,
            "sketches": {str(k): s.json() for k, s in self.sketches.items()},
        }
        if self.observed_max_length is not None:
            result["observed_max_length"] = self.observed_max_length
        return result

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
        result.observed_max_length = value.get("observed_max_length")
        result.adaptive = meta.get("adaptive")
        result.sketches = {
            int(k): BottomKSketch.load(s)
            for k, s in sorted(value["sketches"].items(), key=lambda item: int(item[0]))
        }
        if list(result.sketches) != meta["anchors"]:
            raise ValueError("Serialized anchor keys do not match metadata")
        return result
