"""Independent, persistent Teacher observations for one native system.

First occurrence: one call. Second: verify, with one final vote if needed.
Confirmed entries reuse the chosen response. No approximate prompt matching.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import threading
import time
from collections import Counter
from pathlib import Path


def canonical_response(response):
    choice = response["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("Incomplete Teacher response cannot enter the cache")
    content = choice["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Empty Teacher response cannot enter the cache")
    try:
        value = json.loads(content)
        return "json:" + json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (json.JSONDecodeError, ValueError):
        return "text:" + content.strip()


class TeacherCache:
    def __init__(self, directory, system, *, max_entries=32768, max_bytes=268435456):
        self.system = system
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(
            self.directory / "responses.sqlite3",
            check_same_thread=False,
            isolation_level=None,
        )
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS entries (key TEXT PRIMARY KEY, data TEXT NOT NULL, size INTEGER NOT NULL, used REAL NOT NULL)"
        )
        self.db.execute("CREATE INDEX IF NOT EXISTS cache_lru ON entries(used)")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS imported (identity TEXT PRIMARY KEY)"
        )
        self.count, self.bytes = self.db.execute(
            "SELECT count(*),coalesce(sum(size),0) FROM entries"
        ).fetchone()
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self.lock = threading.RLock()
        self.key_locks = {}

    def key(self, endpoint, forwarded):
        value = {
            "version": "native-system-teacher-cache-v1",
            "system": self.system,
            "endpoint": endpoint.rstrip("/"),
            "request": forwarded,
        }
        return hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def key_lock(self, key):
        with self.lock:
            return self.key_locks.setdefault(key, threading.Lock())

    def _read(self, key):
        with self.lock:
            row = self.db.execute(
                "SELECT data FROM entries WHERE key=?", (key,)
            ).fetchone()
            if row:
                self.db.execute(
                    "UPDATE entries SET used=? WHERE key=?", (time.time(), key)
                )
                return json.loads(row[0])
        return {"responses": [], "values": [], "confirmed": False, "winner": 0}

    def _write(self, key, entry):
        text = json.dumps(
            entry, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        size = len(text.encode())
        with self.lock:
            old = self.db.execute(
                "SELECT size FROM entries WHERE key=?", (key,)
            ).fetchone()
            self.db.execute(
                "INSERT INTO entries VALUES (?,?,?,?) ON CONFLICT(key) DO UPDATE SET data=excluded.data,size=excluded.size,used=excluded.used",
                (key, text, size, time.time()),
            )
            self.count += int(old is None)
            self.bytes += size - (old[0] if old else 0)
            while self.count > self.max_entries or self.bytes > self.max_bytes:
                victim, victim_size = self.db.execute(
                    "SELECT key,size FROM entries ORDER BY used LIMIT 1"
                ).fetchone()
                self.db.execute("DELETE FROM entries WHERE key=?", (victim,))
                self.count -= 1
                self.bytes -= victim_size

    @staticmethod
    def _observe(entry, response, canonical):
        entry["responses"].append(copy.deepcopy(response))
        entry["values"].append(canonical)
        n = len(entry["values"])
        if n >= 2 and entry["values"][0] == entry["values"][1] or n == 3:
            counts = Counter(entry["values"])
            largest = max(counts.values())
            # Preserve a deterministic complete answer even when all three
            # free-text variants differ; disclose the tie instead of inventing consensus.
            entry["winner"] = max(
                i for i, value in enumerate(entry["values"]) if counts[value] == largest
            )
            entry["vote_tie"] = largest < 2
            entry["confirmed"] = True

    def resolve(self, key, infer, normalize=lambda response: response):
        with self.key_lock(key):
            entry = self._read(key)
            info = {
                "key": key,
                "system": self.system,
                "hit": False,
                "verification_calls": 0,
                "vote_calls": 0,
            }
            if entry["confirmed"]:
                info.update(
                    hit=True,
                    observations=len(entry["responses"]),
                    vote_tie=entry.get("vote_tie", False),
                )
                return copy.deepcopy(entry["responses"][entry["winner"]]), info
            cold = not entry["responses"]
            while not entry["confirmed"]:
                before = len(entry["responses"])
                if before == 1:
                    info["verification_calls"] += 1
                elif before == 2:
                    info["vote_calls"] += 1
                response = infer()
                value = canonical_response(normalize(response))
                self._observe(entry, response, value)
                self._write(key, entry)
                if cold:
                    break
            winner = entry["winner"] if entry["confirmed"] else 0
            info.update(
                observations=len(entry["responses"]),
                vote_tie=entry.get("vote_tie", False),
            )
            return copy.deepcopy(entry["responses"][winner]), info

    def seed(self, identity, key, response, normalize=lambda response: response):
        value = canonical_response(normalize(response))
        with self.key_lock(key), self.lock:
            if self.db.execute(
                "SELECT 1 FROM imported WHERE identity=?", (identity,)
            ).fetchone():
                return False
            self.db.execute("BEGIN IMMEDIATE")
            try:
                entry = self._read(key)
                if not entry["confirmed"]:
                    self._observe(entry, response, value)
                    self._write(key, entry)
                self.db.execute("INSERT INTO imported VALUES (?)", (identity,))
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                self.count, self.bytes = self.db.execute(
                    "SELECT count(*),coalesce(sum(size),0) FROM entries"
                ).fetchone()
                raise
        return True

    def close(self):
        with self.lock:
            self.db.close()
