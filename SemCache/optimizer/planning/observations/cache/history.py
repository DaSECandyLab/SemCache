"""Persistent per-model bounded directory of executed prefix summaries."""

import json
import sqlite3
import time
import uuid
from functools import wraps
from threading import RLock

from planning.state import canonical


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
