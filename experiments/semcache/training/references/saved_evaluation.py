"""Read completed heldout answers and certified aggregate captures without APIs."""

import hashlib
from pathlib import Path

from planning.settings.configuration import ROOT
from reference_answers.sql.aggregate import capture_plan
from reference_answers.storage.files import file_digest

from training.artifacts import read


def configured_path(value):
    return (ROOT / value).resolve() if value else None


class SavedEvaluationReferences:
    """Fail closed on changed inputs; historical planner/teacher remain provenance."""

    def __init__(self, source, manifest, databases, *, aggregate_manifest=None):
        self.root = configured_path(source)
        request = read(self.root / "request.json")
        results = read(self.root / "results.json")
        status = read(self.root / "status.json")
        rows = results["queries"]
        frozen = request["manifest"]
        self.items = {item["id"]: item for item in frozen}
        self.rows = {row["query"]: row for row in rows}
        if (
            status.get("status") != "completed"
            or len(rows) != len(self.rows)
            or len(frozen) != len(self.items)
            or set(self.rows) != set(self.items)
            or results.get("total_queries") != len(rows)
            or results.get("completed") != len(rows)
            or any(row.get("status") != "completed" for row in rows)
        ):
            raise ValueError(
                "Saved reference run must contain completed unique queries"
            )
        self.hashes = {}
        self.databases = {
            name: Path(path).resolve() for name, path in databases.items()
        }
        for name in ("request.json", "results.json", "status.json"):
            self._remember(self.root / name)
        self.database_hashes = {}
        for name, path in self.databases.items():
            if (
                name not in request["databases"]
                or Path(request["databases"][name]).resolve() != path
                or self._remember(path) != request["database_sha256"].get(name)
            ):
                raise ValueError(f"Saved reference original database changed: {name}")
            self.database_hashes[name] = self.hashes[str(path)]
        self.replay = None
        if aggregate_manifest:
            replay_path = configured_path(aggregate_manifest)
            self._remember(replay_path)
            self.replay_root = replay_path.parent
            self.replay = read(replay_path)
            if (
                self.replay.get("status") != "completed"
                or self.replay.get("remote_model_calls") != 0
                or Path(self.replay["source_run"]).resolve() != self.root
            ):
                raise ValueError(
                    "Aggregate reference manifest is not a completed offline recovery"
                )
        self.references = {}
        for item in manifest:
            self._load(item)
        # Retain only the historical identity, never credentials or current API settings.
        historical = request.get("teacher_config", {})
        endpoint = historical.get("endpoint", {})
        self.policy = {
            "protocol": "saved-heldout-non-free-text-reference-v1",
            "source_run": str(self.root),
            "teacher": {
                "model_id": historical.get("model_id", results.get("model")),
                "model": endpoint.get("model"),
                "base_url": endpoint.get("base_url"),
            },
            "source_policy": request.get("plan_generation"),
            "source_hashes": self.hashes,
            "scoring": "non-free-text-row-f1-v1; aggregate inputs before aggregation",
        }

    def _remember(self, path, expected=None):
        path = Path(path).resolve()
        key = str(path)
        if key not in self.hashes:
            self.hashes[key] = file_digest(path)
        actual = self.hashes[key]
        if expected is not None and actual != expected:
            raise ValueError(f"Saved reference evidence changed: {path}")
        return actual

    def _load(self, item):
        key = item["id"]
        parts = Path(key)
        if parts.is_absolute() or ".." in parts.parts:
            raise ValueError("Invalid saved reference query ID")
        previous = self.items.get(key)
        if previous is None or previous["sql_sha256"] != item["sql_sha256"]:
            raise ValueError(f"Saved reference query SQL changed: {key}")
        sql = Path(item["path"]).read_text()
        if hashlib.sha256(sql.encode()).hexdigest() != item["sql_sha256"]:
            raise ValueError(f"Current reference query SQL changed: {key}")
        row = self.rows[key]
        folder = Path(row["result_artifact"]).resolve()
        query_root = (self.root / key).resolve()
        if not query_root.is_relative_to(self.root) or not folder.is_relative_to(
            query_root
        ):
            raise ValueError(f"Saved reference artifact leaves query directory: {key}")
        attempt = folder.parent.parent
        if self._remember(attempt / "query.sql") != item["sql_sha256"]:
            raise ValueError(f"Saved reference SQL evidence changed: {key}")
        dataset = {"movies": "movies_nirvana"}.get(
            Path(item["path"]).parent.name, Path(item["path"]).parent.name
        )
        database = self.databases[dataset]
        if Path(row["source_database"]).resolve() != database:
            raise ValueError(f"Saved reference query database changed: {key}")
        for path in (
            folder / "result.csv",
            folder / "result-contract.json",
            attempt / "query_ir.json",
        ):
            self._remember(path)
        expected = {
            "status": "completed",
            "artifact": str(folder),
            "seconds": row["execution_seconds"],
            "query_ir": read(attempt / "query_ir.json"),
            "sql_sha256": item["sql_sha256"],
            "database": str(database),
            "execution_timeout_seconds": None,
            "source_run": str(self.root),
        }
        if capture_plan(sql) is not None:
            if not self.replay:
                raise ValueError(
                    f"Saved aggregate reference needs a certified aggregate manifest: {key}"
                )
            entry = self.replay["queries"][key]
            if (
                entry.get("status") != "completed"
                or entry.get("final_multiset_equal") is not True
                or entry.get("scorable") is not True
                or Path(entry["source_artifact"]).resolve() != folder
                or entry.get("source_sql_sha256") != item["sql_sha256"]
                or entry.get("database_sha256") != self.database_hashes[dataset]
                or not entry.get("source_hashes")
            ):
                raise ValueError(f"Unverified saved aggregate reference: {key}")
            # Builds may legitimately change; only original data/result evidence
            # must remain identical for consuming an already validated capture.
            for name, fingerprint in entry["source_hashes"].items():
                path = Path(name).resolve()
                if path.is_relative_to(self.root) or path in self.databases.values():
                    self._remember(path, fingerprint)
            for name, digest_name in (
                ("pre_aggregate_path", "pre_aggregate_sha256"),
                ("aggregate_input_path", "aggregate_input_sha256"),
            ):
                path = Path(entry[name]).resolve()
                if not path.is_relative_to(self.replay_root):
                    raise ValueError(
                        f"Aggregate reference leaves recovery directory: {key}"
                    )
                self._remember(path, entry[digest_name])
            expected["pre_aggregate_path"] = entry["pre_aggregate_path"]
        self.references[key] = expected

    def reference(self, key, sql, database):
        expected = self.references[key]
        if (
            hashlib.sha256(sql.encode()).hexdigest() != expected["sql_sha256"]
            or str(Path(database).resolve()) != expected["database"]
        ):
            raise ValueError(f"Saved reference execution inputs changed: {key}")
        return dict(expected), True
