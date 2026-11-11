"""Migrate existing observations and prepare reference-only HTTP settings."""

import json
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo

from experiments.core.io import save
from experiments.system_comparison.references.cache import TeacherCache
from experiments.system_comparison.references.protocol import (
    complete_response,
    sema_string_response,
)

SYSTEMS = ("plop", "sema", "nirvana", "lotus", "abacus")


def import_history(run, caches, model):
    marker = run / ".cache/teacher/history-import.json"
    if marker.exists():
        return
    counts = {}
    for system, cache in caches.items():
        imported = 0
        for path in sorted((run / system).glob("*/*/gateway.jsonl")):
            pending, observed = {}, set()
            with path.open() as stream:
                for line in stream:
                    if not line.endswith("\n"):
                        continue
                    record = json.loads(line)
                    serial, event = record["request_id"], record["event"]
                    if event == "submit":
                        pending[serial] = record
                        continue
                    if event == "teacher_observation":
                        observed.add(serial)
                    if (
                        event not in {"teacher_observation", "completed"}
                        or event == "completed"
                        and (serial in observed or record.get("cache", {}).get("hit"))
                    ):
                        continue
                    submitted = pending.get(serial)
                    if (
                        not submitted
                        or record.get("endpoint") != model["endpoint"]
                        or record.get("actual_model") != model["id"]
                    ):
                        continue
                    request = submitted["requested"]
                    normalize = (
                        (
                            lambda response, request=request: sema_string_response(
                                request, response
                            )[0]
                        )
                        if system == "sema"
                        else (
                            lambda response, request=request: complete_response(
                                request, response
                            )[0]
                        )
                    )
                    key = cache.key(model["endpoint"], submitted["forwarded"])
                    identity = (
                        str(path.relative_to(run))
                        + ":"
                        + str(serial)
                        + ":"
                        + str(record.get("observation", "legacy"))
                    )
                    try:
                        imported += cache.seed(
                            identity, key, record["response"], normalize
                        )
                    except (KeyError, ValueError):
                        # Invalid or incomplete responses cannot become observations.
                        continue
        counts[system] = imported
    save(
        marker,
        {
            "imported_observations": counts,
            "scope": "independent cache per system; no cross-system sharing",
        },
    )


def prepare_runtime(run, contract, model, *, retry_systems=None, retry_queries=None):
    retry_systems = set(SYSTEMS if retry_systems is None else retry_systems)
    if unknown := retry_systems - set(SYSTEMS):
        raise ValueError("Unknown retry systems: " + ", ".join(sorted(unknown)))
    if retry_queries is not None:
        retry_queries = set(retry_queries)
        allowed = {
            system + "/" + query
            for system in retry_systems
            for query in contract["query_ids"]
        }
        if invalid := retry_queries - allowed:
            raise ValueError("Invalid retry queries: " + ", ".join(sorted(invalid)))
    if not contract.get("cache_enabled"):
        return {}
    caches = {
        system: TeacherCache(run / ".cache/teacher" / system, system)
        for system in SYSTEMS
    }
    import_history(run, caches, model)
    checkpoint = (
        run
        / "checkpoints"
        / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    )
    archived = []
    for system in SYSTEMS:
        for folder in sorted((run / system).glob("*/*")):
            if not folder.is_dir():
                continue
            summary = folder / "summary.json"
            record = json.loads(summary.read_text()) if summary.exists() else {}
            selected = system in retry_systems and (
                retry_queries is None or str(folder.relative_to(run)) in retry_queries
            )
            if record.get("parent_finalized") and (
                not selected or record.get("status") in {"completed", "skipped"}
            ):
                continue
            destination = checkpoint / folder.relative_to(run)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(folder), str(destination))
            archived.append(str(destination.relative_to(run)))
    if archived:
        save(
            checkpoint / "resume.json",
            {
                "archived_incomplete_queries": archived,
                "reason": "Resume unfinished queries with the updated runtime settings and retained cache; completed answers retained",
            },
        )
    return caches
