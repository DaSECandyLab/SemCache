"""Replace an explicit query subset without discarding other executions."""

import shutil
from datetime import datetime
from zoneinfo import ZoneInfo

from reference_answers.storage.files import file_digest

from training.artifacts import read, save


def start_recollection(output, queries, rounds, checkpoint, searches_per_query):
    path = output / "recollection.json"
    policy = read(path) if path.exists() else None
    expected = sorted(queries)
    if policy and policy.get("status") == "running":
        if policy["queries"] != expected or policy["checkpoint_sha256"] != file_digest(
            checkpoint
        ):
            raise ValueError("Another selective recollection is unfinished")
        return policy
    preserved = {}
    for artifact in (output / "measured").glob(
        "round-*/stream/*/search-*/experience.json"
    ):
        if artifact.parents[1].name not in queries:
            preserved[str(artifact.relative_to(output))] = file_digest(artifact)
    policy = {
        "status": "initializing",
        "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(
            timespec="seconds"
        ),
        "queries": expected,
        "pending_rounds": list(range(1, rounds + 1)),
        "replacement_executions": len(expected) * rounds * searches_per_query,
        "preserved_experiences": preserved,
        "collect_only": True,
        "checkpoint_sha256": file_digest(checkpoint),
    }
    save(path, policy)
    for number in range(1, rounds + 1):
        directory = output / f"measured/round-{number}"
        for identity in expected:
            target = directory / "stream" / identity
            if target.exists():
                shutil.rmtree(target)
        (directory / "experience.json").unlink(missing_ok=True)
        (directory / "summary.json").unlink(missing_ok=True)
    calibration_path = output / "calibration/global.json"
    state = read(calibration_path)
    state["applied"] = {
        token: value
        for token, value in state["applied"].items()
        if token.split("/")[2] not in queries
    }
    save(calibration_path, state)
    policy["status"] = "running"
    save(path, policy)
    return policy
