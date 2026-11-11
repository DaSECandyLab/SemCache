"""Resume a measured round at an explicit cold boundary, never restored KV."""

import copy
from datetime import datetime

from planning.learning.training.data import validate_dataset
from planning.state import digest

from training.artifacts import read, save
from training.calibration.global_profile import shared_profile
from training.query_manifest import validate_round_workload


def validate_measured_prefix(directory, manifest, workloads, calibration, protocol):
    """Validate committed episodes, retaining search 1 across a cold resume."""
    from training.collection.feedback import validate_rollout
    from training.collection.policy import FEEDBACK_PROTOCOL

    stream = directory / "stream"
    names = [w["id"] for w in workloads]
    if len(set(names)) != len(names) or manifest["queries"] != names:
        raise ValueError("Measured resume needs the exact frozen query order")
    if not (directory / "workloads.json").is_file():
        raise ValueError("Measured resume requires frozen round workloads")
    if stream.exists() and {p.name for p in stream.iterdir()} - set(names):
        raise ValueError("Measured resume has unexpected query artifacts")
    prefix, committed, latest, gap = 0, set(), None, False
    for index, workload in enumerate(workloads):
        folder = stream / workload["id"]
        episodes = []
        for slot in (1, 2):
            leaf = folder / f"search-{slot}"
            collection, experience = leaf / "collection.json", leaf / "experience.json"
            if collection.exists() != experience.exists():
                raise ValueError(
                    "Measured resume refuses partially saved execution feedback"
                )
            if not collection.exists():
                if leaf.exists():
                    sql = leaf / "sql"
                    if sql.exists() and any(p.name != "run-1" for p in sql.iterdir()):
                        raise ValueError(
                            "Measured resume refuses a possibly executed search"
                        )
                continue
            if index != prefix or len(episodes) != slot - 1:
                raise ValueError("Measured resume requires a continuous episode prefix")
            summary, payload = read(collection), read(experience)
            validate_rollout(payload, workload, slot)
            if (
                summary.get("status") != "measured_complete"
                or summary.get("records") != 1
                or summary.get("workload") != workload
            ):
                raise ValueError("Measured episode summary is incomplete")
            record = payload["records"][0]
            meta = record["metadata"]
            if (
                meta.get("collection_protocol") != protocol
                or not record.get("context")
                or not meta.get("cache_history_before", {}).get("sha256")
                or any("features" not in item for item in record["trajectory"])
            ):
                raise ValueError("Measured resume refuses unaudited episode context")
            expected_policy = "pareto_random"
            if slot == 2:
                first = read(folder / "search-1/selected-plan.json")[
                    "rollout_selection"
                ]
                expected_policy = (
                    "fastest"
                    if first["selected_is_highest_quality"]
                    else "highest_quality"
                )
            if meta.get("selection_policy") != expected_policy:
                raise ValueError(
                    "Measured resume selection policy differs from the first search"
                )
            delta = read(leaf / "calibration-update.json")
            before, after = (shared_profile(delta[key]) for key in ("before", "after"))
            if latest is not None and digest(before) != digest(latest):
                raise ValueError("Measured resume has a broken calibration chain")
            token = f"measured/{manifest['phase_round']}/{workload['id']}/search-{slot}"
            if workload["split"] == "train":
                if calibration["applied"].get(token) != digest(after):
                    raise ValueError(
                        "Measured episode lacks its exact calibration commit"
                    )
                committed.add(token)
                latest = after
            else:
                latest = before
            episodes.append(record)
        complete = (folder / "collection.json").exists() and (
            folder / "experience.json"
        ).exists()
        if complete:
            summary, payload = (
                read(folder / "collection.json"),
                read(folder / "experience.json"),
            )
            validate_dataset(payload)
            validate_round_workload(payload, {"workloads": [workload]})
            if (
                gap
                or len(episodes) != 2
                or summary.get("status") != "measured_complete"
                or summary.get("workload") != workload
                or summary.get("records") != 2
                or summary.get("feedback_protocol") != FEEDBACK_PROTOCOL
                or payload["records"] != episodes
            ):
                raise ValueError(
                    "Measured query requires both independent search episodes"
                )
            prefix += 1
        else:
            if folder.exists() and index != prefix:
                raise ValueError("Measured resume refuses artifacts after a query gap")
            gap = True
    applied = {
        token
        for token in calibration["applied"]
        if token.startswith(f"measured/{manifest['phase_round']}/")
    }
    if applied != committed:
        raise ValueError("Measured resume has commits without completed episodes")
    if latest is not None and digest(latest) != digest(
        shared_profile(calibration["profile"])
    ):
        raise ValueError(
            "Measured calibration differs from the completed episode prefix"
        )
    return prefix


def begin_resume_segment(
    directory, manifest, prefix, calibration, previous_state, *, reset, snapshot
):
    """Retain finished searches; remove only a proven unexecuted failed attempt."""
    import shutil

    if prefix == len(manifest["queries"]):
        return None
    run_dir = directory.parent.parent
    segments = directory / "cache-segments"
    index = 1
    while (segments / f"resume-{index:04d}.json").exists():
        index += 1
    identity = f"resume-{index:04d}"
    path = segments / f"{identity}.json"
    query = manifest["queries"][prefix]
    destination = directory / "stream" / query
    segment = {
        "id": identity,
        "artifact": str(path.relative_to(run_dir)),
        "boundary": "cold_resume",
        "start_index": prefix,
        "start_query": query,
    }
    event = {
        **segment,
        "status": "pending",
        "previous_state": copy.deepcopy(previous_state),
        "checkpoint_in": manifest["checkpoint_in"],
        "calibration_profile_digest": digest(shared_profile(calibration["profile"])),
        "retained_searches": [],
        "removed_unexecuted_searches": [],
        "created_at": datetime.now().astimezone().isoformat(),
    }
    save(path, event)
    try:
        for slot in (1, 2):
            leaf = destination / f"search-{slot}"
            if (leaf / "collection.json").exists():
                event["retained_searches"].append(slot)
            elif leaf.exists():
                if leaf.is_symlink() or (leaf / "experience.json").exists():
                    raise ValueError("Cannot discard partially saved feedback")
                sql = leaf / "sql"
                if sql.exists() and any(p.name != "run-1" for p in sql.iterdir()):
                    raise ValueError("Cannot discard a possibly executed search")
                shutil.rmtree(leaf)
                event["removed_unexecuted_searches"].append(slot)
        details = reset()
        if snapshot():
            raise ValueError("Measured resume cache reset retained history")
        event.update(
            status="completed",
            cache_reset_details=details,
            cache_history_after_reset={"entries": 0, "sha256": digest([])},
        )
    except BaseException as error:
        event.update(status="failed", error=str(error))
        save(path, event)
        raise
    save(path, event)
    return segment
