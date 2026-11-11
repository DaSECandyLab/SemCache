"""Resume later rounds with a new query cohort and frozen earlier checkpoints."""

from planning.state import digest

from training.artifacts import read, save


def round_query_counts(output, active_count, rounds):
    counts = [active_count] * rounds
    for index in range(rounds):
        directory = output / f"measured/round-{index + 1}"
        if (directory / "model/value.pt").exists():
            counts[index] = len(read(directory / "round.json")["query_ids"])
    return counts


def accept_workload_update(output, previous, current, *, resume_round):
    if resume_round is None:
        return False
    if not 2 <= resume_round <= len(current["round_query_counts"]):
        raise ValueError("A workload continuation must start after a completed round")
    for number in range(1, resume_round):
        directory = output / f"measured/round-{number}"
        if not (directory / "model/value.pt").is_file() or not (
            directory / "experience.json"
        ).is_file():
            raise ValueError(f"Cannot continue without completed round {number}")
    for number in range(resume_round, len(current["round_query_counts"]) + 1):
        if (output / f"measured/round-{number}/model/value.pt").exists():
            raise ValueError("Cannot replace the cohort of an already fitted round")

    allowed = {
        "workload_sql_sha256", "answer_sha256", "reference_contract_sha256",
        "training_manifest_sha256", "round_query_counts",
    }

    def protected(protocol):
        value = {key: item for key, item in protocol.items() if key not in allowed}
        configuration = dict(value["configuration"])
        configuration["training"] = {
            key: item for key, item in configuration["training"].items()
            if key != "queries_per_dataset"
        }
        value["configuration"] = configuration
        return value

    if protected(previous) != protected(current):
        raise ValueError("Workload continuation cannot change model or training settings")
    old, new = previous["workload_sql_sha256"], current["workload_sql_sha256"]
    changed = [key for key in old.keys() & new.keys() if old[key] != new[key]]
    if changed:
        raise ValueError("Existing query SQL changed: " + ", ".join(sorted(changed)))
    old_workloads = read(output / "workloads.json")["workloads"]
    by_id = {workload["id"]: workload for workload in old_workloads}
    for number in range(1, resume_round):
        directory = output / f"measured/round-{number}"
        snapshot = directory / "workloads.json"
        if not snapshot.exists():
            save(snapshot, {
                "suite": "completed-round-frozen-cohort",
                "workloads": [by_id[key] for key in read(directory / "round.json")["query_ids"]],
            })
    for number in range(resume_round, len(current["round_query_counts"]) + 1):
        path = output / f"measured/round-{number}/round.json"
        if path.exists():
            metadata = read(path)
            metadata["query_ids"] = sorted(new)
            save(path, metadata)
    save(output / "workload-update.json", {
        "resume_round": resume_round,
        "previous_sql_sha256": old,
        "active_sql_sha256": new,
        "added": sorted(new.keys() - old.keys()),
        "removed_from_active_cohort": sorted(old.keys() - new.keys()),
        "round_query_counts": current["round_query_counts"],
        "previous_reference_contract_sha256": previous["reference_contract_sha256"],
        "current_reference_contract_sha256": current["reference_contract_sha256"],
        "completed_round_checkpoints_unchanged": True,
        "completed_matching_rollouts_reused": True,
        "replay_policy": "Only matching SQL from the active cohort enters subsequent fitting",
    })
    return True


def matching_replay(records, workloads):
    expected = {workload["id"]: digest(workload["sql"]) for workload in workloads}
    return [
        record for record in records
        if record.get("metadata", {}).get("workload") in expected
        and record["metadata"].get("workload_sql_digest")
        == expected[record["metadata"]["workload"]]
    ]
