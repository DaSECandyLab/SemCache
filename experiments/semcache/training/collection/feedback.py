"""Independent search/execution episodes with immediate frozen-reference F1."""

import gc
import time
from pathlib import Path

from planning.learning.training.data import dataset, validate_dataset

from training.artifacts import read, save
from training.collection.accuracy_labels import apply_accuracy_labels
from training.collection.labels import COLLECTION_PROTOCOL
from training.collection.policy import (
    ACCURACY_FEEDBACK_PROTOCOL,
    FEEDBACK_PROTOCOL,
    logical_structure_signature,
    plan_signature,
)
from training.query_manifest import validate_round_workload
from training.references.scoring import (
    OUTPUT_COMPARISON_REVISION,
    finalize_measured_accuracy,
)


def rollout_entries(workload, destination, count=2):
    return [
        (
            workload,
            Path(destination) / f"search-{index}",
            read(Path(destination) / f"search-{index}" / "experience.json"),
        )
        for index in range(1, count + 1)
    ]


def persist_accuracy_labels(payload, destination):
    """Publish query-group labels into each rollout and its training artifacts."""
    destination = Path(destination)
    apply_accuracy_labels(payload["records"])
    for index, record in enumerate(payload["records"], 1):
        leaf = destination / f"search-{index}"
        original = read(leaf / "experience.json")
        save(leaf / "experience.json", {**original, "records": [record]})
        quality = record["metadata"]["quality"]
        save(leaf / "reference.json", quality)
        save(leaf / "qualities.json", {record["plan_key"]: quality})
        save(leaf / "reward.json", record["reward"])
    save(destination / "experience.json", payload)


def validate_rollout(payload, workload, index, protocol=FEEDBACK_PROTOCOL):
    validate_dataset(payload)
    validate_round_workload(payload, {"workloads": [workload]})
    if len(payload["records"]) != 1:
        raise ValueError("Each feedback search must execute exactly one plan")
    record = payload["records"][0]
    meta = record.get("metadata", {})
    if (
        record["source"] != "measured"
        or meta.get("collection_protocol") != COLLECTION_PROTOCOL
        or meta.get("feedback_protocol") != protocol
        or meta.get("rollout_index") != index
        or meta.get("quality_scope") != "execution"
        or meta.get("feature_audit", {}).get("status") != "passed"
    ):
        raise ValueError("Feedback rollout has incompatible provenance")


def restore_completed_query(
    workload,
    destination,
    *,
    minimum_accuracies,
    calibration,
    reference_sha256,
    round_number,
):
    """Load a committed, scored query once without replaying its lifecycle."""
    destination = Path(destination)
    if (
        not (destination / "collection.json").is_file()
        or not (destination / "experience.json").is_file()
    ):
        return None
    marker = read(destination / "collection.json")
    count = len(minimum_accuracies)
    if (
        marker.get("status") != "measured_complete"
        or marker.get("feedback_protocol") != ACCURACY_FEEDBACK_PROTOCOL
        or marker.get("records") != count
        or marker.get("minimum_accuracies") != list(minimum_accuracies)
    ):
        return None
    payload = read(destination / "experience.json")
    if len(payload["records"]) != count:
        return None
    for index, record in enumerate(payload["records"], 1):
        leaf = destination / f"search-{index}"
        if (
            not (leaf / "experience.json").is_file()
            or not (leaf / "collection.json").is_file()
        ):
            return None
        validate_rollout(
            {**payload, "records": [record]},
            workload,
            index,
            ACCURACY_FEEDBACK_PROTOCOL,
        )
        metadata = record["metadata"]
        if (
            metadata.get("minimum_accuracy") != minimum_accuracies[index - 1]
            or metadata.get("searches_per_query") != count
            or metadata.get("selection_policy") != "minimum_accuracy"
        ):
            return None
        quality = metadata.get("quality", {})
        if quality.get("metric") != workload.get("accuracy_metric"):
            return None
        if record["status"] == "completed":
            if (
                quality.get("status") != "completed"
                or quality.get("output_comparison_revision")
                != OUTPUT_COMPARISON_REVISION
                or quality.get("reference_sha256") != reference_sha256
            ):
                return None
        elif (
            record["status"] != "timeout"
            or quality.get("status") != "execution_timeout"
        ):
            return None
        token = f"measured/{round_number}/{workload['id']}/search-{index}"
        if token not in calibration.state["applied"]:
            return None
    persist_accuracy_labels(payload, destination)
    return payload


def collect_feedback_query(
    workload,
    destination,
    *,
    collect,
    calibration,
    answers,
    teacher_config,
    round_number,
    completed_before,
    update,
    progress,
    minimum_accuracies=None,
):
    """Run separate Beam searches, executing and scoring one selected plan each."""
    thresholds = tuple(minimum_accuracies) if minimum_accuracies is not None else None
    if thresholds is not None and (
        not thresholds
        or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in thresholds)
    ):
        raise ValueError("Feedback accuracy targets must be a nonempty list in [0,1]")
    count = len(thresholds) if thresholds is not None else 2
    protocol = (
        ACCURACY_FEEDBACK_PROTOCOL if thresholds is not None else FEEDBACK_PROTOCOL
    )
    seen = set()
    seen_structures = set()
    selections = []
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    statistics_path = destination / "prepared-statistics.json"

    def prepared_statistics():
        if thresholds is None:
            return None
        if statistics_path.exists():
            snapshot = read(statistics_path)
            if snapshot.get("scope") != "one_query_one_round":
                raise ValueError(
                    "Prepared statistics have an incompatible sharing scope"
                )
            return snapshot
        first = destination / "search-1"
        if (first / "query_ir.json").exists() and (first / "profile.json").exists():
            profile = read(first / "profile.json")
            if profile.get("sampling_strategy") != "independent_base":
                raise ValueError(
                    "Saved query sampling is not independent-base evidence"
                )
            snapshot = {
                "query": read(first / "query_ir.json"),
                "profile": profile,
                "source": str(statistics_path.resolve()),
                "sampled_in": str(first.resolve()),
                "scope": "one_query_one_round",
            }
            save(statistics_path, snapshot)
            return snapshot
        return None

    prepared_statistics()
    started = time.time()
    save(
        destination / "timing.json",
        {"complete": False, "started_at": started, "phases": {}},
    )
    records, phases = [], {}
    second_policy = None
    for index in range(1, count + 1):
        minimum = thresholds[index - 1] if thresholds is not None else None
        policy = (
            "minimum_accuracy"
            if thresholds is not None
            else "pareto_random"
            if index == 1
            else second_policy
        )
        leaf = destination / f"search-{index}"
        update(
            rollout_index=index,
            selection_policy=policy,
            minimum_accuracy=minimum,
            completed_executions=completed_before + index - 1,
        )
        recovered_execution = (leaf / "experience.json").exists() and not (
            leaf / "collection.json"
        ).exists()
        if (leaf / "experience.json").exists():
            payload = read(leaf / "experience.json")
        else:
            payload = (
                collect(
                    leaf,
                    index,
                    policy,
                    minimum_accuracy=minimum,
                    excluded_plan_signatures=tuple(sorted(seen)),
                    excluded_structure_signatures=tuple(sorted(seen_structures)),
                    prepared_statistics=prepared_statistics(),
                )
                if thresholds is not None
                else collect(leaf, index, policy)
            )
        validate_rollout(payload, workload, index, protocol)
        if thresholds is not None:
            metadata = payload["records"][0]["metadata"]
            if (
                metadata.get("minimum_accuracy") != minimum
                or metadata.get("searches_per_query") != count
            ):
                raise ValueError(
                    "Feedback accuracy target or search count differs from requested episode"
                )
        if payload["records"][0]["metadata"].get("selection_policy") != policy:
            raise ValueError(
                "Feedback selection differs from the requested episode policy"
            )
        if recovered_execution:
            # Execution persists its audited record before collection/reporting
            # finishes. Do not discard an already paid, completed execution just
            # because interruption happened before writing its summary marker.
            required = ("selected-plan.json", "calibration-update.json", "timing.json")
            if any(not (leaf / name).exists() for name in required):
                raise ValueError(
                    f"Saved execution lacks recovery artifacts; preserved without rerunning: {leaf}"
                )
            save(
                leaf / "collection.json",
                {
                    "status": "measured_complete",
                    "workload": workload,
                    "records": 1,
                    "recovered_from_execution_record": True,
                },
            )
        finalize_measured_accuracy(
            [(workload, leaf, payload)],
            reference_answers=answers,
            teacher_config=teacher_config,
            wait=True,
            progress=progress,
        )
        if workload["split"] == "train":
            token = f"measured/{round_number}/{workload['id']}/search-{index}"
            if token not in calibration.state["applied"]:
                calibration.commit(token, **read(leaf / "calibration-update.json"))
        record = payload["records"][0]
        records.append(record)
        if thresholds is not None:
            prepared_statistics()
            signature = plan_signature(record["plan"])
            selection = read(leaf / "selected-plan.json")["rollout_selection"]
            if signature in seen and not selection.get("repeat_unavoidable"):
                raise ValueError(
                    "Repeated training plan was not explicitly marked unavoidable"
                )
            seen.add(signature)
            structure = logical_structure_signature(
                record["plan"], read(leaf / "query_ir.json")
            )
            if structure in seen_structures:
                raise ValueError(
                    "Accuracy rollouts must have different logical structures"
                )
            seen_structures.add(structure)
            selections.append(
                {
                    "search": index,
                    "minimum_accuracy": minimum,
                    "plan_signature": signature,
                    "logical_structure_signature": structure,
                    "replacement_scope": selection.get("replacement_scope"),
                    "duplicate_replaced": selection.get("duplicate_replaced", False),
                    "repeat_unavoidable": selection.get("repeat_unavoidable", False),
                    "constraint_satisfied": selection.get("constraint_satisfied"),
                    "accuracy_fallback": selection.get("accuracy_fallback", False),
                }
            )
            save(
                destination / "selection.json",
                {
                    "feedback_protocol": protocol,
                    "minimum_accuracies": thresholds,
                    "selections": selections,
                    "selection_basis": "three accuracy searches; strict distinct logical structures; unseen feasible candidates preferred",
                },
            )
        elif index == 1:
            selection = read(leaf / "selected-plan.json")["rollout_selection"]
            second_policy = (
                "fastest"
                if selection["selected_is_highest_quality"]
                else "highest_quality"
            )
            save(
                destination / "selection.json",
                {
                    "feedback_protocol": FEEDBACK_PROTOCOL,
                    "first_is_highest_quality": selection[
                        "selected_is_highest_quality"
                    ],
                    "second_search_policy": second_policy,
                    "selection_basis": "first search predictions; second search has fresh candidates",
                },
            )
        timing = read(leaf / "timing.json")
        wall = sum(v.get("wall_seconds", 0) for v in timing.get("phases", {}).values())
        wall += record["metadata"].get("quality", {}).get("comparison_seconds", 0)
        phases[f"search-{index}"] = {"wall_seconds": wall}
        save(
            destination / "timing.json",
            {"complete": False, "started_at": started, "phases": phases},
        )
        update(
            completed_executions=completed_before + index,
            last_f1=record.get("accuracy"),
        )
        # Each search has released its clients. Reclaim unreachable native
        # trajectory owners before constructing the next search working set.
        gc.collect()
    payload = dataset(records)
    persist_accuracy_labels(payload, destination)
    save(
        destination / "collection.json",
        {
            "status": "measured_complete",
            "workload": workload,
            "records": count,
            "feedback_protocol": protocol,
            "searches": count,
            "executions": count,
            "minimum_accuracies": thresholds,
        },
    )
    save(
        destination / "timing.json",
        {"complete": True, "started_at": started, "phases": phases},
    )
    return payload
