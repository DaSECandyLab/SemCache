"""User-specified training label adjustments; measured F1 stays available."""

import copy
import math
import statistics
from collections import defaultdict
from pathlib import Path

from training.collection.rewards import execution_reward

FAST_ZERO_ACCURACY_FLOOR = 0.2
ACCURACY_LABEL_POLICY = "fastest_of_three_zero_accuracy_floor_v2"


def _group(record):
    metadata = record.get("metadata", {})
    # The round boundary distinguishes repeated executions of the same query.
    # Retain the run identity as well when replay includes several sources.
    artifacts = metadata.get("timing_artifacts", [])
    run = round_id = None
    if artifacts:
        parts = Path(artifacts[0]).parts
        for index, part in enumerate(parts[:-1]):
            if part == "measured" and parts[index + 1].startswith("round-"):
                run, round_id = parts[:index], parts[index + 1]
                break
    round_id = round_id or metadata.get("cache_segment", {}).get("id")
    if round_id is None:
        return None
    return run, round_id, metadata.get("workload", record.get("query_key"))


def _seconds(record):
    times = record.get("times", [])
    if record.get("status") != "completed" or not times:
        return None
    if any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times):
        return None
    return statistics.mean(times)


def apply_accuracy_labels(records):
    """Adjust only the fastest rollout in each complete three-plan group.

    First undo previous label adjustments using the measured F1. Groups with
    fewer than three scored rollouts remain unadjusted. Timeouts retain zero
    accuracy and cannot be selected as the fastest completed execution.
    """
    groups = defaultdict(list)
    before = {}
    selected = []
    for record in records:
        if (
            record.get("source") != "measured"
            or record.get("split", "train") != "train"
        ):
            continue
        quality = record.get("metadata", {}).get("quality", {})
        raw = quality.get(
            "raw_accuracy",
            quality.get("accuracy", record.get("raw_accuracy", record.get("accuracy"))),
        )
        if raw is None:
            continue
        before[id(record)] = copy.deepcopy(
            (
                record.get("accuracy"),
                record.get("raw_accuracy"),
                quality,
                record.get("reward"),
            )
        )
        selected.append(record)
        record["raw_accuracy"] = raw
        record["accuracy"] = raw
        quality.update(raw_accuracy=raw, accuracy=raw, value=raw, training_accuracy=raw)
        for key in tuple(quality):
            if key.startswith("accuracy_adjustment"):
                quality.pop(key)
        record.setdefault("metadata", {})["quality"] = quality
        group = _group(record)
        if group is not None:
            groups[group].append(record)
    for group in groups.values():
        if len(group) != 3 or {r["metadata"].get("rollout_index") for r in group} != {
            1,
            2,
            3,
        }:
            continue
        completed = [r for r in group if _seconds(r) is not None]
        if not completed:
            continue
        fastest = min(
            completed, key=lambda r: (_seconds(r), r["metadata"]["rollout_index"])
        )
        if fastest["raw_accuracy"] == 0.0:
            fastest["accuracy"] = FAST_ZERO_ACCURACY_FLOOR
            fastest["metadata"]["quality"].update(
                training_accuracy=FAST_ZERO_ACCURACY_FLOOR,
                accuracy_adjustment=ACCURACY_LABEL_POLICY,
                accuracy_adjustment_seconds=_seconds(fastest),
                accuracy_adjustment_group_size=3,
            )
    changed = 0
    for record in selected:
        record["reward"] = execution_reward(record)
        after = (
            record["accuracy"],
            record["raw_accuracy"],
            record["metadata"]["quality"],
            record["reward"],
        )
        changed += before[id(record)] != after
    return changed
