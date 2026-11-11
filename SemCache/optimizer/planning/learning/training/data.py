"""Versioned execution experience and detached feature restoration."""

import math

from planning.learning.features.schema import FEATURE_VERSION

EXPERIENCE_VERSION = 3


def dataset(records):
    return {
        "version": EXPERIENCE_VERSION,
        "feature_version": FEATURE_VERSION,
        "records": records,
    }


def restore_features(features):
    # Experience features are detached and read-only; normalize dictionary keys without copying numeric/token buffers.
    result = dict(features)
    result["nodes"] = {}
    for key, row in features["nodes"].items():
        restored = dict(row)
        restored["l_previous"] = {
            int(source): value for source, value in row["l_previous"].items()
        }
        result["nodes"][int(key)] = restored
    result["work"] = {int(key): dict(row) for key, row in features["work"].items()}
    return result


def finite(value, *, lower: float = 0.0, upper: float = math.inf):
    return (
        type(value) in (int, float) and math.isfinite(value) and lower <= value <= upper
    )


def validate_dataset(data):
    if (
        data.get("version") != EXPERIENCE_VERSION
        or data.get("feature_version") != FEATURE_VERSION
    ):
        raise ValueError("Incompatible training experience version")
    splits, groups = {}, {}
    for record in data["records"]:
        if record["source"] not in ("estimated", "measured"):
            raise ValueError("Unknown feedback source")
        split = record.get("split", "train")
        if split not in ("train", "validation", "evaluation"):
            raise ValueError("Unknown query split")
        for mapping, key in (
            (splits, record["query_key"]),
            (groups, record.get("split_group", record["query_key"])),
        ):
            if key in mapping and mapping[key] != split:
                raise ValueError("Query/template group appears in multiple splits")
            mapping[key] = split
        times, quality = record["times"], record["accuracy"]
        if (
            any(not finite(t) for t in times)
            or quality is not None
            and not finite(quality, upper=1)
        ):
            raise ValueError("Invalid execution time / quality feedback")
        if record.get("status") == "timeout":
            if (
                times
                or record["source"] != "measured"
                or not finite(record.get("time_lower_bound"), lower=1e-12)
            ):
                raise ValueError(
                    "Censored records need a positive lower bound and no complete times"
                )
        elif record.get("status") == "estimated_unknown":
            if (
                record["source"] != "estimated"
                or times
                or record.get("time_lower_bound") is not None
            ):
                raise ValueError(
                    "Unknown estimated times require estimated source and no time or censor label"
                )
        elif record.get("status") != "completed" or not times:
            raise ValueError(
                "Completed records need real or explicitly estimated times"
            )
        if (
            not record["trajectory"]
            or record["trajectory"][-1]["key"] != record["plan_key"]
        ):
            raise ValueError("Incomplete plan trajectory")
