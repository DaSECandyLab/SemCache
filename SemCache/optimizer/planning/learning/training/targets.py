"""Paired completion labels and terminal-return propagation to partial plans."""

import statistics

from planning.state import digest

from .data import finite, restore_features, validate_dataset

TARGET_PROTOCOL = "comparable-state-mean-completion-pair-v6"


def completion_labels(data, source, *, split="train", estimated_timeout=None):

    validate_dataset(data)
    if estimated_timeout is not None and not finite(estimated_timeout, lower=1e-12):
        raise ValueError("Estimated execution horizon must be finite and positive")
    samples = {}
    for record in data["records"]:
        if record["source"] != source or record.get("split", "train") != split:
            continue
        completion_key = digest(
            [
                record["plan_key"],
                record["context"],
                record["times"],
                record["accuracy"],
                record.get("time_lower_bound"),
            ]
        )
        seconds = statistics.mean(record["times"]) if record["times"] else None
        terminal_depth = record["trajectory"][-1]["depth"]
        for i, entry in enumerate(record["trajectory"]):
            features = restore_features(entry["features"])
            features.pop("accuracy_floor", None)
            key = (record["query_key"], entry["key"], digest(features))
            terminal = i == len(record["trajectory"]) - 1
            row = samples.setdefault(
                key,
                {
                    "query_key": key[0],
                    "key": key[1],
                    "features": features,
                    "base_key": digest(key),
                    "depth": entry["depth"],
                    "terminal": terminal,
                    "contexts": set(),
                    "plans": {},
                },
            )
            if row["depth"] != entry["depth"] or row["terminal"] != terminal:
                raise ValueError("Comparable state has inconsistent trajectory depth")
            row["contexts"].add(record["context"])
            row["plans"].setdefault(
                completion_key,
                {
                    "key": record["plan_key"],
                    "completion_key": completion_key,
                    "time": seconds,
                    "accuracy": record["accuracy"],
                    "bound": record.get("time_lower_bound"),
                    "timeout_penalty": record.get("status") == "timeout"
                    and record.get("accuracy") == 0.0,
                    "remaining_steps": max(0, terminal_depth - entry["depth"]),
                },
            )
    output = []
    for row in samples.values():
        plans = list(row.pop("plans").values())
        contexts = sorted(row.pop("contexts"))
        row["source_contexts"] = contexts
        row["context"] = (
            contexts[0] if len(contexts) == 1 else "merged-" + digest(contexts)
        )
        paired = [
            p for p in plans if p["time"] is not None and p["accuracy"] is not None
        ]
        seconds = quality = bound = bound_source = None
        if paired and not any(p.get("timeout_penalty") for p in plans):
            # Both means use exactly the same completed executions. A timeout
            # is not an exact 600s outcome and must not enter this paired mean.
            selected = paired
            seconds = statistics.mean(p["time"] for p in paired)
            quality = statistics.mean(p["accuracy"] for p in paired)
            aggregation = "paired_completed_mean"
        else:
            selected = plans
            costs = [p["time"] if p["time"] is not None else p["bound"] for p in plans]
            if all(p["time"] is not None for p in plans):
                seconds = statistics.mean(p["time"] for p in plans)
            elif all(v is not None for v in costs):
                # For a mean return, the mean of the observed lower bounds is
                # a lower bound on that return, including at every prefix.
                bound = statistics.mean(costs)
                bound_source = "measured"
            known_quality = [p["accuracy"] for p in plans if p["accuracy"] is not None]
            if known_quality:
                quality = statistics.mean(known_quality)
            aggregation = "masked_mean"
        if (
            source == "estimated"
            and estimated_timeout is not None
            and seconds is not None
            and seconds > estimated_timeout
        ):
            seconds, bound, bound_source = None, estimated_timeout, "estimated"
        horizons = {p["remaining_steps"] for p in selected}
        if len(horizons) != 1:
            raise ValueError("Comparable prefixes have different completion horizons")
        row.update(
            time=seconds,
            accuracy=quality,
            time_lower_bound=bound,
            time_lower_bound_source=bound_source,
            label_aggregation=aggregation,
            target_plan_keys=sorted({p["key"] for p in selected}),
            target_completion_keys=sorted(p["completion_key"] for p in selected),
            target_remaining_steps=next(iter(horizons)),
            completions=len(paired),
            observations=len(plans),
            estimated_prefix_time_omitted=False,
        )
        if any(v is not None for v in (seconds, quality, bound)):
            output.append(row)
    return output


def monte_carlo_returns(
    data,
    source,
    *,
    split="train",
    discount=1.0,
    estimated_timeout=None,
):
    """Build Balsa returns by propagating each terminal outcome to its prefixes.

    The current objective is a terminal cost/quality objective, so ``discount``
    defaults to one: a partial plan is valued by the eventual complete plan it
    can produce.  Keeping this transformation explicit prevents future TD or
    n-step targets from being confused with ordinary per-state labels.
    """
    if not finite(discount, lower=0.0, upper=1.0):
        raise ValueError("Discount must be in [0,1]")
    rows = completion_labels(
        data,
        source,
        split=split,
        estimated_timeout=estimated_timeout,
    )
    if discount == 1.0:
        return rows
    for row in rows:
        # A terminal-only reward has no intermediate rewards; discounting is by
        # the selected completion's horizon, including after episodes merge.
        factor = discount ** row["target_remaining_steps"]
        if row["time"] is not None:
            row["time"] *= factor
        if row["time_lower_bound"] is not None:
            row["time_lower_bound"] *= factor
    return rows
