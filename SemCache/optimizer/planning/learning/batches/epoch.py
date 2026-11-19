"""One complete pass over all comparable-state mean supervision rows."""

import math
from collections import defaultdict

import torch


def training_schedule(samples, datasets, rng, balanced):
    keys = sorted(samples)
    if not balanced:
        rng.shuffle(keys)
        return keys
    groups = defaultdict(list)
    for key in keys:
        groups[datasets.get(key, "default")].append(key)
    for queue in groups.values():
        rng.shuffle(queue)
    result = []
    while groups:
        names = sorted(groups)
        rng.shuffle(names)
        for name in names:
            result.append(groups[name].pop())
            if not groups[name]:
                del groups[name]
    return result


def epoch_batches(groups, datasets, rng, batch_size, balanced):
    """Batch within contexts without skipping states or repeating small datasets."""
    for query in training_schedule(groups, datasets, rng, balanced):
        contexts = sorted(groups[query])
        rng.shuffle(contexts)
        for context in contexts:
            rows = list(groups[query][context])
            rng.shuffle(rows)
            for start in range(0, len(rows), batch_size):
                yield rows[start : start + batch_size]


def batches_per_epoch(groups, batch_size):
    return sum(
        math.ceil(len(rows) / batch_size)
        for contexts in groups.values()
        for rows in contexts.values()
    )


def predict_batch(model, rows, *, log_time=True):
    """Encode each frozen state once and regress its paired mean label."""
    unique, indices, mapping = [], [], {}
    for row in rows:
        key = row["base_key"]
        if key not in mapping:
            mapping[key] = len(unique)
            unique.append(row)
        indices.append(mapping[key])
    cache = rows[0].get("_frozen_batches") if rows else None
    if cache is not None:
        return cache.predict(model, unique, rows, log_time=log_time)
    z = model.encoder.forward_batch(
        [r["features"] for r in unique],
        incremental=False,
        packed=[r.get("packed") for r in unique],
    )
    return model.predictions(
        z[z.new_tensor(indices, dtype=torch.long)],
        log_time=log_time,
    )
