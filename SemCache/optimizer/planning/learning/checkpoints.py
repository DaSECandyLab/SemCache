"""Epoch checkpoints with paired complete-plan validation diagnostics."""

import itertools
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .epoch_batches import predict_batch


def complete_metrics(model, rows, config):
    rows = [r for r in rows if r["terminal"]]
    if not rows:
        return {"states": 0}
    with torch.no_grad():
        prediction = (
            torch.cat(
                [
                    predict_batch(
                        model, rows[i : i + config.batch_size], log_time=False
                    )
                    for i in range(0, len(rows), config.batch_size)
                ]
            )
            .cpu()
            .tolist()
        )
    metrics: dict[str, Any] = {"states": len(rows)}
    for axis, key in enumerate(("time", "accuracy")):
        valid = [i for i, r in enumerate(rows) if r[key] is not None]
        outcomes = defaultdict(list)
        for i in valid:
            outcomes[rows[i]["query_key"], rows[i]["context"]].append(i)
        rankings = []
        for ids in outcomes.values():
            pairs = []
            for i, j in itertools.combinations(ids, 2):
                delta = rows[i][key] - rows[j][key]
                if abs(delta) < 1e-6:
                    continue
                guessed = prediction[i][axis] - prediction[j][axis]
                pairs.append(0.5 if abs(guessed) < 1e-6 else float(delta * guessed > 0))
            if pairs:
                rankings.append(statistics.mean(pairs))
        values = [p[axis] for p in prediction]
        metrics[key] = {
            "labels": len(valid),
            "mae": statistics.mean(
                abs(prediction[i][axis] - rows[i][key]) for i in valid
            )
            if valid
            else None,
            "pairwise_accuracy": statistics.mean(rankings) if rankings else None,
            "ranking_contexts": len(rankings),
            "prediction_min": min(values),
            "prediction_max": max(values),
            "prediction_std": statistics.pstdev(values),
        }
        if key == "time":
            metrics[key]["log_mse"] = (
                statistics.mean(
                    (
                        math.log1p(prediction[i][axis] / config.time_scale)
                        - math.log1p(rows[i][key] / config.time_scale)
                    )
                    ** 2
                    for i in valid
                )
                if valid
                else None
            )
    bounds = [i for i, r in enumerate(rows) if r["time_lower_bound"] is not None]
    metrics["time_lower_bound"] = {
        "states": len(bounds),
        "violations": sum(
            prediction[i][0] < rows[i]["time_lower_bound"] for i in bounds
        ),
    }
    return metrics


def save_checkpoint(model, directory, epoch, information, *, training_state=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"epoch-{epoch:03d}.pt"
    previous = model.training_info
    try:
        model.training_info = {**information, "epoch": epoch}
        temporary = path.with_suffix(".tmp")
        model.save(temporary, training_state=training_state)
        temporary.replace(path)
    finally:
        model.training_info = previous
    return str(path)
