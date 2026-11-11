"""Paired time/accuracy regression and one-sided censored time loss."""

import math

import torch


def value_loss_terms(predictions, rows, config, *, log_time=False):
    """Mean-pair regression with masked F1 and prefix/terminal timeout bounds."""
    terms = {}
    for key, axis, weight in (
        ("time", 0, 1.0),
        ("accuracy", 1, config.accuracy_weight),
        ("time_lower_bound", 0, config.censor_weight),
    ):
        groups = []
        for terminal, group_weight in (
            (True, config.complete_state_weight),
            (False, config.partial_state_weight),
        ):
            indices = [
                i
                for i, row in enumerate(rows)
                if row[key] is not None and row.get("terminal", True) == terminal
            ]
            if not indices:
                continue
            value = predictions[indices, axis]
            targets = value.new_tensor(
                [
                    math.log1p(rows[i][key] / config.time_scale)
                    if axis == 0
                    else rows[i][key]
                    for i in indices
                ]
            )
            if axis == 0 and not log_time:
                value = torch.log1p(value / config.time_scale)
            error = value - targets
            if key == "time_lower_bound":
                error = error.clamp(max=0)
            groups.append((error.square().mean(), group_weight))
        normalizer = sum(w for _, w in groups)
        terms[key] = weight * sum(
            (v * w / normalizer for v, w in groups), predictions.sum() * 0
        )
    terms["total"] = sum(terms.values(), predictions.sum() * 0)
    return terms


def value_loss(predictions, rows, config, *, log_time=False):
    """Mean-pair regression with one-sided prefix/terminal timeout bounds."""
    return value_loss_terms(predictions, rows, config, log_time=log_time)["total"]
