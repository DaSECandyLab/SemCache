"""Paired time/accuracy regression and one-sided censored time loss."""

import math

import torch


def value_loss_terms(predictions, rows, config, *, log_time=False):
    """Mean-pair regression with masked F1 and unified per-sample latency loss."""
    zero = predictions.sum() * 0
    level_sum = config.complete_state_weight + config.partial_state_weight
    level_scale = 2.0 / level_sum if level_sum > 0 else 1.0

    def row_weights(indices):
        return predictions.new_tensor(
            [
                (
                    config.complete_state_weight
                    if rows[i].get("terminal", True)
                    else config.partial_state_weight
                )
                * level_scale
                / 2.0
                for i in indices
            ]
        )

    pred_log_time = (
        predictions[:, 0]
        if log_time
        else torch.log1p(predictions[:, 0] / config.time_scale)
    )
    time_idx = [i for i, row in enumerate(rows) if row["time"] is not None]
    bound_idx = [
        i
        for i, row in enumerate(rows)
        if row["time"] is None and row["time_lower_bound"] is not None
    ]
    n_latency = len(time_idx) + len(bound_idx)
    if time_idx:
        t_targets = predictions.new_tensor(
            [math.log1p(rows[i]["time"] / config.time_scale) for i in time_idx]
        )
        t_error = pred_log_time[time_idx] - t_targets
        time_loss = (row_weights(time_idx) * t_error.square()).sum() / n_latency
    else:
        time_loss = zero
    if bound_idx:
        b_targets = predictions.new_tensor(
            [
                math.log1p(rows[i]["time_lower_bound"] / config.time_scale)
                for i in bound_idx
            ]
        )
        b_error = (pred_log_time[bound_idx] - b_targets).clamp(max=0)
        bound_loss = (
            config.censor_weight
            * (row_weights(bound_idx) * b_error.square()).sum()
            / n_latency
        )
    else:
        bound_loss = zero
    acc_idx = [i for i, row in enumerate(rows) if row["accuracy"] is not None]
    if acc_idx:
        a_targets = predictions.new_tensor([rows[i]["accuracy"] for i in acc_idx])
        a_error = predictions[acc_idx, 1] - a_targets
        acc_loss = (
            config.accuracy_weight * (row_weights(acc_idx) * a_error.square()).mean()
        )
    else:
        acc_loss = zero
    terms = {
        "time": time_loss,
        "accuracy": acc_loss,
        "time_lower_bound": bound_loss,
    }
    terms["total"] = time_loss + acc_loss + bound_loss
    return terms


def value_loss(predictions, rows, config, *, log_time=False):
    """Mean-pair regression with one-sided prefix/terminal timeout bounds."""
    return value_loss_terms(predictions, rows, config, log_time=log_time)["total"]
