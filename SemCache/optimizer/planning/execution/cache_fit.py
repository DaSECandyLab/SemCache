"""Query-disjoint fitting with train-only scaling and best-MAE checkpoints."""

import copy
import hashlib
import math

import torch
from torch import nn

from planning.execution.cache_features import FEATURE_SCHEMA, feature_values


def validation_queries(rows, seed=17):
    groups = sorted(
        {r["query"] for r in rows},
        key=lambda q: hashlib.sha256(f"{seed}:{q}".encode()).digest(),
    )
    return set(groups[: max(1, len(groups) // 5)]) if len(groups) >= 5 else set()


def fit_cache_predictor(
    rows,
    *,
    seed=17,
    epochs=1500,
    patience=150,
    on_epoch=None,
    feature_mode="operator",
    hidden_sizes=(64, 32),
    target_transform="log1p",
    learning_rate=0.003,
    weight_decay=0.001,
    loss_kind="mse",
    scale_floor=1.0,
    huber_beta=0.5,
    holdout_queries=None,
):
    from planning.execution.cache_prediction import PROTOCOL, CacheTokenMLP

    if len(rows) < 16:
        raise ValueError("Need at least 16 complete operator observations")
    values = torch.tensor(
        [feature_values(row, feature_mode) for row in rows], dtype=torch.float32
    )
    labels = torch.tensor([float(r["mean_cached_tokens"]) for r in rows])
    if not torch.isfinite(labels).all() or (labels < 0).any():
        raise ValueError("Invalid cache prediction labels")
    validation = (
        validation_queries(rows, seed)
        if holdout_queries is None
        else set(holdout_queries)
    )
    holdout = torch.tensor([r["query"] in validation for r in rows])
    train, metric = ~holdout, holdout if holdout.any() else ~holdout
    if not train.any():
        raise ValueError("Cache predictor split has no training rows")
    if loss_kind not in {"mse", "huber", "absolute"}:
        raise ValueError("Unknown cache predictor training loss")
    if (
        not math.isfinite(scale_floor)
        or scale_floor <= 0
        or not math.isfinite(huber_beta)
        or huber_beta <= 0
    ):
        raise ValueError("Cache predictor scales must be finite and positive")
    target = torch.log1p(labels) if target_transform == "log1p" else labels
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = CacheTokenMLP(
            feature_mode=feature_mode,
            hidden_sizes=hidden_sizes,
            target_transform=target_transform,
        )
        inputs = (
            torch.log1p(values[train]) if feature_mode == "prefix" else values[train]
        )
        model.input_mean.copy_(inputs.mean(0))
        model.input_scale.copy_(inputs.std(0, unbiased=False).clamp_min(scale_floor))
        model.target_mean.copy_(target[train].mean())
        model.target_scale.copy_(target[train].std(unbiased=False).clamp_min(1e-3))
        normalized = (target - model.target_mean) / model.target_scale
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=40, factor=0.5, min_lr=1e-5
        )
        best, best_weights, stale, completed, best_epoch = math.inf, None, 0, 0, 0
        for epoch in range(epochs):
            model.train()
            prediction = model(values[train])
            if loss_kind == "mse":
                loss = nn.functional.mse_loss(prediction, normalized[train])
            elif loss_kind == "huber":
                loss = nn.functional.smooth_l1_loss(
                    prediction, normalized[train], beta=huber_beta
                )
            else:
                loss = nn.functional.l1_loss(prediction, normalized[train])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                raw = model(values[metric])
                monitored = float((model.tokens(raw) - labels[metric]).abs().mean())
                monitor_loss = float(nn.functional.mse_loss(raw, normalized[metric]))
            scheduler.step(monitored)
            completed = epoch + 1
            if monitored < best - 1e-4:
                best, best_weights, stale, best_epoch = (
                    monitored,
                    copy.deepcopy(model.state_dict()),
                    0,
                    completed,
                )
            else:
                stale += 1
            if on_epoch:
                on_epoch(
                    {
                        "epoch": completed,
                        "training_loss": float(loss.detach()),
                        "monitor_loss": monitor_loss,
                        "monitor_mae_tokens": monitored,
                        "best_monitor_mae_tokens": best,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "stale_epochs": stale,
                    }
                )
            if completed >= 150 and stale >= patience:
                break
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        observed = model.tokens(model(values))
        absolute = (observed[metric] - labels[metric]).abs()
    model.metadata = {
        "protocol": PROTOCOL,
        "feature_schema": FEATURE_SCHEMA,
        "feature_mode": feature_mode,
        "architecture": [values.shape[1], *hidden_sizes, 1],
        "epochs": completed,
        "best_epoch": best_epoch,
        "training_samples": int(train.sum()),
        "validation_samples": int(holdout.sum()),
        "training_queries": len(
            {r["query"] for r in rows if r["query"] not in validation}
        ),
        "validation_queries": sorted(validation),
        "validation_is_query_disjoint": bool(holdout.any()),
        "log_mse": float(
            nn.functional.mse_loss(
                torch.log1p(observed[metric]), torch.log1p(labels[metric])
            )
        ),
        "mae_tokens": float(absolute.mean()),
        "wape": float(absolute.sum() / labels[metric].sum().clamp_min(1e-8)),
        "metric_scope": "validation" if holdout.any() else "training_only",
        "prediction_target": "operator mean matched prefix length per HTTP request",
        "target_unit": "tokens",
        "label": "mean usage.prompt_tokens_details.cached_tokens per consumed HTTP response",
        "input_transform": "train-only standardized GAT operator features"
        if feature_mode != "prefix"
        else "train-only standardized log1p prefix lengths",
        "target_transform": target_transform,
        "loss_kind": loss_kind,
        "scale_floor": scale_floor,
        "huber_beta": huber_beta,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "seed": seed,
    }
    return model
