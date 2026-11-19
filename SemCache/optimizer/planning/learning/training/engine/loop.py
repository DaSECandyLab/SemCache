"""Run training epochs, early stopping evaluation and checkpoint emission."""

import copy
import math
import statistics
import time
from collections import Counter

import torch

from planning.learning.batches.epoch import epoch_batches, predict_batch
from planning.learning.checkpoints import complete_metrics, save_checkpoint

from ..evaluation import evaluation_metrics
from ..loss import value_loss_terms
from ..targets import TARGET_PROTOCOL


def run_training_loop(session):
    model = session.model
    data = session.data
    source = session.source
    config = session.config
    loss_config = session.loss_config
    progress = session.progress
    progress_epoch = session.progress_epoch
    checkpoint_dir = session.checkpoint_dir
    stop_after_epoch = session.stop_after_epoch
    start_epoch = session.start_epoch
    elapsed_before = session.elapsed_before
    started = session.started
    labels = session.labels
    groups = session.groups
    datasets = session.datasets
    validation = session.validation
    frozen_batches = session.frozen_batches
    reference = session.reference
    estimated_censoring = session.estimated_censoring
    effective_censor_weight = session.effective_censor_weight
    preparation_seconds = session.preparation_seconds
    rng = session.rng
    optimizer = session.optimizer
    scheduler = session.scheduler
    warmup = session.warmup
    steps = session.steps
    coverage_epochs = session.coverage_epochs
    checkpoints = session.checkpoints
    expected_states = session.expected_states
    losses = session.losses
    validation_epochs = session.validation_epochs
    loss_terms = session.loss_terms
    validation_terms = session.validation_terms
    best_loss = session.best_loss
    best_epoch = session.best_epoch
    best_weights = session.best_weights
    stopping = session.stopping
    monitor = session.monitor
    stopped_early = session.stopped_early
    availability = session.availability
    best_availability = session.best_availability
    epoch_timings = session.epoch_timings
    resume_information = session.resume_information
    epoch_limit = config.epochs if stop_after_epoch is None else stop_after_epoch
    for epoch_index in () if stopped_early else range(start_epoch, epoch_limit):
        epoch_started = time.perf_counter()
        model.train()
        epoch = []
        epoch_terms = []
        visited_states, visited_rows = set(), 0
        for rows in epoch_batches(
            groups, datasets, rng, config.batch_size, config.balance_datasets
        ):
            visited_states.update(r["base_key"] for r in rows)
            visited_rows += len(rows)
            optimizer.zero_grad(set_to_none=True)
            predictions = predict_batch(model, rows, log_time=True)
            terms = value_loss_terms(predictions, rows, loss_config, log_time=True)
            loss = terms["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            optimizer.step()
            availability["time"] |= any(
                row["time"] is not None or row["time_lower_bound"] is not None
                for row in rows
            )
            availability["accuracy"] |= any(row["accuracy"] is not None for row in rows)
            scheduler.step()
            epoch_terms.append({name: value.detach() for name, value in terms.items()})
        if epoch_terms:
            term_names = list(epoch_terms[0])
            host_rows = (
                torch.stack(
                    [torch.stack([row[name] for name in term_names]) for row in epoch_terms]
                )
                .cpu()
                .tolist()
            )
            epoch_terms = [dict(zip(term_names, row)) for row in host_rows]
            for item in epoch_terms:
                if not math.isfinite(item["total"]):
                    raise ValueError("Non-finite value regression loss")
            epoch = [item["total"] for item in epoch_terms]
        coverage_epochs.append(
            {
                "states": expected_states,
                "visited_states": len(visited_states),
                "training_rows": len(labels),
                "visited_state_rows": visited_rows,
                "updates": len(epoch),
            }
        )
        losses.append(statistics.mean(epoch))
        loss_terms.append(
            {
                name: statistics.mean(t[name] for t in epoch_terms)
                for name in epoch_terms[0]
            }
        )
        training_seconds = time.perf_counter() - epoch_started
        validation_started = time.perf_counter()
        score = None
        if validation:
            validation_metrics, terms = evaluation_metrics(
                model, validation, loss_config
            )
            score = validation_metrics["value_loss"]
            validation_epochs.append(score)
            validation_terms.append(terms)
        elif config.convergence:
            # Evaluate the same complete training population with dropout off.
            # In-epoch loss mixes different model weights and shuffled batches.
            metrics, _ = evaluation_metrics(model, labels, loss_config)
            score = metrics["value_loss"]
        if score is not None:
            if score < best_loss:
                best_loss, best_epoch, best_weights = (
                    score,
                    epoch_index + 1,
                    copy.deepcopy(model.state_dict()),
                )
                best_availability = dict(availability)
            stopped_early = stopping.update(epoch_index + 1, score)
        checkpoint = None
        if (
            epoch_index + 1 in config.checkpoint_epochs
            or (
                config.convergence
                and (stopped_early or epoch_index + 1 == config.epochs)
            )
            or (stop_after_epoch is not None and epoch_index + 1 == stop_after_epoch)
        ):

            model.head_availability = dict(availability)
            model.value_available = all(availability.values())
            model.eval()
            checkpoint = {
                "epoch": epoch_index + 1,
                "epoch_coverage": coverage_epochs[-1],
                "training": complete_metrics(model, labels, loss_config),
                "validation": complete_metrics(model, validation, loss_config),
            }
            if checkpoint_dir is not None:
                checkpoint["path"] = save_checkpoint(
                    model,
                    checkpoint_dir,
                    epoch_index + 1,
                    {
                        "source": source,
                        "value_target": "unconditioned_mean_completion_pair",
                        "target_protocol": TARGET_PROTOCOL,
                        "time_output": "log1p_scaled_seconds",
                        "label_aggregation": "comparable_state_paired_mean",
                        "hyperparameters": vars(config),
                        **checkpoint,
                    },
                    training_state={
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "lr_schedule": {"warmup_steps": warmup, "total_steps": steps},
                        "batch_rng": rng.getstate(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all()
                        if reference.device.type == "cuda"
                        else [],
                        "best_loss": best_loss,
                        "best_epoch": best_epoch,
                        "best_weights": best_weights,
                        "best_availability": best_availability,
                        "convergence": stopping.snapshot(),
                    },
                )
            checkpoints.append(checkpoint)
        epoch_timings.append(
            {
                "training_seconds": training_seconds,
                "validation_seconds": time.perf_counter() - validation_started,
            }
        )
        if progress:
            progress(
                f"Epoch {epoch_index + 1}/{config.epochs}: value_loss={losses[-1]:.6f}"
                + (
                    f" monitor_{monitor}_loss={score:.6f} stale_epochs={epoch_index + 1 - stopping.reference_epoch}"
                    if score is not None
                    else ""
                )
            )
        if progress_epoch:
            progress_epoch(
                {
                    "epoch": epoch_index + 1,
                    "epochs": config.epochs,
                    "value_loss": losses[-1],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "monitor_loss": score,
                    "monitor_source": monitor,
                    "terms": dict(loss_terms[-1]),
                    "validation": validation_epochs[-1] if validation_epochs else None,
                    "seconds": round(elapsed_before + time.perf_counter() - started, 3),
                    "validation_terms": validation_terms[-1]
                    if validation_terms
                    else None,
                    **epoch_timings[-1],
                    "estimated_censoring": estimated_censoring,
                    "epoch_coverage": coverage_epochs[-1],
                    "checkpoint": checkpoint,
                    "resume": resume_information,
                    "frozen_input_cache": {
                        "bytes": frozen_batches.bytes,
                        "max_bytes": frozen_batches.max_bytes,
                        "entries": len(frozen_batches.entries),
                        "hits": frozen_batches.hits,
                        "misses": frozen_batches.misses,
                    }
                    if frozen_batches is not None
                    else None,
                }
            )
        if stopped_early:
            break
    if best_weights is not None:
        model.load_state_dict(best_weights)
        availability = best_availability
    model.head_availability = availability
    model.value_available = all(availability.values())
    model.encoder.memo.clear()
    training_metrics, training_terms = evaluation_metrics(model, labels, loss_config)
    validation_metrics, _ = evaluation_metrics(model, validation, loss_config)
    report = {
        "source": source,
        "time_output": "log1p_scaled_seconds",
        "value_target": "unconditioned_mean_completion_pair",
        "target_protocol": TARGET_PROTOCOL,
        "label_aggregation": "comparable_state_paired_mean",
        "epoch_coverage": coverage_epochs,
        "checkpoints": checkpoints,
        "preparation_seconds": preparation_seconds,
        "frozen_input_cache": {
            "hits": frozen_batches.hits,
            "misses": frozen_batches.misses,
            "bytes": frozen_batches.bytes,
            "max_bytes": frozen_batches.max_bytes,
        }
        if frozen_batches is not None
        else None,
        "epoch_timings": epoch_timings,
        "objective": config.algorithm,
        "return_target": "unconditioned_mean_completion_pair"
        if config.algorithm == "balsa_monte_carlo"
        else "direct_completion_targets",
        "discount": config.discount,
        "encoder": "GAT-PE",
        "epochs": len(losses),
        "best_epoch": best_epoch,
        "early_stopping": {
            "monitor": monitor,
            "patience": config.patience,
            "min_epochs": stopping.min_epochs,
            "relative_min_delta": stopping.relative_min_delta,
            "reason": "user_epoch_limit"
            if stop_after_epoch is not None
            else "loss_plateau"
            if stopped_early
            else "max_epochs",
            **stopping.snapshot(),
        }
        if monitor
        else None,
        "queries": len(groups),
        "states": len({row["base_key"] for row in labels}),
        "training_rows": len(labels),
        "validation_rows": len(validation),
        "head_availability": dict(availability),
        "value_available": model.value_available,
        "supervised_states": {
            "time": len(
                {
                    row["base_key"]
                    for row in labels
                    if row["time"] is not None or row["time_lower_bound"] is not None
                }
            ),
            "accuracy": len(
                {row["base_key"] for row in labels if row["accuracy"] is not None}
            ),
        },
        "censored_states": len(
            {r["base_key"] for r in labels if r["time_lower_bound"] is not None}
        ),
        "estimated_censoring": estimated_censoring,
        "effective_censor_weight": effective_censor_weight,
        "datasets": dict(Counter(datasets[q] for q in groups)),
        "losses": losses,
        "loss_terms": loss_terms,
        "validation_terms": validation_terms,
        "training_terms": training_terms,
        "validation_epochs": validation_epochs,
        "training": training_metrics,
        "validation": validation_metrics,
        "feature_groups": list(model.config["feature_groups"]),
        "hyperparameters": vars(config),
        "seconds": elapsed_before + time.perf_counter() - started,
        "resume": resume_information,
    }
    previous = model.training_info
    model.training_info = {
        **report,
        "previous_source": previous.get("source") if previous else None,
        "estimated_initialization": source == "estimated"
        or bool(previous and previous.get("estimated_initialization")),
        "query_keys": sorted(groups),
        "previous_training": previous,
        "query_sources": {
            r["query_key"]: {
                "workload": r["metadata"].get("workload"),
                "sql_digest": r["metadata"].get("workload_sql_digest"),
            }
            for r in data["records"]
            if r["source"] == source and r.get("split", "train") == "train"
        },
    }
    return report
