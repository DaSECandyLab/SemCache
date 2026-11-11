"""Fit value heads over frozen experience, with checkpoint recovery and monitoring."""

import copy
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Any

import torch

from planning.learning.epoch_batches import (
    batches_per_epoch,
    epoch_batches,
    predict_batch,
)

from .config import TrainingConfig
from .evaluation import evaluation_metrics
from .loss import value_loss_terms
from .targets import TARGET_PROTOCOL, monte_carlo_returns


def train(
    model,
    data,
    *,
    source,
    config=None,
    progress=None,
    progress_epoch=None,
    checkpoint_dir=None,
    resume_history=None,
    resume_state=None,
    stop_after_epoch=None,
    extend_epoch_budget=False,
):
    if source not in ("estimated", "measured"):
        raise ValueError("Training source must be estimated or measured")
    config = config or TrainingConfig()
    if model.time_scale != config.time_scale:
        raise ValueError("Model and training time scales must match")
    effective_censor_weight = (
        config.estimated_censor_weight
        if source == "estimated"
        else config.censor_weight
    )
    loss_config = replace(config, censor_weight=effective_censor_weight)
    # Simulator pretraining always consumes its full epoch budget. Validation
    # still selects the best checkpoint; measured RL updates retain patience.
    if source == "estimated":
        config = replace(config, patience=0)
    history: list[dict[str, Any]] = list(resume_history or ())
    start_epoch = len(history)
    if stop_after_epoch is not None and (
        type(stop_after_epoch) is not int
        or not start_epoch <= stop_after_epoch <= config.epochs
    ):
        raise ValueError(
            "Explicit epoch limit must lie between the resume epoch and total budget"
        )
    learning_rate_restart = None
    epoch_budget_extension = None
    if start_epoch > config.epochs or any(
        entry["epoch"] != index + 1
        or not entry["epoch"] <= entry["epochs"] <= config.epochs
        for index, entry in enumerate(history)
    ):
        raise ValueError(
            "Resume history must contain consecutive epochs within the total budget"
        )
    if start_epoch:
        previous = model.training_info or {}
        previous_config = previous.get("hyperparameters")
        compatible_config = previous_config
        if extend_epoch_budget:
            if not previous_config or not resume_state:
                raise ValueError(
                    "Extending the epoch budget requires a full training checkpoint"
                )
            candidate = copy.deepcopy(previous_config)
            old_convergence = candidate.get("convergence")
            new_convergence = config.convergence
            if (
                config.epochs < previous_config["epochs"]
                or not old_convergence
                or not new_convergence
                or new_convergence["min_epochs"] < old_convergence["min_epochs"]
                or (
                    config.epochs == previous_config["epochs"]
                    and new_convergence["min_epochs"]
                    == old_convergence["min_epochs"]
                )
                or new_convergence["relative_min_delta"]
                != old_convergence["relative_min_delta"]
            ):
                raise ValueError(
                    "Epoch or minimum-epoch extension must increase a budget "
                    "and preserve the relative loss threshold"
                )
            candidate.update(
                epochs=config.epochs,
                checkpoint_epochs=config.checkpoint_epochs,
                convergence=config.convergence,
            )
            if candidate != vars(config):
                raise ValueError(
                    "Epoch extension cannot change other training hyperparameters"
                )
            epoch_budget_extension = {
                "previous_epochs": previous_config["epochs"],
                "epochs": config.epochs,
                "previous_min_epochs": old_convergence["min_epochs"],
                "min_epochs": new_convergence["min_epochs"],
                "optimizer_reset": False,
                "warmup_restarted": False,
            }
            compatible_config = vars(config)
        if (
            not epoch_budget_extension
            and previous_config
            and previous_config != vars(config)
        ):
            compatible_config = {
                **previous_config,
                "learning_rate": config.learning_rate,
            }
            if (
                compatible_config == vars(config)
                and config.learning_rate < previous_config["learning_rate"]
                and resume_state
                and resume_state.get("best_weights") is not None
            ):
                learning_rate_restart = {
                    "previous_base_rate": previous_config["learning_rate"],
                    "base_rate": config.learning_rate,
                    "weights_from_epoch": resume_state["best_epoch"],
                    "optimizer_reset": True,
                    "warmup_restarted": False,
                }
            else:
                compatible_config = previous_config
        if (
            previous.get("epoch") != start_epoch
            or previous.get("source") != source
            or compatible_config != vars(config)
            or previous.get("target_protocol") != TARGET_PROTOCOL
        ):
            raise ValueError(
                "Resume checkpoint does not match the training history, configuration "
                "or target protocol; restart fitting when completion targets change"
            )
        if (
            any(entry.get("validation") is not None for entry in history)
            and not resume_state
        ):
            raise ValueError(
                "Validation resume requires saved best-model training state"
            )
        if learning_rate_restart and resume_state is not None:
            model.load_state_dict(resume_state["best_weights"])
            model.head_availability = dict(resume_state["best_availability"])
            model.value_available = all(model.head_availability.values())
    elapsed_before = history[-1]["seconds"] if history else 0.0
    started = time.perf_counter()
    # Each prefix is a state in one episode.  completion_labels propagates the
    # terminal execution outcome to every prefix: this is the Monte-Carlo return
    # target V(s), rather than an independently invented label for the prefix.
    labels = monte_carlo_returns(
        data,
        source,
        discount=config.discount,
        estimated_timeout=config.estimated_timeout,
    )
    if not labels:
        raise ValueError("No value targets for training")
    groups = defaultdict(lambda: defaultdict(list))
    datasets = {}
    for record in data["records"]:
        key, name = record["query_key"], record.get("dataset", "default")
        if key in datasets and datasets[key] != name:
            raise ValueError("One query belongs to conflicting datasets")
        datasets[key] = name
    for row in labels:
        groups[row["query_key"]][row["context"]].append(row)
    validation = monte_carlo_returns(
        data,
        source,
        split="validation",
        discount=config.discount,
        estimated_timeout=config.estimated_timeout,
    )
    # Reuse only frozen inputs. GRU/embedding/attention activations are rebuilt
    # with gradients for each optimizer update. Stochastic encoders retain the
    # original graph ordering and dropout RNG consumption.
    from planning.kernels.features import pack
    from planning.learning.frozen_batches import FrozenBatchCache

    reference = model.encoder.words.weight
    packed_bases = {}
    frozen_batches = (
        FrozenBatchCache(model.encoder) if not model.config.get("dropout", 0) else None
    )
    for row in list(labels) + list(validation):
        if frozen_batches is not None:
            row["_frozen_batches"] = frozen_batches
            continue
        key = row.get("base_key", id(row))
        if key not in packed_bases:
            packed_bases[key] = pack([row["features"]], reference)[0]
        row["packed"] = packed_bases[key]
    estimated_censoring = {
        "threshold_seconds": config.estimated_timeout
        if source == "estimated"
        else None,
        "terminal_states": len(
            {
                row["base_key"]
                for row in labels
                if row["time_lower_bound_source"] == "estimated"
            }
        ),
        "prefix_time_targets_omitted": len(
            {row["base_key"] for row in labels if row["estimated_prefix_time_omitted"]}
        ),
        "validation_terminal_states": len(
            {
                row["base_key"]
                for row in validation
                if row["time_lower_bound_source"] == "estimated"
            }
        ),
    }
    preparation_seconds = time.perf_counter() - started
    if progress:
        progress(
            f"Prepared {len(labels)} training and {len(validation)} validation states "
            f"in {preparation_seconds:.3f}s"
        )
        if source == "estimated" and config.estimated_timeout is not None:
            progress(f"Estimated execution-horizon feedback: {estimated_censoring}")
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    steps = batches_per_epoch(groups, config.batch_size) * config.epochs
    warmup = max(1, int(steps * config.warmup_fraction))
    if resume_state and resume_state.get("lr_schedule"):
        warmup = resume_state["lr_schedule"]["warmup_steps"]
    elif epoch_budget_extension:
        warmup = max(
            1,
            int(
                batches_per_epoch(groups, config.batch_size)
                * epoch_budget_extension["previous_epochs"]
                * config.warmup_fraction
            ),
        )

    def lr_factor(step):
        if step < warmup:
            return max(0.1, (step + 1) / warmup)
        return 0.5 * (
            1 + math.cos(math.pi * min(1, (step - warmup) / max(1, steps - warmup)))
        )

    completed_steps = start_epoch * batches_per_epoch(groups, config.batch_size)
    for group in optimizer.param_groups:
        group["initial_lr"] = config.learning_rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_factor, last_epoch=completed_steps - 1
    )
    if resume_state:
        if not learning_rate_restart:
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
            if scheduler.last_epoch != completed_steps:
                raise ValueError(
                    "Resume scheduler progress does not match the completed epochs"
                )
        rng.setstate(resume_state["batch_rng"])
        torch.set_rng_state(resume_state["torch_rng"])
        if reference.device.type == "cuda":
            torch.cuda.set_rng_state_all(resume_state["cuda_rng"])
    elif start_epoch:
        for _ in range(start_epoch):
            for _batch in epoch_batches(
                groups, datasets, rng, config.batch_size, config.balance_datasets
            ):
                pass
    coverage_epochs = [entry["epoch_coverage"] for entry in history]
    checkpoints = [entry["checkpoint"] for entry in history if entry.get("checkpoint")]
    expected_states = len({r["base_key"] for r in labels})
    losses = [entry["value_loss"] for entry in history]
    validation_epochs = [
        entry["validation"] for entry in history if entry.get("validation") is not None
    ]
    loss_terms = [entry["terms"] for entry in history]
    validation_terms = [
        entry["validation_terms"]
        for entry in history
        if entry.get("validation_terms") is not None
    ]
    best_loss, best_epoch, best_weights = math.inf, None, None
    from planning.learning.convergence import LossPlateau

    stopping = LossPlateau(config.patience, **(config.convergence or {}))
    monitor = "validation" if validation else "training" if config.convergence else None
    stopped_early = False
    availability = dict(model.head_availability)
    best_availability = dict(availability)
    if resume_state:
        best_loss = resume_state["best_loss"]
        best_epoch = resume_state["best_epoch"]
        best_weights = resume_state["best_weights"]
        best_availability = resume_state["best_availability"]
        if learning_rate_restart:
            stopping.reference_loss = best_loss
            stopping.reference_epoch = start_epoch
        elif "convergence" in resume_state:
            for key, value in resume_state["convergence"].items():
                setattr(stopping, key, value)
    elif history:
        for entry in history:
            score = (
                entry.get("monitor_loss")
                if config.convergence
                else entry.get("validation")
            )
            if score is not None:
                stopping.update(entry["epoch"], score)
    stopped_early = bool(monitor and start_epoch and stopping.should_stop(start_epoch))
    epoch_timings = [
        {name: entry[name] for name in ("training_seconds", "validation_seconds")}
        for entry in history
    ]
    resume_information = (
        {
            "from_epoch": start_epoch,
            "optimizer_state_restored": bool(resume_state)
            and not bool(learning_rate_restart),
            "torch_rng_restored": bool(resume_state),
            "completed_updates": completed_steps,
            "initial_learning_rate": optimizer.param_groups[0]["lr"],
            "learning_rate_restart": learning_rate_restart,
            "epoch_budget_extension": epoch_budget_extension,
        }
        if start_epoch
        else None
    )
    if progress and start_epoch:
        progress(
            f"Resuming epoch {start_epoch + 1}/{config.epochs}: {resume_information}"
        )
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
            values = torch.stack(list(terms.values())).detach().cpu().tolist()
            epoch_terms.append(dict(zip(terms, values)))
            if not math.isfinite(epoch_terms[-1]["total"]):
                raise ValueError("Non-finite value regression loss")
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
            epoch.append(epoch_terms[-1]["total"])
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
            from planning.learning.checkpoints import complete_metrics, save_checkpoint

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
