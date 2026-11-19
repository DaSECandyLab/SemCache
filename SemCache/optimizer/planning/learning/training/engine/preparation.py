"""Prepare experience targets, resume state, optimizer and LR schedule."""

import copy
import math
import random
import time
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import torch

from planning.kernels.features import pack
from planning.learning.batches.epoch import batches_per_epoch, epoch_batches
from planning.learning.batches.frozen import FrozenBatchCache
from planning.learning.convergence import LossPlateau

from ..config import TrainingConfig
from ..targets import TARGET_PROTOCOL, monte_carlo_returns


def prepare_training(
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
    return SimpleNamespace(
        model=model,
        data=data,
        source=source,
        config=config,
        loss_config=loss_config,
        progress=progress,
        progress_epoch=progress_epoch,
        checkpoint_dir=checkpoint_dir,
        stop_after_epoch=stop_after_epoch,
        history=history,
        start_epoch=start_epoch,
        elapsed_before=elapsed_before,
        started=started,
        labels=labels,
        groups=groups,
        datasets=datasets,
        validation=validation,
        frozen_batches=frozen_batches,
        reference=reference,
        estimated_censoring=estimated_censoring,
        effective_censor_weight=effective_censor_weight,
        preparation_seconds=preparation_seconds,
        rng=rng,
        optimizer=optimizer,
        scheduler=scheduler,
        warmup=warmup,
        steps=steps,
        coverage_epochs=coverage_epochs,
        checkpoints=checkpoints,
        expected_states=expected_states,
        losses=losses,
        validation_epochs=validation_epochs,
        loss_terms=loss_terms,
        validation_terms=validation_terms,
        best_loss=best_loss,
        best_epoch=best_epoch,
        best_weights=best_weights,
        stopping=stopping,
        monitor=monitor,
        stopped_early=stopped_early,
        availability=availability,
        best_availability=best_availability,
        epoch_timings=epoch_timings,
        resume_information=resume_information,
    )
