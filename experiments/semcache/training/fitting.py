"""Fit audited collected experience; no search or query execution occurs here."""

import json

import torch
from planning.learning.model import PlanValueModel
from planning.learning.training.config import TrainingConfig
from planning.learning.training.data import dataset, validate_dataset
from planning.learning.training.engine import train
from planning.settings.configuration import load_config

from training.artifacts import save
from training.collection.accuracy_labels import apply_accuracy_labels
from training.collection.collector import COLLECTION_PROTOCOL
from training.collection.policy import ACCURACY_FEEDBACK_PROTOCOL, FEEDBACK_PROTOCOL


def fit(
    records,
    output,
    *,
    epochs,
    checkpoint=None,
    initialize_remaining_from=None,
    seed=17,
    source="measured",
    settings=None,
    extend_epoch_budget=False,
):
    if not records or any(
        r.get("metadata", {}).get("collection_protocol") != COLLECTION_PROTOCOL
        or r.get("source") == "measured"
        and r.get("metadata", {}).get("feedback_protocol")
        not in (FEEDBACK_PROTOCOL, ACCURACY_FEEDBACK_PROTOCOL)
        or r["metadata"].get("feature_audit", {}).get("status") != "passed"
        for r in records
    ):
        raise ValueError(
            "Training requires audited current-protocol inference-time features"
        )
    apply_accuracy_labels(records)
    output.mkdir(parents=True, exist_ok=True)
    epochs_log = output / "epochs.jsonl"
    history = (
        [json.loads(line) for line in epochs_log.read_text().splitlines() if line]
        if epochs_log.exists()
        else []
    )
    available = sorted(
        path
        for path in (output / "checkpoints").glob("epoch-*.pt")
        if path.stem.removeprefix("epoch-").isdigit()
        and int(path.stem.removeprefix("epoch-")) <= len(history)
    )
    resume_checkpoint = available[-1] if available else None
    if resume_checkpoint:
        history = history[: int(resume_checkpoint.stem.removeprefix("epoch-"))]
    else:
        history = []
    torch.manual_seed(seed)
    settings = settings or load_config()
    if checkpoint and initialize_remaining_from:
        raise ValueError(
            "Use either a fitted checkpoint or explicit attention initialization"
        )
    learning = dict(settings["learning"])
    checkpoint_epochs = settings["training"]["checkpoint_epochs"]
    if source == "measured":
        from planning.learning.convergence import validate_update_policy

        policy_name = (
            "round_update" if "round_update" in settings["training"] else "final_update"
        )
        policy = validate_update_policy(
            settings["training"][policy_name], f"training.{policy_name}"
        )
        epochs = policy["max_epochs"]
        learning["patience"] = policy["patience"]
        learning["convergence"] = {
            key: policy[key] for key in ("min_epochs", "relative_min_delta")
        }
        checkpoint_epochs = sorted(
            set(checkpoint_epochs)
            | set(range(policy["min_epochs"], epochs + 1, 32))
            | {epochs}
        )
    torch.set_num_threads(
        min(learning.pop("threads", 1), settings["runtime"]["max_cpu_threads"])
    )
    device = torch.device(learning.pop("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available; training on CPU", flush=True)
        device = torch.device("cpu")
    if resume_checkpoint or checkpoint:
        model = PlanValueModel.load(
            resume_checkpoint or checkpoint, resume_training=bool(resume_checkpoint)
        )
    elif initialize_remaining_from:
        from planning.learning.initialization import initialize_remaining_attention

        model = initialize_remaining_attention(initialize_remaining_from)
    else:
        model = PlanValueModel(
            **settings["value_model"], time_scale=learning["time_scale"]
        )
    initialization_info = (
        model.training_info
        if initialize_remaining_from and not resume_checkpoint
        else None
    )
    model.to(device)
    print(f"Value model training device: {device}", flush=True)
    payload = dataset(records)
    validate_dataset(payload)
    if history:
        epochs_log.write_text(
            "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in history)
        )
    else:
        epochs_log.unlink(missing_ok=True)

    def report_epoch(entry):
        with epochs_log.open("a") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")

    report = train(
        model,
        payload,
        source=source,
        config=TrainingConfig(
            epochs=epochs,
            checkpoint_epochs=tuple(checkpoint_epochs),
            seed=seed,
            estimated_timeout=settings["training"]["estimated_timeout"],
            **learning,
        ),
        progress=lambda value: print(value, flush=True),
        progress_epoch=report_epoch,
        checkpoint_dir=output / "checkpoints",
        resume_history=history,
        resume_state=model.__dict__.pop("_training_state", None),
        extend_epoch_budget=extend_epoch_budget,
    )
    report["device"] = str(device)
    if initialization_info:
        report["initialization"] = initialization_info
        model.training_info["initialization"] = initialization_info
    temporary = output / "value.tmp"
    model.save(temporary)
    temporary.replace(output / "value.pt")
    save(output / "training.json", report)
    return report
