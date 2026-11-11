"""Check the value model's loss after every finished training query.

For each completed query (``*/round-*/stream/<query>/experience.json``) that has
not been checked yet, rebuild its labels and evaluate the loss of the newest
available checkpoint on them, splitting it into the time / accuracy / censor
terms. Results append to ``<round>/loss-checks.jsonl`` so the progress table can
show a per-query loss curve without retraining the model each time.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "SemCache/optimizer"))
sys.path.insert(0, str(project_root / "experiments/semcache"))

from planning.learning.model import PlanValueModel  # noqa: E402
from planning.learning.training.config import TrainingConfig
from planning.learning.training.data import dataset
from planning.learning.training.evaluation import evaluate_terms
from planning.learning.training.targets import completion_labels


def read(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def model_path(round_dir: Path):
    """Only the round's own fitted checkpoint counts; there is no preview model."""
    candidate = round_dir / "model" / "value.pt"
    return candidate if candidate.exists() else None


def check(run: Path) -> int:
    import torch
    from planning.settings.configuration import load_config

    configuration = (read(run / "protocol.json") or {}).get("configuration")
    configuration = configuration or load_config()
    torch.set_num_threads(configuration["learning"]["threads"])
    checked = 0
    for round_dir in sorted(run.glob("*/round-*")):
        journal = round_dir / "loss-checks.jsonl"
        done = set()
        if journal.exists():
            for line in journal.read_text().splitlines():
                if line.strip():
                    try:
                        done.add(json.loads(line)["query"])
                    except (json.JSONDecodeError, KeyError):
                        continue
        source = round_dir.parent.name
        experiences = [
            *round_dir.glob("stream/*/experience.json"),
            *round_dir.glob("cold/*/experience.json"),
        ]
        path = model_path(round_dir)
        if path is None:
            continue
        model = None
        for experience in sorted(experiences):
            query = experience.parent.name
            if query in done:
                continue
            data = read(experience) or {}
            records = data.get("records") or []
            if not records:
                continue
            try:
                if model is None:
                    model = PlanValueModel.load(path)
                # Checkpoints freeze label policy; a missing simulator horizon selects uncensored monitoring targets.
                parameters = (model.training_info or {}).get("hyperparameters", {})
                config = TrainingConfig(**{"estimated_timeout": None, **parameters})
                if source == "estimated":
                    config = replace(
                        config, censor_weight=config.estimated_censor_weight
                    )
                labels = completion_labels(
                    dataset(records),
                    source,
                    split=records[0].get("split", "train"),
                    estimated_timeout=config.estimated_timeout,
                )
                if not labels:
                    raise ValueError("No supervision rows for this query split")
                terms = evaluate_terms(model, labels, config)
            except Exception as error:   
                record = {
                    "query": query,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "checked_at": datetime.now().astimezone().isoformat(),
                }
            else:
                record = {
                    "query": query,
                    "status": "ok",
                    "checkpoint": str(path.relative_to(round_dir)),
                    "records": len(records),
                    "states": len(labels),
                    "split": records[0].get("split", "train"),
                    "time_targets": sum(r["time"] is not None for r in labels),
                    "accuracy_targets": sum(r["accuracy"] is not None for r in labels),
                    "censor_targets": sum(
                        r["time_lower_bound"] is not None for r in labels
                    ),
                    **{name: round(value, 6) for name, value in terms.items()},
                    "checked_at": datetime.now().astimezone().isoformat(),
                }
            with journal.open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            checked += 1
    return checked


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: loss_check.py <run-dir>")
    print(check(Path(sys.argv[1]).resolve()))
