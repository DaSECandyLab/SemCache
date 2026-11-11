"""User-authorized multi-model extension built from native LOTUS filter pieces.

This is not an official whole-query LOTUS optimizer. Each level uses the native
filter, log-probability scorer, importance sampler and threshold learner. Only
unresolved rows move upward; accepted negatives do not become downstream input.
All map operations remain the native main-model sem_map.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from experiments.system_comparison.reporting.timing import stage


def save_plan(job):
    from experiments.core.io import save

    folder = Path(job["output"])
    graph = folder / "logical-graph.json"
    metadata = Path(job["query_file"]).with_suffix(".json")

    def events(name):
        path = folder / name
        return (
            [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if path.exists()
            else []
        )

    save(
        folder / "plan.json",
        {
            "kind": "executed program and native operator routing; no global LOTUS plan search",
            "logical_program": json.loads(graph.read_text())
            if graph.exists()
            else json.loads(metadata.read_text()).get("semantic_operations"),
            "filter_cascades": events("cascade-events.jsonl"),
            "semantic_join_plans": events("lotus-join-events.jsonl"),
            "map_model": job["models"][-1]["id"],
            "four_model_filter_extension": len(job["models"]) == 4,
            "filter_cascade_model_count": len(job["models"]),
        },
    )


class FourModelCascade:
    def __init__(self, models, folder, *, precision=0.9, recall=0.9):
        if len(models) < 2:
            raise ValueError("At least two models are required for a filter cascade")
        self.models = models
        self.folder = Path(folder)
        self.serial = 0
        self.precision = precision
        self.recall = recall
        self.timing = {
            "sampling_seconds": 0.0,
            "execution_seconds": 0.0,
            "operations": 0,
            "sampling_operations": 0,
        }

    def filter(self, frame, instruction, *args, **kwargs):
        import lotus
        import numpy as np
        from lotus.sem_ops.cascade_utils import importance_sampling
        from lotus.sem_ops.sem_filter import learn_filter_cascade_thresholds, sem_filter
        from lotus.templates import task_instructions
        from lotus.types import CascadeArgs

        if args or kwargs:
            raise ValueError("The v4 cascade exports use plain semantic filters only")
        self.serial += 1
        columns = lotus.nl_expression.parse_cols(instruction)
        docs = task_instructions.df2multimodal_info(frame, columns)
        task = lotus.nl_expression.nle2str(instruction, columns)
        outputs = [None] * len(frame)
        pending = list(range(len(frame)))
        stages = []
        main = self.models[-1]
        for level, helper in enumerate(self.models[:-1]):
            if not pending:
                break
            current_docs = [docs[i] for i in pending]
            started = time.monotonic()
            prediction = sem_filter(
                current_docs, helper, task, logprobs=True, show_progress_bar=False
            )
            self.timing["execution_seconds"] += time.monotonic() - started
            if prediction.logprobs is None:
                raise ValueError("LOTUS cascade requires real token log probabilities")
            scores = helper.format_logprobs_for_filter_cascade(
                prediction.logprobs
            ).positive_probs
            if len(scores) != len(pending) or not all(
                np.isfinite(p) and 0 <= p <= 1 for p in scores
            ):
                raise ValueError("Invalid native LOTUS helper probabilities")
            # Avoid the upstream zero-sample edge on small query intermediates.
            options = CascadeArgs(
                precision_target=self.precision,
                recall_target=self.recall,
                failure_probability=0.05 / (len(self.models) - 1),
                sampling_percentage=min(1.0, max(0.1, 30 / len(pending))),
                cascade_IS_random_seed=20261024 + 100 * self.serial + level,
            )
            # Positive scores extremely close to zero must still allow native
            # importance sampling to form finite weights. No decision is changed.
            started = time.monotonic()
            with stage("sampling", "native LOTUS importance sampling"):
                indices, weights = importance_sampling(
                    [max(float(p), 1e-12) for p in scores], options
                )
            self.timing["sampling_seconds"] += time.monotonic() - started
            sample = list(map(int, indices))
            started = time.monotonic()
            with stage(
                "sampling", "native LOTUS calibration labels and threshold learning"
            ):
                high, low = learn_filter_cascade_thresholds(
                    sample_multimodal_data=[current_docs[i] for i in sample],
                    lm=main,
                    formatted_usr_instr=task,
                    default=True,
                    cascade_args=options,
                    proxy_scores=[scores[i] for i in sample],
                    sample_correction_factors=weights[indices],
                )
            self.timing["sampling_seconds"] += time.monotonic() - started
            self.timing["sampling_operations"] += 1
            unresolved = []
            positive = negative = 0
            for position, original in enumerate(pending):
                score = scores[position]
                # Conflicting/tied cutoffs are escalated, not guessed.
                if score >= high and score > low:
                    outputs[original] = True
                    positive += 1
                elif score <= low and score < high:
                    outputs[original] = False
                    negative += 1
                else:
                    unresolved.append(original)
            stages.append(
                {
                    "level": level,
                    "helper": helper.model,
                    "input_rows": len(pending),
                    "calibration_draws": len(sample),
                    "calibration_unique": len(set(sample)),
                    "positive_threshold": float(high),
                    "negative_threshold": float(low),
                    "accepted_positive": positive,
                    "accepted_negative": negative,
                    "escalated_rows": len(unresolved),
                }
            )
            pending = unresolved
        if pending:
            started = time.monotonic()
            result = sem_filter(
                [docs[i] for i in pending], main, task, show_progress_bar=False
            )
            self.timing["execution_seconds"] += time.monotonic() - started
            for index, value in zip(pending, result.outputs, strict=True):
                outputs[index] = bool(value)
        if any(value is None for value in outputs):
            raise ValueError("Cascade left unresolved rows")
        path = self.folder / "cascade-events.jsonl"
        self.timing["operations"] += 1
        with path.open("a") as stream:
            stream.write(
                json.dumps(
                    {
                        "operation": self.serial,
                        "instruction": instruction,
                        "input_rows": len(frame),
                        "stages": stages,
                        "main_model_rows": len(pending),
                        "output_rows": sum(outputs),
                        "sampling_seconds": self.timing["sampling_seconds"],
                        "execution_seconds": self.timing["execution_seconds"],
                        "extension": True,
                        "global_quality_guarantee_claimed": False,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        (self.folder / "cascade-timing.json").write_text(
            json.dumps(self.timing, ensure_ascii=False, indent=2) + "\n"
        )
        return frame.iloc[[i for i, keep in enumerate(outputs) if keep]].copy()
