"""Native LOTUS join optimization with lossless multi-column input packing."""

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from experiments.core.paths import ROOT
from experiments.core.resources import BATCH_SIZE
from experiments.system_comparison.reporting.timing import stage
from experiments.validation.native_api import instruction


def packed_inputs(left, right, options):
    frames = []
    names = ("left_record", "right_record")
    for frame, columns, name in zip(
        (left, right),
        (options["left_columns"], options["right_columns"]),
        names,
        strict=True,
    ):
        frame = frame.reset_index(drop=True).copy()
        clean = frame[columns].astype(object).where(pd.notna(frame[columns]), None)
        frame[name] = [
            json.dumps(row, ensure_ascii=False, allow_nan=False, default=str)
            for row in clean.to_dict("records")
        ]
        frames.append(frame)
    arguments = [
        f'the value of "{name}" in '
        + (
            "{left_record:left}"
            if name in options["left_columns"]
            else "{right_record:right}"
        )
        for name in options["columns"]
    ]
    task = instruction(options["task"], arguments)
    return *frames, task


def configure_retrieval(job):
    import lotus
    from lotus.models import SentenceTransformersRM
    from lotus.vector_store import FaissVS

    if lotus.settings.rm is None:
        with stage("planning", "load local LOTUS retrieval model and index backend"):
            lotus.settings.configure(
                rm=SentenceTransformersRM(
                    model=str(ROOT / "sglang/models/all-MiniLM-L6-v2"),
                    device="cpu",
                    max_batch_size=job.get("batch_size", BATCH_SIZE),
                ),
                vs=FaissVS(),
            )


@contextmanager
def observe_optimizer():
    import importlib

    module = importlib.import_module("lotus.sem_ops.sem_join")
    optimize = module.join_optimizer
    learn = module.learn_join_cascade_threshold

    def planning(*args, **kwargs):
        with stage(
            "planning", "native LOTUS search-filter/map-search-filter plan comparison"
        ):
            return optimize(*args, **kwargs)

    def sampling(*args, **kwargs):
        with stage("sampling", "native LOTUS join calibration labels and thresholds"):
            return learn(*args, **kwargs)

    with (
        patch.object(module, "join_optimizer", planning),
        patch.object(module, "learn_join_cascade_threshold", sampling),
    ):
        yield


def join(left, right, options, job):
    from lotus.types import CascadeArgs

    left, right, task = packed_inputs(left, right, options)
    candidates = len(left) * len(right)
    args = CascadeArgs(
        precision_target=0.9,
        recall_target=0.9,
        failure_probability=0.05,
        cascade_IS_random_seed=42,
    )
    if candidates >= args.min_join_cascade_size:
        configure_retrieval(job)
    with observe_optimizer():
        output = left.sem_join(right, task, cascade_args=args, return_stats=True)
    frame, stats = output if isinstance(output, tuple) else (output, {})
    folder = Path(job["output"])
    evidence = {
        "input_rows": [len(left), len(right)],
        "candidate_pairs": candidates,
        "input_adapter": "lossless per-side JSON packing; all original semantic arguments retained",
        "native_cascade_requested": True,
        "native_cascade_applied": bool(stats),
        "small_join_threshold": args.min_join_cascade_size,
        "native_stats": stats,
        "output_rows": len(frame),
        "filter_four_model_extension": False,
        "join_proxy": "native local embedding search; oracle/main is 12B",
    }
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "lotus-join-events.jsonl").open("a") as stream:
        stream.write(
            json.dumps(
                evidence,
                ensure_ascii=False,
                default=lambda value: (
                    value.model_dump(mode="json")
                    if hasattr(value, "model_dump")
                    else str(value)
                ),
            )
            + "\n"
        )
    return frame.drop(columns=["left_record", "right_record"])
