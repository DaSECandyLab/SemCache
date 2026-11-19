"""§4.4: calibrated estimate pretraining, then iterative native-SQL feedback."""

import fcntl
import os
from datetime import datetime
from pathlib import Path

from planning.settings.configuration import ROOT as SEMCACHE_ROOT
from planning.settings.configuration import (
    activate_native,
    load_config,
    runtime_path,
)
from training.artifacts import read, save
from training.workloads.datasets import publish_model
from training.workloads.query_manifest import load_manifest, workload_counts

from .phases import run_pipeline_phases
from .setup import (
    calibration_progress_writer as calibration_progress_writer,
)
from .setup import (
    candidate_models as candidate_models,
)
from .setup import (
    prepare_pipeline_context,
)


def run(args):
    configured_settings = load_config(getattr(args, "config", None))
    settings = {
        **configured_settings,
        "response_cache": {
            **configured_settings["response_cache"],
            "directory": None,
            "verify_on_second_use": False,
        },
    }
    activate_native(settings)
    selected_dataset = settings["training"].get("dataset")
    configured_counts = workload_counts(args.queries_per_dataset)
    canonical_path = SEMCACHE_ROOT / settings["training"]["workload_manifest"]
    canonical_manifest = load_manifest(canonical_path)
    collect_only = getattr(
        args, "collect_only", settings["training"].get("collect_only", False)
    )
    if collect_only and (
        args.pretrain_rounds != 1 or args.rounds != 0 or args.evaluate_selected
    ):
        raise ValueError(
            "Collection-only mode requires pretrain_rounds=1, rounds=0 and no evaluation"
        )
    configured_build = runtime_path(settings, "build")
    assert isinstance(configured_build, Path)
    build = Path(getattr(args, "build", configured_build)).resolve()
    run_dir = args.run.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    shared_lock = SEMCACHE_ROOT.parent / "experiments/results/semcache/model_training/.collection.lock"
    shared_lock.parent.mkdir(parents=True, exist_ok=True)
    with (
        shared_lock.open("a") as global_lock,
        (run_dir / "pipeline.lock").open("a") as lock,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (run_dir / "pipeline.pid").write_text(str(os.getpid()) + "\n")
        previous_state = (
            read(run_dir / "pipeline-status.json")
            if (run_dir / "pipeline-status.json").exists()
            else {}
        )
        state = {
            "status": "running",
            "phase": "initializing",
            "pid": os.getpid(),
            "completed_queries": 0,
        }

        def update(**values):
            state.update(values, updated_at=datetime.now().astimezone().isoformat())
            save(run_dir / "pipeline-status.json", state)

        try:
            ctx = prepare_pipeline_context(
                args,
                configured_settings,
                settings,
                build,
                run_dir,
                previous_state,
                state,
                update,
                global_lock=global_lock,
                selected_dataset=selected_dataset,
                configured_counts=configured_counts,
                canonical_path=canonical_path,
                canonical_manifest=canonical_manifest,
                collect_only=collect_only,
            )
            if ctx is None:
                return
            checkpoint = run_pipeline_phases(
                args, settings, build, run_dir, previous_state, state, update, ctx
            )
            teacher = ctx.teacher
            workloads = ctx.workloads
            heldout = ctx.heldout
            sampling_config = ctx.sampling_config
            if args.evaluate_selected:
                from training.evaluation.runner import evaluate

                update(phase="testing", current_query=None)
                tested = evaluate(
                    run_dir,
                    checkpoint,
                    settings=configured_settings,
                    timeout=args.timeout,
                    beam_width=args.candidates,
                    search_seconds=args.search_seconds,
                    max_expansions=args.max_expansions,
                    sampling_config=sampling_config,
                    teacher=teacher,
                    update=update,
                    execution_cpu_parallelism=settings["execution"]["cpu_parallelism"],
                    max_concurrent_requests_per_model=settings["execution"][
                        "max_concurrent_requests_per_model"
                    ],
                    enforce_model_operator_order=settings["execution"][
                        "enforce_model_operator_order"
                    ],
                )
                update(
                    test_queries=len(tested),
                    completed_test_queries=sum(
                        t["status"] == "completed" for t in tested
                    ),
                )
                if len(tested) != len(heldout) or any(
                    t["status"] != "completed" for t in tested
                ):
                    raise RuntimeError(
                        "Heldout evaluation incomplete; inspect evaluation/results.json"
                    )
            publish_model(run_dir, checkpoint, selected_dataset, workloads)
            update(
                status="complete",
                phase="collection_complete" if collect_only else "complete",
                current_query=None,
                checkpoint=str(checkpoint) if checkpoint is not None else None,
                model_training="not_started" if collect_only else "complete",
            )
        except BaseException as error:
            update(
                status="stopped" if isinstance(error, KeyboardInterrupt) else "error",
                error=str(error),
            )
            raise
