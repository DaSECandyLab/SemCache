"""Calibration, pretraining and measured-round execution stages."""

import gc
from datetime import datetime
from pathlib import Path

from planning.learning.training.data import dataset, validate_dataset
from planning.observations.cache import CacheHistory
from planning.settings.configuration import ROOT as SEMCACHE_ROOT
from planning.settings.model_config import unavailable_models
from planning.settings.teacher import TeacherModel
from training.artifacts import read, save
from training.calibration.calibrate import calibrate_models
from training.calibration.epoch import (
    calibration_paths,
)
from training.calibration.global_profile import GlobalCalibration
from training.collection.collector import (
    COLLECTION_PROTOCOL,
    collect_native,
    reset_context,
)
from training.collection.feedback import collect_feedback_query, rollout_entries
from training.collection.lifecycle.recovery import collect_with_retry
from training.collection.lifecycle.resume import (
    begin_resume_segment,
    validate_measured_prefix,
)
from training.collection.policy import (
    FEEDBACK_PROTOCOL,
    ordered_workloads,
)
from training.fitting import fit
from training.references.answers import TrainingAnswers
from training.references.scoring import (
    RESULT_ACCURACY_METRIC,
    finalize_measured_accuracy,
)
from training.workloads.query_manifest import (
    validate_round_workload,
)

from .setup import calibration_progress_writer


def run_pipeline_phases(
    args, settings, build, run_dir, previous_state, state, update, ctx
):
    models = ctx.models
    catalog = ctx.catalog
    teacher = ctx.teacher
    databases = ctx.databases
    sampling_config = ctx.sampling_config
    resolved_sampling = ctx.resolved_sampling
    retry_policy = ctx.retry_policy
    teacher_config = ctx.teacher_config
    teacher_epoch = ctx.teacher_epoch
    teacher_epoch_id = ctx.teacher_epoch_id
    workloads = ctx.workloads
    feedback_cycle = ctx.feedback_cycle
    policy = ctx.policy
    estimated_policy = ctx.estimated_policy
    selected_dataset = ctx.selected_dataset
    collect_only = ctx.collect_only
    checkpoint = None
    replay = []
    answers = None
    if args.rounds:
        update(phase="checking_reference_answers", current_query=None)
        reference_root = (
            SEMCACHE_ROOT / settings["training"]["reference_answers_dir"]
        )
        pointer = run_dir / "reference_answers_path.json"
        if pointer.is_file():
            reference_root = Path(read(pointer)["root"])
        elif (
            not (reference_root / "contract.json").is_file()
            and (reference_root / "latest.json").is_file()
        ):
            reference_root = Path(read(reference_root / "latest.json")["root"])
        answers = TrainingAnswers(
            reference_root,
            workloads,
            databases,
            allow_subset=selected_dataset is not None,
        )
        # Other datasets' pending answers must not delay this run.
        for workload in workloads:
            answers.load(workload)
        save(
            run_dir / "reference_answers_path.json",
            {
                "root": str(reference_root.resolve()),
                "system": answers.system,
                "metric": RESULT_ACCURACY_METRIC,
            },
        )
    history = CacheHistory(
        run_dir / "history.sqlite",
        capacity=settings["sampling"]["history_capacity"],
    )
    try:
        update(phase="initial_calibration", current_query=None)
        global_path = run_dir / "calibration/global.json"
        if not global_path.exists() and any(
            run_dir.glob("*/round-*/*/*/collection.json")
        ):
            raise ValueError(
                "Missing global calibration for an existing collection; use a fresh run directory"
            )

        def open_calibration(paths, selected_teacher):
            return GlobalCalibration(
                paths["global"],
                models,
                selected_teacher,
                lambda: calibrate_models(
                    workloads,
                    models,
                    selected_teacher,
                    databases,
                    build,
                    paths["probes"],
                    history=history,
                    adaptive=sampling_config.get("adaptive_sketch"),
                    parallelism=resolved_sampling["parallelism"],
                    sample_rows=settings["calibration"]["sample_rows"],
                    cohort_seed=settings["calibration"]["cohort_seed"],
                    progress=calibration_progress_writer(paths["progress"]),
                    shared_cache_dir=(
                        SEMCACHE_ROOT / settings["calibration"]["cache_dir"]
                    ).resolve()
                    if settings["calibration"].get("cache_dir") is not None
                    else None,
                ),
            )

        if teacher_epoch is None:
            calibration = open_calibration(calibration_paths(run_dir), teacher)
        else:
            calibration = GlobalCalibration(
                global_path,
                models,
                TeacherModel(**teacher_epoch["source_teacher"]),
            )

        def round_tests(phase, number, checkpoint):
            from training.evaluation.round import run_requested

            run_requested(
                run_dir,
                phase=phase,
                round_number=number,
                checkpoint=checkpoint,
                settings=settings,
                workloads=workloads,
                models=models,
                catalog=catalog,
                teacher=teacher,
                databases=databases,
                profile=calibration.snapshot(),
                answers=answers,
                update=update,
            )

        for phase, rounds in (
            ("estimated", args.pretrain_rounds),
            ("measured", args.rounds),
        ):
            if phase == "measured" and rounds and teacher_epoch is not None:
                update(
                    phase="initial_calibration",
                    current_query=None,
                    teacher_epoch=teacher_epoch_id,
                    calibration_path=str(calibration_paths(run_dir)["global"]),
                )
                calibration = open_calibration(
                    calibration_paths(run_dir), teacher
                )
            for iteration in range(rounds):
                global_round = (
                    iteration
                    + 1
                    + (args.pretrain_rounds if phase == "measured" else 0)
                )
                directory = run_dir / phase / f"round-{iteration + 1}"
                checkpoint_out = directory / "model/value.pt"
                if (directory / "experience.json").exists():
                    completed = read(directory / "experience.json")
                    validate_dataset(completed)
                    frozen_round_workload = read(directory / "workloads.json")
                    validate_round_workload(completed, frozen_round_workload)
                    if phase == "measured" and frozen_round_workload != read(
                        run_dir / "workloads.json"
                    ):
                        raise ValueError(
                            "Completed measured round SQL differs from the current workload"
                        )
                    manifest = read(directory / "round.json")
                    update(
                        round=iteration + 1,
                        global_round=global_round,
                        total_rounds=args.pretrain_rounds + args.rounds,
                        query_order=manifest["query_order"],
                        current_query=None,
                    )
                    if phase == "measured":
                        replay.extend(completed["records"])
                    if collect_only:
                        continue
                    if not checkpoint_out.exists():
                        update(phase="value_update", round=iteration + 1)
                        fit(
                            replay
                            if phase == "measured"
                            else completed["records"],
                            directory / "model",
                            epochs=args.epochs,
                            checkpoint=checkpoint,
                            seed=args.seed + iteration,
                            source=phase,
                            settings=settings,
                        )
                    checkpoint = checkpoint_out
                    round_tests(phase, iteration + 1, checkpoint)
                    continue
                if checkpoint_out.exists():
                    raise RuntimeError(
                        "Checkpoint has no complete round feedback"
                    )
                interrupted_measured = (
                    phase == "measured" and directory.exists()
                )
                if directory.exists() and (
                    not (directory / "round.json").exists()
                    or phase == "estimated"
                    and (directory / "stream").exists()
                ):
                    raise RuntimeError(
                        "Interrupted round has no compatible frozen manifest"
                    )
                cycle_item = (
                    feedback_cycle[iteration % len(feedback_cycle)]
                    if phase == "measured"
                    else {
                        "plan_generation": "heuristic_requests",
                        "query_order": "shuffled",
                    }
                )
                plan_generation = cycle_item["plan_generation"]
                query_order = cycle_item["query_order"]
                ordered = ordered_workloads(
                    workloads, query_order, args.seed + global_round
                )
                directory.mkdir(parents=True, exist_ok=True)
                round_workloads = directory / "workloads.json"
                current_workloads = read(run_dir / "workloads.json")
                if (
                    round_workloads.exists()
                    and read(round_workloads) != current_workloads
                ):
                    raise ValueError(
                        "Frozen round workload differs from current feedback SQL"
                    )
                if interrupted_measured and not round_workloads.exists():
                    raise ValueError(
                        "Measured resume requires frozen round workloads"
                    )
                if not round_workloads.exists():
                    save(round_workloads, current_workloads)
                manifest = {
                    "round": global_round,
                    "phase": phase,
                    "phase_round": iteration + 1,
                    "query_order": query_order,
                    "query_order_seed": args.seed + global_round
                    if query_order == "shuffled"
                    else None,
                    "feedback_protocol": FEEDBACK_PROTOCOL
                    if phase == "measured"
                    else None,
                    "plan_generation": plan_generation,
                    "cycle_index": (
                        iteration % len(feedback_cycle)
                        if phase == "measured"
                        else None
                    ),
                    "queries": [w["id"] for w in ordered],
                    "checkpoint_in": str(checkpoint) if checkpoint else None,
                    "search_guide": plan_generation,
                    "cache_mode": "cold" if phase == "estimated" else "stream",
                    "cache_reset": "pending",
                }
                if phase == "measured" and teacher_epoch is not None:
                    manifest.update(
                        teacher_epoch=teacher_epoch_id,
                        calibration_path=str(
                            calibration_paths(run_dir)["global"].relative_to(
                                run_dir
                            )
                        ),
                    )
                manifest_path = directory / "round.json"
                if manifest_path.exists():
                    old_manifest = read(manifest_path)
                    for key in (
                        "round",
                        "phase_round",
                        "queries",
                        "checkpoint_in",
                        "search_guide",
                        "phase",
                        "cache_mode",
                        "query_order",
                        "feedback_protocol",
                        "plan_generation",
                        "query_order_seed",
                        "cycle_index",
                        "teacher_epoch",
                        "calibration_path",
                    ):
                        if old_manifest.get(key) != manifest.get(key):
                            raise ValueError(
                                "Frozen cold round manifest differs"
                            )
                    if interrupted_measured:
                        manifest = old_manifest
                prefix = (
                    validate_measured_prefix(
                        directory,
                        manifest,
                        ordered,
                        calibration.state,
                        COLLECTION_PROTOCOL,
                    )
                    if interrupted_measured
                    else 0
                )
                if not interrupted_measured or prefix < len(ordered):
                    update(
                        phase="resetting_cache",
                        round=iteration + 1,
                        global_round=global_round,
                        total_rounds=args.pretrain_rounds + args.rounds,
                        current_query=None,
                    )
                if interrupted_measured:
                    cache_segment = begin_resume_segment(
                        directory,
                        manifest,
                        prefix,
                        calibration.state,
                        previous_state,
                        reset=lambda: reset_context(models, history),
                        snapshot=history.snapshot,
                    )
                else:
                    save(manifest_path, manifest)
                    reset_details = reset_context(models, history)
                    manifest.update(
                        cache_reset="completed",
                        cache_reset_details=reset_details,
                        reset_at=datetime.now().astimezone().isoformat(),
                    )
                    save(manifest_path, manifest)
                    cache_segment = {
                        "id": "initial",
                        "boundary": "round_start",
                        "start_index": 0,
                        "start_query": ordered[0]["id"] if ordered else None,
                    }
                records = []
                round_entries = []
                update(
                    phase=phase,
                    round=iteration + 1,
                    rounds=rounds,
                    query_order=manifest["query_order"],
                    total_queries=len(workloads),
                    completed_queries=0,
                    records=0,
                    cache_segment=cache_segment
                    if phase == "measured"
                    else None,
                )
                for query_index, w in enumerate(ordered):
                    destination = directory / manifest["cache_mode"] / w["id"]
                    update(
                        current_query=w["id"],
                        network_retry=None,
                        plan_generation=plan_generation,
                    )
                    reused = (destination / "collection.json").exists() and (
                        destination / "experience.json"
                    ).exists()
                    if reused:
                        payload = read(destination / "experience.json")
                        validate_dataset(payload)
                        if any(
                            r.get("metadata", {}).get("collection_protocol")
                            != COLLECTION_PROTOCOL
                            or r["metadata"]
                            .get("feature_audit", {})
                            .get("status")
                            != "passed"
                            for r in payload["records"]
                        ):
                            raise ValueError(
                                "Completed query has unaudited feedback"
                            )
                    else:

                        def collect_query(
                            target=destination,
                            rollout_index=1,
                            selection_policy="pareto_random",
                            completed_times=(),
                        ):
                            missing = unavailable_models(models)
                            if missing:
                                raise RuntimeError(
                                    "Model endpoints unavailable: "
                                    + ", ".join(missing)
                                )
                            return collect_native(
                                w,
                                models,
                                catalog,
                                databases[w["dataset"]],
                                build,
                                target,
                                checkpoint=checkpoint
                                if phase == "measured"
                                else None,
                                phase=phase,
                                plan_generation=plan_generation,
                                repeats=args.repeats,
                                candidates=args.candidates,
                                seed=args.seed
                                + global_round * len(ordered)
                                + query_index
                                + (rollout_index - 1) * 1000003,
                                history=history,
                                calibration_profile=calibration.snapshot(),
                                cache_segment=cache_segment
                                if phase == "measured"
                                else None,
                                timeout_policy=estimated_policy
                                if phase == "estimated"
                                else policy,
                                sampling_config=sampling_config,
                                search_seconds=args.search_seconds,
                                max_expansions=args.max_expansions,
                                teacher=teacher,
                                exploration_fraction=settings["training"].get(
                                    "exploration_fraction", 0.0
                                ),
                                inference_settings=settings,
                                execution_cpu_parallelism=settings["execution"][
                                    "cpu_parallelism"
                                ],
                                max_concurrent_requests_per_model=settings[
                                    "execution"
                                ]["max_concurrent_requests_per_model"],
                                enforce_model_operator_order=settings[
                                    "execution"
                                ]["enforce_model_operator_order"],
                                rollout_index=rollout_index,
                                selection_policy=selection_policy,
                                completed_times=completed_times,
                            )

                        if phase == "measured":
                            payload = collect_feedback_query(
                                w,
                                destination,
                                collect=collect_query,
                                calibration=calibration,
                                answers=answers,
                                teacher_config=teacher_config,
                                round_number=iteration + 1,
                                completed_before=2 * query_index,
                                update=update,
                                progress=print,
                            )
                        else:
                            payload = collect_with_retry(
                                collect_query,
                                destination=destination,
                                run_dir=run_dir,
                                phase=phase,
                                policy=retry_policy,
                                reset=lambda: reset_context(models, history),
                                update=update,
                                previous_failure=previous_state.get("error")
                                if previous_state.get("current_query")
                                == w["id"]
                                and previous_state.get("phase") == phase
                                and previous_state.get("round") == iteration + 1
                                else None,
                            )
                    validate_round_workload(payload, {"workloads": [w]})
                    if w["split"] == "train" and phase == "estimated":
                        calibration.commit(
                            f"{phase}/{iteration + 1}/{w['id']}",
                            **read(destination / "calibration-update.json"),
                        )
                    if phase == "estimated" and not reused:
                        reset_context(models, history)
                        save(
                            destination / "cache-reset.json",
                            {
                                "status": "completed",
                                "after_query": w["id"],
                                "reset_at": datetime.now()
                                .astimezone()
                                .isoformat(),
                            },
                        )
                    assert payload is not None
                    records.extend(payload["records"])
                    if phase == "measured":
                        round_entries.extend(rollout_entries(w, destination))
                    update(
                        completed_queries=state["completed_queries"] + 1,
                        records=len(records),
                        network_retry=None,
                    )
                    # Reclaim unreachable trajectory cycles after serialization before constructing the next sampling working set.
                    gc.collect()
                if phase == "measured":
                    assert answers is not None
                    assert teacher_config is not None
                    update(phase="round_accuracy", current_query=None)
                    finalize_measured_accuracy(
                        round_entries,
                        reference_answers=answers,
                        teacher_config=teacher_config,
                        wait=True,
                        update=update,
                    )
                    update(phase="round_accuracy", pending_reference_answers=0)
                    save(
                        directory / "round-accuracy.json",
                        {
                            "status": "complete",
                            "queries": len(round_entries),
                            "labeled": sum(
                                r.get("accuracy") is not None for r in records
                            ),
                            "method": f"frozen_{answers.system}_teacher_answer",
                            "metric": RESULT_ACCURACY_METRIC,
                        },
                    )
                    replay.extend(records)
                save(directory / "experience.json", dataset(records))
                if collect_only:
                    update(phase="collection_complete", current_query=None)
                    continue
                update(phase="value_update", current_query=None)
                fit(
                    replay if phase == "measured" else records,
                    directory / "model",
                    epochs=args.epochs,
                    checkpoint=checkpoint,
                    seed=args.seed + iteration,
                    source=phase,
                    settings=settings,
                )
                checkpoint = checkpoint_out
                round_tests(phase, iteration + 1, checkpoint)
    finally:
        history.close()
    return checkpoint
