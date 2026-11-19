"""Feedback rollout execution and periodic checkpoint fitting."""

import os
import shutil
import signal
import threading

from planning.execution.timeouts import TimeoutPolicy
from planning.learning.training.data import dataset
from planning.learning.training.targets import TARGET_PROTOCOL
from planning.observations.cache import CacheHistory
from planning.settings.configuration import (
    runtime_path,
)
from reference_answers.storage.files import file_digest
from training.artifacts import read, save
from training.calibration.calibrate import calibrate_models
from training.calibration.global_profile import GlobalCalibration
from training.collection.collector import collect_native, reset_context
from training.collection.feedback import collect_feedback_query, restore_completed_query
from training.collection.policy import ordered_workloads
from training.fitting import fit
from training.workloads.workload_resume import (
    matching_replay,
)

from .cli import render_progress_report
from .preparation import now

ACCURACY_MINIMUMS = (0.2, 0.5, 0.8)
SEARCHES_PER_QUERY = len(ACCURACY_MINIMUMS)


def run_feedback_rounds(ctx):
    output = ctx.output
    settings = ctx.settings
    workloads = ctx.workloads
    models = ctx.models
    catalog = ctx.catalog
    teacher = ctx.teacher
    databases = ctx.databases
    collect_only = ctx.collect_only
    collection_checkpoint = ctx.collection_checkpoint
    recollect_queries = ctx.recollect_queries
    recollection = ctx.recollection
    resume_round = ctx.resume_round
    preserve_resume_cache = ctx.preserve_resume_cache
    total_executions = ctx.total_executions
    excluded_rounds = ctx.excluded_rounds
    answers = ctx.answers
    frozen = ctx.frozen
    recollection_path = ctx.recollection_path
    reference_evidence = ctx.reference_evidence
    teacher_config = ctx.teacher_config
    training_scope = ctx.training_scope
    initial = ctx.initial
    stopping = ctx.stopping
    def report(**values):
        return render_progress_report(ctx, **values)

    def progress(message, **_):
        print(message, flush=True)
        report(last_event=str(message)[:500])

    def refresh():
        while not stopping.wait(10):
            report()

    monitor = threading.Thread(target=refresh, daemon=True)
    monitor.start()

    def interrupt(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    history = CacheHistory(output / "history.sqlite")
    try:
        report()

        def initialize_calibration():
            save(output / "initial-cache-reset.json", reset_context(models, history))

            def calibration_progress(event):
                save(output / "calibration/progress.json", event)
                report()

            return calibrate_models(
                workloads,
                models,
                teacher,
                databases,
                runtime_path(settings, "build"),
                output / "calibration/probes",
                history=history,
                adaptive=settings["sampling"]["adaptive_sketch"],
                parallelism=settings["sampling"]["cpu_parallelism"],
                sample_rows=settings["calibration"]["sample_rows"],
                cohort_seed=settings["calibration"]["cohort_seed"],
                progress=calibration_progress,
            )

        calibration = GlobalCalibration(
            output / "calibration/global.json",
            models,
            teacher,
            initialize=initialize_calibration,
        )

        def saved_request_timings():
            for token in calibration.state["applied"]:
                phase, number, episode = token.split("/", 2)
                if phase != "measured":
                    continue
                path = (
                    output
                    / f"measured/round-{number}/stream"
                    / episode
                    / "online-timing.json"
                )
                if not path.exists():
                    raise ValueError(f"Missing committed request-time evidence: {path}")
                yield from read(path).values()

        if calibration.merge_saved_request_timings(saved_request_timings()):
            progress("已将保存的采样和执行请求耗时合并到全局均值；未发送新请求。")
        checkpoint, replay, replay_sources = initial, [], {}
        for number, cycle in enumerate(settings["training"]["feedback_cycle"], 1):
            directory = output / f"measured/round-{number}"
            directory.mkdir(parents=True, exist_ok=True)
            checkpoint_out = directory / "model/value.pt"
            if (collect_only and (directory / "experience.json").exists()) or (
                not collect_only and checkpoint_out.exists()
            ):
                if number not in excluded_rounds:
                    prior = read(directory / "experience.json")["records"]
                    selected = matching_replay(prior, workloads)
                    replay.extend(selected)
                    replay_sources[number] = len(selected)
                checkpoint = collection_checkpoint if collect_only else checkpoint_out
                continue
            seed = settings["training"]["seed"] + number * 104729
            ordered = ordered_workloads(workloads, "shuffled", seed)
            save(
                directory / "round.json",
                {**cycle, "seed": seed, "query_ids": [w["id"] for w in ordered]},
            )
            save(directory / "workloads.json", {"workloads": workloads})
            completed_prior_rounds = (
                sum(frozen["round_query_counts"][: number - 1]) * SEARCHES_PER_QUERY
            )
            report(
                phase="measured", round=number, current_query=None, completed_queries=0
            )
            keep_cache = (
                preserve_resume_cache
                and number == resume_round
                and (directory / "cache-reset.json").exists()
            )
            if keep_cache:
                events = directory / "cache-resume-events.json"
                prior = read(events) if events.exists() else []
                save(
                    events,
                    [
                        *prior,
                        {
                            "time": now(),
                            "pid": os.getpid(),
                            "policy": "retain_live_serving_kv_and_persisted_history",
                            "reason": "resume the same round after a process/library update",
                        },
                    ],
                )
            else:
                reset = reset_context(models, history)
                save(directory / "cache-reset.json", reset)
            records = []
            for query_index, workload in enumerate(ordered):
                identity = workload["id"]
                destination = directory / "stream" / identity
                restored = restore_completed_query(
                    workload,
                    destination,
                    minimum_accuracies=ACCURACY_MINIMUMS,
                    calibration=calibration,
                    reference_sha256=reference_evidence[identity],
                    round_number=number,
                )
                if restored is not None:
                    records.extend(restored["records"])
                    continue
                if recollect_queries and identity not in recollect_queries:
                    raise ValueError(
                        f"Preserved execution failed validation; refusing to rerun {identity}"
                    )
                report(current_query=identity, completed_queries=query_index)

                def collect(
                    target,
                    rollout_index,
                    selection_policy,
                    *,
                    minimum_accuracy=None,
                    excluded_plan_signatures=(),
                    excluded_structure_signatures=(),
                    excluded_join_positions=(),
                    prepared_statistics=None,
                    completed_times=(),
                ):
                    if target.exists():
                        # A reused sampling snapshot sends no new requests. If
                        # this search never started SQL execution, its discarded
                        # CPU work does not invalidate the live serving cache.
                        unused_search = (
                            keep_cache
                            and (target.parent / "prepared-statistics.json").exists()
                            and not (target / "sql/run-2/query.sql").exists()
                        )
                        if not unused_search:
                            reset_context(models, history)
                        shutil.rmtree(target)
                    return collect_native(
                        workload,
                        models,
                        catalog,
                        databases[workload["dataset"]],
                        runtime_path(settings, "build"),
                        target,
                        calibration_profile=calibration.snapshot(),
                        checkpoint=checkpoint,
                        phase="measured",
                        candidates=settings["search"]["beam_width"],
                        seed=seed + query_index + (rollout_index - 1) * 1000003,
                        history=history,
                        timeout_policy=TimeoutPolicy(
                            settings["training"]["timeout"],
                            settings["training"]["minimum_timeout"],
                            settings["training"]["timeout_slack"],
                        ),
                        sampling_config=settings["sampling"],
                        search_seconds=settings["search"]["training"]["max_seconds"],
                        max_expansions=settings["search"]["training"]["max_expansions"],
                        teacher=teacher,
                        exploration_fraction=settings["training"][
                            "exploration_fraction"
                        ],
                        inference_settings=settings,
                        execution_cpu_parallelism=16,
                        max_concurrent_requests_per_model=16,
                        enforce_model_operator_order=settings["execution"][
                            "enforce_model_operator_order"
                        ],
                        plan_generation="heuristic_value",
                        operator_priority_mode=cycle["operator_priority_mode"],
                        rollout_index=rollout_index,
                        selection_policy=selection_policy,
                        minimum_accuracy=minimum_accuracy,
                        excluded_plan_signatures=excluded_plan_signatures,
                        excluded_structure_signatures=excluded_structure_signatures,
                        excluded_join_positions=excluded_join_positions,
                        searches_per_query=SEARCHES_PER_QUERY,
                        prepared_statistics=prepared_statistics,
                        completed_times=completed_times,
                        cache_segment={
                            "id": f"round-{number}",
                            "boundary": "round_start",
                        },
                        progress=progress,
                    )

                payload = collect_feedback_query(
                    workload,
                    destination,
                    collect=collect,
                    calibration=calibration,
                    answers=answers,
                    teacher_config=teacher_config,
                    round_number=number,
                    completed_before=SEARCHES_PER_QUERY * query_index,
                    update=lambda **values: report(
                        **{
                            k: v
                            for k, v in values.items()
                            if k != "completed_executions"
                        },
                        completed_executions=completed_prior_rounds
                        + values.get(
                            "completed_executions", SEARCHES_PER_QUERY * query_index
                        ),
                    ),
                    progress=progress,
                    minimum_accuracies=ACCURACY_MINIMUMS,
                )
                records.extend(payload["records"])
                report(
                    completed_queries=query_index + 1,
                    completed_executions=completed_prior_rounds
                    + SEARCHES_PER_QUERY * (query_index + 1),
                )
            save(directory / "experience.json", dataset(records))
            replay.extend(records)
            replay_sources[number] = len(records)
            if collect_only:
                if (
                    file_digest(collection_checkpoint)
                    != frozen["collection_checkpoint_sha256"]
                ):
                    raise ValueError(
                        "Collection checkpoint changed during data collection"
                    )
                save(
                    directory / "summary.json",
                    {
                        "round": number,
                        "executions": len(records),
                        "scored": sum(r["accuracy"] is not None for r in records),
                        "collect_only": True,
                        "checkpoint": str(collection_checkpoint),
                        "fitting_performed": False,
                    },
                )
                if recollect_queries:
                    recollection["pending_rounds"] = [
                        n for n in recollection["pending_rounds"] if n != number
                    ]
                    save(recollection_path, recollection)
                continue
            save(
                directory / "model/experience-sources.json",
                {
                    "included_rounds": sorted(replay_sources),
                    "excluded_rounds": sorted(excluded_rounds),
                    "records_by_round": replay_sources,
                    "records": len(replay),
                    "initial_checkpoint": str(checkpoint),
                },
            )
            report(phase="value_update", current_query=None)
            fit(
                replay,
                directory / "model",
                epochs=settings["training"]["epochs"],
                checkpoint=checkpoint,
                seed=seed,
                source="measured",
                settings=settings,
            )
            checkpoint = checkpoint_out
            save(
                directory / "summary.json",
                {
                    "round": number,
                    "executions": len(records),
                    "scored": sum(r["accuracy"] is not None for r in records),
                    "mean_f1": sum(
                        r["accuracy"] for r in records if r["accuracy"] is not None
                    )
                    / max(1, sum(r["accuracy"] is not None for r in records)),
                    "checkpoint": str(checkpoint),
                },
            )
        if not collect_only:
            shutil.copyfile(checkpoint, output / "value.pt")
            save(
                output / "model.json",
                {
                    "checkpoint": str(output / "value.pt"),
                    "initialization": "random",
                    "pretraining": False,
                    "training_queries": list(frozen["workload_sql_sha256"]),
                    "training_scope": training_scope,
                    "target_protocol": TARGET_PROTOCOL,
                    "source_checkpoint": str(checkpoint),
                },
            )
        if recollect_queries:
            for relative, expected in recollection["preserved_experiences"].items():
                if file_digest(output / relative) != expected:
                    raise ValueError(
                        f"Preserved execution changed during recollection: {relative}"
                    )
            recollection.update(status="completed", finished_at=now())
            save(recollection_path, recollection)
        report(
            status="completed",
            phase="completed",
            current_query=None,
            completed_executions=total_executions,
            finished_at=now(),
        )
    except BaseException as error:
        report(
            status="stopped" if isinstance(error, KeyboardInterrupt) else "failed",
            error=str(error),
        )
        raise
    finally:
        stopping.set()
        monitor.join(timeout=2)
        history.close()
