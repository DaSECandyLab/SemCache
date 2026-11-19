"""Evaluate a saved value-model checkpoint in separate accuracy-target passes."""

import argparse
import copy
import fcntl
import json
import os
import shutil
import signal
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from planning.execution.cache.reset import clear_prompt_answers
from planning.execution.feedback import reset_serving_cache
from planning.learning.devices import load_inference_model
from planning.observations.cache import CacheHistory
from planning.search.beam import SearchConfig
from planning.search.heuristics import PRIORITY_POLICY
from planning.settings.configuration import (
    ROOT,
    activate_native,
    teacher_path,
)
from planning.settings.parallelism import configure_cpu_resources
from planning.settings.teacher import default_teacher_model, load_teacher
from reference_answers.storage.files import file_digest
from training.artifacts import read, save
from training.calibration.global_profile import GlobalCalibration
from training.reporting.telemetry import TelemetryCounts

from .execution import execute_saved_fallback, render
from .queries import (
    ensure_offline_statistics,
    evaluate_query,
    evaluation_suite,
    wait_for_training,
)

CASES = (
    ("min-20", "fastest", 0.2),
    ("min-50", "fastest", 0.5),
    ("min-80", "fastest", 0.8),
)


def run(
    run_dir,
    output,
    *,
    preserve_cache=False,
    cache_directory=None,
    global_http_limit=None,
    heldout=False,
    references_root=None,
    checkpoint_round=None,
    scheduled_inputs=None,
    runtime_adaptation_path=None,
    model_checkpoint=None,
):
    output.mkdir(parents=True, exist_ok=True)
    guard = (output / "job.lock").open("a")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    resource = (
        ROOT.parent / "experiments/results/semcache/model_training/.collection.lock"
    ).open("a")
    fcntl.flock(resource, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = read(run_dir / "protocol.json")
    status_path = run_dir / "pipeline-status.json"
    training_state = read(
        status_path if status_path.exists() else run_dir / "status.json"
    )
    if training_state["status"] != "completed":
        saved_stop = training_state.get("stop_checkpoint")
        selected_checkpoint = run_dir / (
            f"measured/round-{checkpoint_round or protocol['configuration']['training']['rounds']}/model/value.pt"
        )
        if not (
            training_state["status"] == "stopped"
            and training_state.get("phase") == "checkpoint"
            and training_state.get("fitting_performed") is True
            and saved_stop
            and (run_dir / saved_stop).is_file()
            and selected_checkpoint.is_file()
            and file_digest(run_dir / saved_stop) == file_digest(selected_checkpoint)
        ):
            raise ValueError(
                "Evaluation requires completed training or its saved stop checkpoint"
            )
    settings = copy.deepcopy(protocol["configuration"])
    settings["response_cache"]["directory"] = str(
        cache_directory or output / "prompt-cache"
    )
    settings["response_cache"]["verify_on_second_use"] = False
    settings["inference"].update(plan_generation="heuristic_value")
    activate_native(settings)
    configure_cpu_resources(settings, configure_torch=True)
    models, catalog = (
        read(run_dir / "models.json"),
        read(run_dir / "model-catalog.json"),
    )
    teacher_file = teacher_path(settings)
    if file_digest(teacher_file) != protocol["teacher_config_sha256"]:
        raise ValueError("Training teacher configuration changed")
    teacher = load_teacher(default_teacher_model(teacher_file), models, teacher_file)
    workloads, databases, references, inputs = evaluation_suite(
        run_dir, protocol, heldout=heldout, references_root=references_root
    )
    if scheduled_inputs is not None and inputs != scheduled_inputs:
        raise ValueError("Scheduled evaluation inputs changed")
    count = len(workloads)
    total = count * len(CASES)
    checkpoint_round = checkpoint_round or settings["training"]["rounds"]
    checkpoint = (
        Path(model_checkpoint).resolve()
        if model_checkpoint is not None
        else run_dir / f"measured/round-{checkpoint_round}/model/value.pt"
    )
    ranker = load_inference_model(checkpoint, settings)
    if not ranker.value_available:
        raise ValueError("The final checkpoint has no fitted value heads")
    runtime_adaptation = None
    if runtime_adaptation_path:
        import torch
        from planning.execution.adaptation.monitor import AdaptationSettings

        adaptation = AdaptationSettings(**read(runtime_adaptation_path))
        if not adaptation.enabled:
            raise ValueError("Runtime adaptation configuration is disabled")
        runtime_settings = copy.deepcopy(settings)
        runtime_settings["inference"]["device"] = adaptation.device
        runtime_adaptation = {
            "model": load_inference_model(checkpoint, runtime_settings),
            "config": SearchConfig.configured(settings, "heuristic_value"),
            "settings": adaptation,
        }
        if adaptation.device == "cpu":
            torch.set_num_threads(1)
    profile = GlobalCalibration(
        run_dir / "calibration/global.json", models, teacher
    ).snapshot()
    if global_http_limit is not None:
        if not 1 <= global_http_limit <= 1024:
            raise ValueError("Global HTTP concurrency must be between 1 and 1024")
        os.environ["SEMCACHE_GLOBAL_HTTP_CONCURRENCY"] = str(global_http_limit)
        os.environ["SEMCACHE_GLOBAL_HTTP_AUDIT"] = str(
            output / "http-concurrency.jsonl"
        )
    request = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_digest(checkpoint),
        "settings": settings,
        "cases": CASES,
        "training_run": str(run_dir),
        "generalization_evaluation": heldout,
        **inputs,
        "checkpoint_round": checkpoint_round,
        "model_checkpoint_override": str(checkpoint)
        if model_checkpoint is not None
        else None,
        "use_cache_features": settings.get("value_model", {}).get(
            "use_cache_features", True
        ),
        "training_profile_sha256": file_digest(run_dir / "calibration/global.json"),
        "cache_reset_scope": "case",
        "training_feedback_used": False,
        "runtime_adaptation": read(runtime_adaptation_path)
        if runtime_adaptation_path
        else None,
        "layer_search_protocol": "pareto-layers-final-accuracy-quality-fallback-v4",
        "operator_priority_policy": PRIORITY_POLICY,
        **(
            {"global_http_limit": global_http_limit}
            if global_http_limit is not None
            else {}
        ),
    }
    request = json.loads(json.dumps(request))
    request_path = output / "request.json"
    if request_path.exists() and read(request_path) != request:
        raise ValueError("Evaluation inputs changed")
    save(request_path, request)
    save(
        output / "execution-policy.json",
        {
            "execute_all_selected_plans": True,
            "skipped_plans_are_scored": False,
        },
    )
    # Query-time cache estimation requires statistics tied to the test databases,
    # whose files differ from the training databases even when tables coincide.
    ensure_offline_statistics(settings, databases)
    rows = []
    for case, _, _ in CASES:
        for w in workloads:
            path = output / case / w["id"] / "summary.json"
            if path.exists():
                rows.append(read(path))
    state = {
        "status": "running",
        "pid": os.getpid(),
        "stage": "准备中",
        "preserve_cache": preserve_cache,
        "global_http_limit": global_http_limit,
        "external_teacher_concurrency": settings["execution"].get(
            "max_concurrent_external_teacher_requests", 16
        ),
        "query_count": count,
        "checkpoint_round": checkpoint_round,
        "heldout": heldout,
        "runtime_adaptation": read(runtime_adaptation_path)
        if runtime_adaptation_path
        else None,
    }
    lock, stop = threading.RLock(), threading.Event()
    telemetry = TelemetryCounts()

    def update(**values):
        with lock:
            state.update(
                values,
                updated_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                completed=len(rows),
                total=total,
                pending_fallback_executions=sum(
                    r.get("status") == "infeasible" for r in rows
                ),
            )
            if (
                state.get("stage") == "执行中"
                and state.get("case")
                and state.get("query")
            ):
                path = Path(
                    state.get("live", {}).get("execution_telemetry")
                    or (
                        output
                        / state["case"]
                        / state["query"]
                        / "sql/run-2/telemetry.jsonl"
                    )
                )
                if path.exists():
                    state["live_counts"] = telemetry.read(path)[0]
            save(output / "job-status.json", state)
            save(
                output / "results.json",
                {"queries": rows, "total": total, "training_feedback_used": False},
            )
            render(output, state, rows, workloads, cases=CASES)

    def refresh():
        while not stop.wait(10):
            update()

    reporter = threading.Thread(target=refresh, daemon=True)
    reporter.start()

    def terminate(signum, _frame):
        raise KeyboardInterrupt(f"收到退出信号 {signal.Signals(signum).name}")

    previous_handlers = {
        sig: signal.signal(sig, terminate) for sig in (signal.SIGTERM, signal.SIGHUP)
    }
    try:
        events_path = output / "run-events.json"
        events = read(events_path) if events_path.exists() else []
        events.append(
            {
                "event": "resume" if rows else "start",
                "pid": os.getpid(),
                "time": datetime.now().astimezone().isoformat(),
                "completed": len(rows),
                "preserve_cache": preserve_cache,
            }
        )
        save(events_path, events)
        update()
        for case, preference, minimum in CASES:
            pending = [
                w
                for w in workloads
                if not any(
                    r["case"] == case
                    and r["query"] == w["id"]
                    and r.get("status") != "infeasible"
                    for r in rows
                )
            ]
            if not pending:
                continue
            warm_resume = (output / case / "cache-reset.json").exists()
            update(
                case=case,
                query=None,
                stage="保留本组缓存"
                if preserve_cache or warm_resume
                else "清理本组缓存",
                live={},
                live_counts={},
            )
            if preserve_cache or warm_resume:
                save(
                    output / case / "cache-policy.json",
                    {
                        "policy": "preserve_existing_model_and_prompt_caches",
                        "user_requested": preserve_cache,
                        "existing_case_cache_reset_preserved": warm_resume,
                        "time": datetime.now().astimezone().isoformat(),
                        "resumed_completed": count - len(pending),
                    },
                )
            else:
                reset = {
                    "models": reset_serving_cache(models),
                    "prompt": clear_prompt_answers(),
                    "started_at": datetime.now().astimezone().isoformat(),
                    "resume": len(pending) != count,
                }
                reset_path = output / case / "cache-reset.json"
                if reset_path.exists():
                    reset_path = output / case / "resume-cache-resets.json"
                    resets = read(reset_path) if reset_path.exists() else []
                    save(reset_path, [*resets, reset])
                else:
                    save(reset_path, reset)
            with CacheHistory(":memory:") as history:
                for w in pending:
                    root = output / case / w["id"]
                    update(query=w["id"], live={}, live_counts={})

                    def stage(name, **live):
                        update(stage=name, stage_started_at=time.time(), live=live)

                    started = time.monotonic()
                    try:
                        previous_path = root / "summary.json"
                        saved_fallback = (
                            previous_path.exists()
                            and read(previous_path).get("status") == "infeasible"
                        )
                        if not saved_fallback and (root / "sql").exists():
                            events.append(
                                {
                                    "event": "retry_incomplete_query",
                                    "case": case,
                                    "query": w["id"],
                                    "time": datetime.now().astimezone().isoformat(),
                                }
                            )
                            save(events_path, events)
                            shutil.rmtree(root / "sql")
                        if saved_fallback:
                            row = execute_saved_fallback(
                                w,
                                root,
                                settings=settings,
                                models=models,
                                teacher=teacher,
                                database=databases[w["dataset"]],
                                history=history,
                                minimum=minimum,
                                reference=references[w["id"]],
                                stage=stage,
                                runtime_adaptation=runtime_adaptation,
                            )
                        else:
                            row = evaluate_query(
                                w,
                                root,
                                settings=settings,
                                models=models,
                                catalog=catalog,
                                teacher=teacher,
                                database=databases[w["dataset"]],
                                profile=profile,
                                history=history,
                                ranker=ranker,
                                preference=preference,
                                minimum=minimum,
                                reference=references[w["id"]],
                                stage=stage,
                                runtime_adaptation=runtime_adaptation,
                            )
                    except (
                        OSError,
                        ValueError,
                        KeyError,
                        RuntimeError,
                        TimeoutError,
                    ) as error:
                        row = {
                            "query": w["id"],
                            "status": "failed",
                            "error": str(error),
                        }
                    row.update(case=case, process_seconds=time.monotonic() - started)
                    save(root / "summary.json", row)
                    with lock:
                        rows[:] = [
                            r
                            for r in rows
                            if (r["case"], r["query"]) != (case, w["id"])
                        ]
                        rows.append(row)
                    print(
                        f"{case} {w['public_query_id']}: {row['status']} F1={row.get('f1')} execution={row.get('execution_seconds')}",
                        flush=True,
                    )
                    update(stage="本条完成", live={})
        succeeded = len(rows) == total and all(
            r["status"] == "completed" and r.get("f1") is not None for r in rows
        )
        update(
            status="completed" if succeeded else "finished_with_issues",
            query=None,
            stage="完成",
        )
    except KeyboardInterrupt as error:
        update(status="stopped", error=str(error) or "测试被中断")
        raise
    except Exception as error:
        update(status="error", error=str(error))
        raise
    finally:
        stop.set()
        reporter.join(timeout=2)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        resource.close()
        guard.close()


def main():
    global CASES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=ROOT.parent
        / "experiments/results/semcache/model_training/feedback-training",
    )
    parser.add_argument(
        "--output",
        type=Path,
    )
    parser.add_argument("--preserve-cache", action="store_true")
    parser.add_argument("--cache-directory", type=Path)
    parser.add_argument("--global-http-limit", type=int)
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument("--references", type=Path)
    parser.add_argument("--checkpoint-round", type=int)
    parser.add_argument("--wait-for-training", action="store_true")
    parser.add_argument("--runtime-adaptation-config", type=Path)
    parser.add_argument(
        "--model-checkpoint",
        type=Path,
        help="Optional checkpoint from a separate model run; metadata/query suite still comes from --run",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=[case[0] for case in CASES],
        help="Accuracy groups to run, in the supplied order",
    )
    args = parser.parse_args()
    if args.cases:
        if len(args.cases) != len(set(args.cases)):
            parser.error("--cases must not repeat a group")
        choices = {case[0]: case for case in CASES}
        CASES = tuple(choices[name] for name in args.cases)
    output = args.output or ROOT.parent / "experiments/results/semcache/evaluation" / (
        ("trained-heldout-" if args.heldout else "trained-workload-")
        + datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    schedule = (output / "schedule.lock").open("a")
    fcntl.flock(schedule, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        scheduled_inputs = None
        if args.wait_for_training:
            if args.checkpoint_round is None or args.preserve_cache:
                parser.error(
                    "Scheduled evaluation needs --checkpoint-round and cold case starts"
                )
            scheduled_inputs = wait_for_training(
                args.run.resolve(),
                output,
                checkpoint_round=args.checkpoint_round,
                heldout=args.heldout,
                references_root=args.references,
                global_http_limit=args.global_http_limit,
                cases=CASES,
            )
        run(
            args.run.resolve(),
            output,
            preserve_cache=args.preserve_cache,
            cache_directory=args.cache_directory.resolve()
            if args.cache_directory
            else None,
            global_http_limit=args.global_http_limit,
            heldout=args.heldout,
            references_root=args.references,
            checkpoint_round=args.checkpoint_round,
            scheduled_inputs=scheduled_inputs,
            runtime_adaptation_path=args.runtime_adaptation_config,
            model_checkpoint=args.model_checkpoint,
        )
    except BaseException as error:
        previous_status = (
            read(output / "job-status.json")
            if (output / "job-status.json").exists()
            else {}
        )
        save(
            output / "job-status.json",
            {
                **previous_status,
                "status": "stopped"
                if isinstance(error, KeyboardInterrupt)
                else "error",
                "error": str(error),
                "pid": os.getpid(),
                "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            },
        )
        raise
    finally:
        schedule.close()
