"""Run requested fixed-model diagnostics at a completed training-round boundary."""

import copy
import fcntl
import gc
import hashlib
import shutil
import time
from pathlib import Path

from experiments.system_comparison.accuracy.matching import score_non_free_text
from planning.backends.execution import DuckDBEvaluator
from planning.backends.sampling import NativeSamplingFactory
from planning.execution.cache_reset import clear_prompt_answers
from planning.execution.feedback import reset_serving_cache
from planning.observations.cache import CacheHistory
from planning.observations.statistics import ingest_history
from planning.search.beam import ParetoBeamSearch, SearchConfig
from planning.search.heuristics import fixed_layout
from planning.settings.configuration import runtime_path
from planning.settings.modes import profile_for_mode, sampling_for_mode

from training.artifacts import read, save
from training.collection.features import normalize_explain
from training.references.scoring import candidate_result


def fingerprint(value):
    import json

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def render(root, request, state, rows):
    models = request["models"]
    by_key = {(r["model"], r["query"]): r for r in rows}
    lines = [
        "# 第一轮后固定模型启发式测试",
        "",
        f"状态：{state['status']}；已完成 {len(rows)}/{len(request['queries']) * len(models)}。",
        f"当前模型：{state.get('model') or '-'}；查询：{state.get('query') or '-'}。",
        "",
        "模式：heuristic_requests，固定 GOTDF、单模型、fastest；两次使用相同查询顺序。",
        f"CPU {state.get('cpu_threads', 16)} 线程，每模型并发 {state.get('model_concurrency', 16)}，"
        "查询串行；每个模型测试前清一次模型与响应缓存，轮内连续使用。",
        "中断续跑保留已完成结果，从未完成查询冷启动；缓存边界记录在各模型的 cache-reset.json。",
        "F1 直接比较已保存的 teacher 答案；这些测试结果不加入训练样本或全局训练校准。",
        "F1 范围为 0～1，1 表示结果完全一致，0 表示没有匹配行；超时或未评分显示 —，不计作 0。",
        "",
    ]
    for model in models:
        scored = [
            r["f1"] for r in rows if r["model"] == model and r.get("f1") is not None
        ]
        mean = f"{sum(scored) / len(scored):.4f}" if scored else "—"
        lines.append(
            f"- {model}：已评分 {len(scored)}/{len(request['queries'])}；"
            f"平均 F1 **{mean}**；零分 {sum(value == 0 for value in scored)} 条。"
        )
    lines.append("")
    headers = ["查询"]
    for model in models:
        label = model.removeprefix("gemma4-").upper()
        headers.extend(
            f"{label} {metric}" for metric in ("Beam(s)", "执行(s)", "F1", "HTTP")
        )
    lines += ["| " + " | ".join(headers) + " |", "|---|" + "---:|" * (len(headers) - 1)]
    for query in request["queries"]:
        cells = [query.split("-train-", 1)[-1]]
        for model in models:
            row = by_key.get((model, query), {})
            cells.extend(
                f"{row[key]:.{digits}f}" if row.get(key) is not None else "—"
                for key, digits in (
                    ("beam_seconds", 2),
                    ("execution_seconds", 2),
                    ("f1", 4),
                )
            )
            cells.append(str(row.get("http_requests", "—")))
        lines.append("| " + " | ".join(cells) + " |")
    if state.get("error"):
        lines += ["", "错误：" + state["error"]]
    from experiments.semcache.monitoring.files import publish

    publish(root / "progress.md", "\n".join(lines) + "\n")


def evaluate_query(
    workload,
    model,
    root,
    settings,
    models,
    catalog,
    teacher,
    database,
    profile,
    history,
    reference,
):
    from planning.state import SearchProblem

    root.mkdir(parents=True, exist_ok=True)
    (root / "query.sql").write_text(workload["sql"])
    build = runtime_path(settings, "build")
    sampling = sampling_for_mode(
        settings["sampling"], "heuristic_requests", fixed_model_id=model
    )
    with DuckDBEvaluator(
        build,
        teacher.transport_models(models),
        workload["sql"],
        database=database,
        output=root / "sql",
        history_sink=history,
        timeout=settings["training"]["timeout"],
        parallelism=settings["execution"]["cpu_parallelism"],
        max_concurrent_requests_per_model=settings["execution"][
            "max_concurrent_requests_per_model"
        ],
        enforce_model_operator_order=settings["execution"][
            "enforce_model_operator_order"
        ],
    ) as evaluator:
        query = normalize_explain(evaluator.run(explain=True))
        query_profile = copy.deepcopy(profile)
        query_profile.update(models=catalog, constraints={"allowed_models": [model]})
        teacher.apply_profile(query_profile)
        problem = SearchProblem(
            query,
            profile_for_mode(
                query_profile,
                query,
                "heuristic_requests",
                fixed_model_id=model,
            ),
        )
        save(root / "query_ir.json", query)
        with NativeSamplingFactory(
            build,
            models,
            database=database,
            history=history,
            config=sampling,
            teacher=teacher,
            catalog=catalog,
        )(problem) as sampler:
            problem.sampler = sampler
            search = ParetoBeamSearch(
                None, SearchConfig.configured(settings, "heuristic_requests")
            )
            begin = time.monotonic()
            result = search.run(problem)
            beam_seconds = time.monotonic() - begin
            spec = search.choose(problem, result, "fastest")
            expected_layouts = {
                op["operator_id"]: fixed_layout(op)
                for op in query["semantic_operators"]
            }
            if any(
                a["model_id"] != model
                or a["layout"] != expected_layouts[a["operator_id"]]
                for a in spec["actions"]
            ):
                raise ValueError("Diagnostic escaped its fixed model/layout contract")
            calls = dict(sampler.calls)
            save(root / "sampling.json", {"calls": calls})
            save(root / "profile.json", problem.profile)
            save(
                root / "search.json", {k: v for k, v in result.items() if k != "states"}
            )
            save(root / "plan.json", spec)
        actual = evaluator.run(
            spec, source_query=query, allow_timeout=True, include_rows=False
        )
        ingest_history(
            actual["events"], history, adaptive=sampling.get("adaptive_sketch")
        )
        row = {
            "query": workload["id"],
            "model": model,
            "status": actual["status"],
            "beam_seconds": beam_seconds,
            "execution_seconds": actual["seconds"],
            "result_artifact": actual["artifact"],
            "f1": None,
            "sampling_http_requests": calls.get(
                "model_http_requests", calls.get("model_requests", 0)
            ),
            "execution_http_requests": sum(
                e["event"] in ("sampling_llm_submit", "sampling_llm_retransmit")
                for e in actual["events"]
            ),
        }
        row["http_requests"] = (
            row["sampling_http_requests"] + row["execution_http_requests"]
        )
        if actual["status"] == "completed":
            observed = candidate_result(
                actual["artifact"], reference["result"]["columns"]
            )
            metrics = score_non_free_text(observed, reference["result"], workload["id"])
            save(root / "accuracy.json", metrics)
            row.update(
                f1=metrics["accuracy"],
                result_rows=metrics["result_rows"],
                reference_rows=metrics["reference_rows"],
                matched_rows=metrics["matched_rows"],
            )
        return row


def run_requested(
    run,
    *,
    phase,
    round_number,
    checkpoint,
    settings,
    workloads,
    models,
    catalog,
    teacher,
    databases,
    profile,
    answers,
    update,
):
    root = Path(run) / "round-tests"
    request_path = root / "request.json"
    if not request_path.exists():
        return
    request = read(request_path)
    if request.get("phase") != phase or request.get("after_round") != round_number:
        return
    if set(request) != {"phase", "after_round", "models", "queries"}:
        raise ValueError("Invalid round-test request")
    by_id = {w["id"]: w for w in workloads}
    if (
        not request["models"]
        or len(set(request["models"])) != len(request["models"])
        or not set(request["models"]) <= set(models)
        or len(request["queries"]) != len(by_id)
        or set(request["queries"]) != set(by_id)
    ):
        raise ValueError(
            "Round tests must use the exact frozen workload and available models"
        )
    if any(
        Path(name).name != name or name in (".", "..")
        for name in (*request["models"], *request["queries"])
    ):
        raise ValueError("Round diagnostic names must be single path components")
    if answers is None or not Path(checkpoint).is_file():
        raise ValueError(
            "Round tests require completed fitting and frozen teacher answers"
        )
    request_sha = fingerprint(request)
    checkpoint_sha = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    previous = read(root / "status.json") if (root / "status.json").exists() else {}
    if previous.get("status") in ("completed", "completed_with_issues"):
        if (
            previous.get("request_sha256") != request_sha
            or previous.get("checkpoint_sha256") != checkpoint_sha
        ):
            raise ValueError("Completed round-test request or checkpoint changed")
        return
    references = {key: answers.load(by_id[key]) for key in request["queries"]}
    if any(r["basis"] != "final_output" for r in references.values()):
        raise ValueError(
            "Round diagnostics currently require non-aggregate training outputs"
        )
    contract = {
        "request": request,
        "checkpoint_sha256": checkpoint_sha,
        "sql_sha256": {
            key: hashlib.sha256(by_id[key]["sql"].encode()).hexdigest()
            for key in request["queries"]
        },
        "reference_sha256": {
            key: ref["answer_sha256"] for key, ref in references.items()
        },
        "profile_sha256": fingerprint(profile),
        "settings_sha256": fingerprint(settings),
    }
    if (root / "contract.json").exists() and read(root / "contract.json") != contract:
        raise ValueError("Round diagnostic inputs changed")
    save(root / "contract.json", contract)
    state = {
        "status": "running",
        "request_sha256": request_sha,
        "checkpoint_sha256": checkpoint_sha,
        "model": None,
        "query": None,
        "cpu_threads": settings["execution"]["cpu_parallelism"],
        "model_concurrency": settings["execution"]["max_concurrent_requests_per_model"],
    }
    rows = read(root / "results.json") if (root / "results.json").exists() else []

    def publish():
        save(root / "results.json", rows)
        save(root / "status.json", state)
        render(root, request, state, rows)

    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            publish()
            for model in request["models"]:
                if all(
                    any(r["model"] == model and r["query"] == key for r in rows)
                    for key in request["queries"]
                ):
                    continue
                scoped = copy.deepcopy(settings)
                scoped["search"]["fixed_model_id"] = model
                # Each model run starts cold, including a resumed partial run.
                save(
                    root / model / "cache-reset.json",
                    {
                        "model_cache": reset_serving_cache(models),
                        "prompt_cache": clear_prompt_answers(),
                        "scope": "before_model_run",
                        "resume": any(r["model"] == model for r in rows),
                    },
                )
                with CacheHistory(":memory:") as history:
                    for key in request["queries"]:
                        if any(r["model"] == model and r["query"] == key for r in rows):
                            continue
                        state.update(model=model, query=key)
                        update(
                            phase="round_tests",
                            current_query=key,
                            test_model=model,
                            completed_test_queries=len(rows),
                            total_test_queries=len(request["models"]) * len(by_id),
                        )
                        publish()
                        query_root = root / model / key
                        if (query_root / "result.json").is_file():
                            row = read(query_root / "result.json")
                            if row.get("model") != model or row.get("query") != key:
                                raise ValueError(
                                    "Saved diagnostic result has a different identity"
                                )
                            rows.append(row)
                            publish()
                            continue
                        if query_root.exists():
                            if (
                                query_root.is_symlink()
                                or not query_root.resolve().is_relative_to(
                                    root.resolve()
                                )
                            ):
                                raise ValueError(
                                    "Diagnostic output leaves its owned directory"
                                )
                            shutil.rmtree(query_root)
                        row = evaluate_query(
                            by_id[key],
                            model,
                            query_root,
                            scoped,
                            {model: models[model]},
                            {model: catalog[model]},
                            teacher,
                            databases[by_id[key]["dataset"]],
                            profile,
                            history,
                            references[key],
                        )
                        rows.append(row)
                        save(root / model / key / "result.json", row)
                        publish()
                        gc.collect()
            state.update(
                status="completed"
                if all(r["status"] == "completed" for r in rows)
                else "completed_with_issues",
                model=None,
                query=None,
            )
            publish()
            update(
                phase="round_tests_complete",
                current_query=None,
                test_model=None,
                completed_test_queries=len(rows),
                total_test_queries=len(request["models"]) * len(by_id),
            )
        except BaseException as error:
            state.update(
                status="stopped" if isinstance(error, KeyboardInterrupt) else "error",
                error=str(error),
            )
            publish()
            raise


def main():
    """Continue requested diagnostics independently, leaving training stopped."""
    import argparse
    import os
    import signal

    from planning.settings.configuration import (
        ROOT,
        activate_native,
        load_config,
        teacher_path,
    )
    from planning.settings.parallelism import configure_cpu_resources
    from planning.settings.teacher import default_teacher_model, load_teacher

    from training.calibration.global_profile import GlobalCalibration
    from training.references.answers import TrainingAnswers

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    root = run / "round-tests"
    request = read(root / "request.json")
    configuration = run / "resume-config.json"
    if not configuration.exists():
        configuration = run / "training-config.json"
    settings = load_config(configuration)
    activate_native(settings)
    configure_cpu_resources(settings)
    models, catalog = read(run / "models.json"), read(run / "model-catalog.json")
    teacher_file = teacher_path(settings)
    teacher = load_teacher(default_teacher_model(teacher_file), models, teacher_file)
    workloads = read(run / "workloads.json")["workloads"]
    databases = read(run / "source-data.json")["databases"]
    reference_root = read(run / "reference_answers_path.json")["root"]
    answers = TrainingAnswers(reference_root, workloads, databases, allow_subset=True)
    profile = GlobalCalibration(
        run / "calibration/global.json", models, teacher
    ).snapshot()

    def interrupt(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    with (ROOT.parent / "experiments/results/semcache/model_training/.collection.lock").open(
        "a"
    ) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (root / "process.pid").write_text(str(os.getpid()) + "\n")

        standalone = {"pid": os.getpid(), "training_resumed": False}

        def update(**values):
            standalone.update(values)
            save(root / "standalone-status.json", standalone)

        update(status="running")
        try:
            run_requested(
                run,
                phase=request["phase"],
                round_number=request["after_round"],
                checkpoint=run
                / request["phase"]
                / f"round-{request['after_round']}"
                / "model/value.pt",
                settings=settings,
                workloads=workloads,
                models=models,
                catalog=catalog,
                teacher=teacher,
                databases=databases,
                profile=profile,
                answers=answers,
                update=update,
            )
            update(status=read(root / "status.json")["status"])
        except BaseException as error:
            update(
                status="stopped" if isinstance(error, KeyboardInterrupt) else "error",
                error=str(error),
            )
            raise


if __name__ == "__main__":
    main()
