"""Evaluate a saved value-model checkpoint in separate accuracy-target passes."""

import copy
import fcntl
import json
import os
import time
from contextlib import closing, nullcontext
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from planning.backends.execution import DuckDBEvaluator
from planning.backends.sampling import NativeSamplingFactory
from planning.execution.feedback import freeze_problem
from planning.search.beam import ParetoBeamSearch, SearchConfig
from planning.settings.configuration import (
    ROOT,
    runtime_path,
    teacher_path,
)
from planning.settings.modes import profile_for_mode, sampling_for_mode
from planning.state import SearchProblem, digest
from reference_answers.storage.files import file_digest
from training.artifacts import read, save
from training.collection.features import normalize_explain
from training.references.answers import TrainingAnswers
from training.reporting.telemetry import MODELS
from training.workloads.query_manifest import heldout_manifest, ordered_heldout_manifest

CASES = (
    ("min-20", "fastest", 0.2),
    ("min-50", "fastest", 0.5),
    ("min-80", "fastest", 0.8),
)


def digest_sql(sql):
    import hashlib

    return hashlib.sha256(sql.encode()).hexdigest()


def evaluation_suite(run_dir, protocol, *, heldout=False, references_root=None):
    """Resolve and validate test SQL and saved answers without calling models."""
    if heldout:
        workloads = []
        for item in ordered_heldout_manifest(heldout_manifest()):
            short = item["id"].removeprefix("translated/semcache/")
            domain = short.split("/")[0]
            workloads.append(
                {
                    **item,
                    "sql": Path(item["path"]).read_text(),
                    "dataset": {"movies": "movies_nirvana"}.get(domain, domain),
                    "public_query_id": short,
                    "split": "test",
                    "group": item["id"],
                }
            )
    else:
        workloads = read(run_dir / "workloads.json")["workloads"]
        if any(
            digest_sql(w["sql"]) != protocol["workload_sql_sha256"][w["id"]]
            for w in workloads
        ):
            raise ValueError("The frozen training-query suite changed")
    databases = {
        name: ROOT.parent / "benchmarks/databases/semcache" / f"{name}.duckdb"
        for name in sorted({w["dataset"] for w in workloads})
    }
    reference_root = Path(references_root or protocol["reference_root"]).resolve()
    answers = TrainingAnswers(
        reference_root, workloads, databases, allow_subset=heldout
    )
    references = {w["id"]: answers.load(w) for w in workloads}
    reference_hashes = {
        key: value["answer_sha256"] for key, value in references.items()
    }
    if not heldout and reference_hashes != protocol["answer_sha256"]:
        raise ValueError("The frozen training answers changed")
    contract = {
        "queries": {w["id"]: w["sql_sha256"] for w in workloads},
        "databases": {name: str(path) for name, path in databases.items()},
        "database_sha256": {
            name: file_digest(path) for name, path in databases.items()
        },
        "reference_root": str(reference_root),
        "reference_contract_sha256": file_digest(reference_root / "contract.json"),
        "reference_sha256": reference_hashes,
    }
    return workloads, databases, references, contract


def ensure_offline_statistics(settings, databases):
    """Build missing full test-table statistics before opening a search client."""
    from planning.backends.duckdb import DuckDBConnection
    from planning.observations.offline_statistics import (
        OfflineStatisticsStore,
        build_database,
    )

    build = runtime_path(settings, "build")
    directory = settings["sampling"]["offline_statistics_dir"]
    for database in databases.values():
        store = OfflineStatisticsStore(directory, database)
        with closing(DuckDBConnection(build, database)) as db:
            tables = [
                row[0]
                for row in db.execute(
                    "SELECT table_name FROM duckdb_tables() WHERE schema_name='main'"
                )
            ]
        missing = [name for name in tables if store.load("main", name) is None]
        if missing:
            build_database(
                database,
                directory,
                build,
                tables=missing,
                config=settings["sampling"].get("adaptive_sketch"),
                parallelism=settings["statistics"]["build_parallelism"],
            )


def wait_for_training(
    run_dir,
    output,
    *,
    checkpoint_round,
    heldout,
    references_root,
    global_http_limit,
    cases=CASES,
):
    """Freeze inputs now; wait for successful fitting and released training locks."""
    protocol = read(run_dir / "protocol.json")
    if checkpoint_round != protocol["configuration"]["training"]["rounds"]:
        raise ValueError("Scheduled evaluation must follow the final training round")
    _, _, _, inputs = evaluation_suite(
        run_dir, protocol, heldout=heldout, references_root=references_root
    )
    request = {
        "training_run": str(run_dir),
        "checkpoint_round": checkpoint_round,
        "checkpoint": str(
            run_dir / f"measured/round-{checkpoint_round}/model/value.pt"
        ),
        "inputs": inputs,
        "cases": list(cases),
        "global_http_limit": global_http_limit,
        "prompt_cache": True,
        "verify_on_second_use": False,
        "cache_reset_scope": "case",
        "training_feedback_used": False,
        "teacher_config_sha256": protocol["teacher_config_sha256"],
    }
    request = json.loads(json.dumps(request))
    path = output / "schedule-request.json"
    if path.exists() and read(path) != request:
        raise ValueError("Scheduled evaluation inputs changed")
    save(path, request)
    teacher_concurrency = (
        protocol.get("configuration", {})
        .get("execution", {})
        .get("max_concurrent_external_teacher_requests", 16)
    )
    count = len(inputs["queries"])
    while True:
        training = read(run_dir / "pipeline-status.json")
        state = {
            "status": "waiting_for_training",
            "pid": os.getpid(),
            "training_status": training["status"],
            "training_phase": training.get("phase"),
            "training_round": training.get("round"),
            "after_round": checkpoint_round,
            "completed": 0,
            "total": count * len(cases),
            "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        }
        save(output / "job-status.json", state)
        (output / "测试结果.md").write_text(
            f"# 训练后 {count} 条查询测试\n\n"
            f"状态：等待第 {checkpoint_round} 轮训练完成。当前训练第 {training.get('round')} 轮，"
            f"阶段 {training.get('phase')}，状态 {training['status']}。\n\n"
            f"使用第 {checkpoint_round} 轮最终模型，依次完整运行 "
            + "、".join(f"{case}（下限 {minimum:.0%}）" for case, _, minimum in cases)
            + f"，共 {count * len(cases)} 条记录。\n\n"
            f"本地模型全局 HTTP 并发 {global_http_limit}（外网 Teacher 独立并发上限 {teacher_concurrency}）；CPU 16；查询串行。每组开始前清空 prompt cache 和模型 KV cache，组内连续复用。\n\n"
            "测试开启 prompt cache、关闭二次核对；每条执行后计算 F1；复用已保存的 teacher 答案。\n"
        )
        if training["status"] == "completed":
            if training.get("round") != checkpoint_round:
                raise ValueError("Training completed at a different round")
            checkpoint = run_dir / f"measured/round-{checkpoint_round}/model/value.pt"
            if (
                not checkpoint.is_file()
                or not checkpoint.with_name("training.json").is_file()
            ):
                raise ValueError("Final training checkpoint/report is missing")
            if (
                file_digest(
                    teacher_path(read(run_dir / "protocol.json")["configuration"])
                )
                != request["teacher_config_sha256"]
            ):
                raise ValueError("Sampling teacher configuration changed while waiting")
            # The completed status can precede process teardown by a few ms.
            with (run_dir / "pipeline.lock").open("a") as owner:
                try:
                    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(1)
                    continue
            _, _, _, current = evaluation_suite(
                run_dir,
                read(run_dir / "protocol.json"),
                heldout=heldout,
                references_root=references_root,
            )
            if current != inputs:
                raise ValueError(
                    "Test SQL, databases or reference answers changed while waiting"
                )
            return inputs
        # Stopped/failed training remains pending; it must never launch evaluation.
        time.sleep(10)


def evaluate_query(
    workload,
    root,
    *,
    settings,
    models,
    catalog,
    teacher,
    database,
    profile,
    history,
    ranker,
    preference,
    minimum,
    reference,
    stage,
    plan_only=False,
    prepared_statistics=None,
    runtime_adaptation=None,
):
    root.mkdir(parents=True, exist_ok=True)
    sql = workload["sql"]
    (root / "query.sql").write_text(sql)
    build = runtime_path(settings, "build")
    sampling = sampling_for_mode(settings["sampling"], "heuristic_value")
    with DuckDBEvaluator(
        build,
        teacher.transport_models(models),
        sql,
        database=database,
        output=root / "sql",
        history_sink=history,
        timeout=600,
        parallelism=settings["execution"]["cpu_parallelism"],
        max_concurrent_requests_per_model=settings["execution"][
            "max_concurrent_requests_per_model"
        ],
        enforce_model_operator_order=settings["execution"][
            "enforce_model_operator_order"
        ],
    ) as evaluator:
        stage("准备中")
        query = normalize_explain(evaluator.run(explain=True))
        save(root / "query_ir.json", query)
        query_profile = copy.deepcopy(profile)
        constraints = {"allowed_models": list(models)}
        if minimum is not None:
            constraints["a_min"] = minimum
        query_profile.update(models=catalog, constraints=constraints)
        teacher.apply_profile(query_profile)
        if prepared_statistics is not None:
            from training.collection.prepared_sampling import merge_prepared_statistics

            query_profile = merge_prepared_statistics(
                query, query_profile, prepared_statistics
            )
        problem = SearchProblem(
            query, profile_for_mode(query_profile, query, "heuristic_value")
        )
        factory = NativeSamplingFactory(
            build,
            models,
            database=database,
            history=history,
            config=sampling,
            teacher=teacher,
            catalog=catalog,
        )
        stage("采样与搜索中")
        started = time.monotonic()
        try:
            if prepared_statistics is not None:
                from planning.kernels.sampling import prepare_profile

                sampler_context = nullcontext(prepare_profile(problem))
            else:
                sampler_context = factory(problem)
            with sampler_context as sampler:
                problem.sampler = sampler
                search = ParetoBeamSearch(
                    ranker,
                    SearchConfig.configured(
                        settings, "heuristic_value", preference=preference
                    ).for_problem(problem),
                )
                result = search.run(problem)
                beam_seconds = time.monotonic() - started
                latency = (
                    sampler.request_latency()
                    if prepared_statistics is None
                    else {
                        "http_requests_by_model": {},
                        "reported_requests": 0,
                        "source": "reused same-query frozen operator statistics; no sampling requests",
                    }
                )
                save(
                    root / "sampling.json",
                    {"calls": sampler.calls, "request_latency": latency},
                )
                save(
                    root / "search.json",
                    {k: v for k, v in result.items() if k != "states"},
                )
                row = {
                    "query": workload["id"],
                    "beam_seconds": beam_seconds,
                    "minimum_accuracy": minimum,
                    "preference": preference,
                    "predicted_execution_seconds": None,
                    "predicted_f1": None,
                    "execution_seconds": None,
                    "f1": None,
                    "execution_http_requests_by_model": dict.fromkeys(MODELS, 0),
                    "sampling_http_requests_by_model": latency[
                        "http_requests_by_model"
                    ],
                    "constraint_satisfied": result.get("constraint_satisfied"),
                    "candidates": len(result["complete"]),
                    "expansions": result["expansions"],
                    "layer_ranking": result["layer_ranking"],
                    "accuracy_fallback": result.get("accuracy_fallback", False),
                    "selection_basis": result.get("selection_basis"),
                }
                if not result["states"]:
                    row.update(status="infeasible", reason=result.get("reason"))
                    save(root / "profile.json", problem.profile)
                    return row
                selected = search.choose_state(problem, result, preference)
                assessment = next(
                    (
                        a
                        for a in result.get("assessments", [])
                        if a.get("plan_key") == digest(selected.history)
                    ),
                    {},
                )
                row.update(
                    predicted_execution_seconds=assessment.get("time_prediction"),
                    predicted_f1=assessment.get("accuracy"),
                )
                frozen, states = freeze_problem(problem, [selected])
                spec = frozen.spec(states[0])
                save(root / "plan.json", spec)
                save(root / "profile.json", frozen.profile)
                if runtime_adaptation:
                    from planning.execution.adaptation.profile import initial_references

                    runtime_adaptation = {
                        **runtime_adaptation,
                        "initial_references": initial_references(frozen, states[0]),
                    }
        finally:
            factory.close()
        if plan_only:
            row.update(status="planned", execution_skipped="plan_only")
            save(root / "selection.json", row)
            return row
        from .execution import execute_selected

        return execute_selected(
            workload,
            root,
            evaluator,
            query,
            spec,
            row,
            reference,
            stage,
            root / "sql/run-2/telemetry.jsonl",
            runtime_adaptation={**runtime_adaptation, "profile": frozen.profile}
            if runtime_adaptation
            else None,
        )
