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
from collections import Counter
from contextlib import closing, nullcontext
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from experiments.system_comparison.accuracy.matching import score_non_free_text
from planning.backends.execution import DuckDBEvaluator
from planning.backends.sampling import NativeSamplingFactory
from planning.execution.cache_reset import clear_prompt_answers
from planning.execution.feedback import freeze_problem, reset_serving_cache
from planning.learning.devices import load_inference_model
from planning.observations.cache import CacheHistory
from planning.search.beam import ParetoBeamSearch, SearchConfig
from planning.search.heuristics import PRIORITY_POLICY
from planning.settings.configuration import (
    ROOT,
    activate_native,
    runtime_path,
    teacher_path,
)
from planning.settings.modes import profile_for_mode, sampling_for_mode
from planning.settings.parallelism import configure_cpu_resources
from planning.settings.teacher import default_teacher_model, load_teacher
from planning.state import SearchProblem, digest
from reference_answers.sql.aggregate import capture_plan
from reference_answers.storage.files import file_digest

from training.artifacts import read, save
from training.calibration.global_profile import GlobalCalibration
from training.collection.features import normalize_explain
from training.collection.policy import appstore_filter_before_join
from training.query_manifest import heldout_manifest, ordered_heldout_manifest
from training.references.answers import TrainingAnswers
from training.references.scoring import candidate_result
from training.reporting.telemetry import MODELS, TelemetryCounts

CASES = (
    ("min-20", "fastest", 0.2),
    ("min-50", "fastest", 0.5),
    ("min-80", "fastest", 0.8),
)


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
    run_dir, output, *, checkpoint_round, heldout, references_root, global_http_limit
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
        "cases": list(CASES),
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
            "total": count * len(CASES),
            "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        }
        save(output / "job-status.json", state)
        (output / "测试结果.md").write_text(
            f"# 训练后 {count} 条查询测试\n\n"
            f"状态：等待第 {checkpoint_round} 轮训练完成。当前训练第 {training.get('round')} 轮，"
            f"阶段 {training.get('phase')}，状态 {training['status']}。\n\n"
            f"使用第 {checkpoint_round} 轮最终模型，依次完整运行 20%、50%、80% 三组，共 {count * len(CASES)} 条记录。\n\n"
            f"全局 HTTP 并发 {global_http_limit}；CPU 16；查询串行。每组开始前清空 prompt cache 和模型 KV cache，组内连续复用。\n\n"
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


def appstore_plan_skip_reason(workload, query, spec):
    """User's execution gate: Appstore must join before semantic filtering."""
    if workload["dataset"] == "appstore" and appstore_filter_before_join(query, spec):
        return "appstore 语义过滤先于关系连接，按用户要求跳过执行"
    return None


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
                    ),
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
                    from planning.execution.adaptation_profile import initial_references

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


def execute_selected(
    workload,
    root,
    evaluator,
    query,
    spec,
    row,
    reference,
    stage,
    telemetry,
    runtime_adaptation=None,
):
    reason = appstore_plan_skip_reason(workload, query, spec)
    if reason:
        row.update(status="skipped_plan", reason=reason)
        save(root / "selection.json", row)
        return row
    row["execution_telemetry"] = str(telemetry)
    save(root / "selection.json", row)
    stage("执行中", **row)
    runtime_arguments = {}
    if runtime_adaptation:
        from planning.backends.execution.evaluator import ExplicitPolicy
        from planning.execution.adaptation_service import RuntimeExecutionPolicy

        runtime_arguments = {
            "policy_override": RuntimeExecutionPolicy(
                ExplicitPolicy(spec, query),
                runtime_adaptation["profile"],
                runtime_adaptation["model"],
                runtime_adaptation["config"],
                runtime_adaptation["settings"],
                log_path=root / "runtime-optimizer.jsonl",
                initial_references=runtime_adaptation.get("initial_references"),
            ),
            "runtime_options": {
                "version": 2,
                "value_policy": {
                    "max_switches": runtime_adaptation["settings"].max_switches,
                    "estimated_input_rows": {
                        str(oid): reference["estimated_input_rows"]
                        for oid, reference in runtime_adaptation.get(
                            "initial_references", {}
                        ).items()
                    },
                    "timeout_ms": 10000,
                },
            },
        }
    actual = evaluator.run(
        spec,
        source_query=query,
        allow_timeout=True,
        capture_aggregate_input=capture_plan(workload["sql"]) is not None,
        include_rows=False,
        **runtime_arguments,
    )
    counts = Counter(
        e.get("model_id")
        for e in actual["events"]
        if e.get("event") in ("sampling_llm_submit", "sampling_llm_retransmit")
    )
    row.update(
        status=actual["status"],
        execution_seconds=actual["seconds"],
        result_artifact=actual["artifact"],
        execution_http_requests_by_model={m: counts[m] for m in MODELS},
    )
    if runtime_adaptation:
        runtime_events = [
            e for e in actual["events"] if e.get("event", "").startswith("runtime_")
        ]
        save(root / "runtime.json", runtime_events)
        row["runtime"] = {
            "chunks": sum(
                e["event"] == "runtime_chunk_completed" for e in runtime_events
            ),
            "checks": sum(
                e["event"] == "runtime_adaptation_check" for e in runtime_events
            ),
            "triggered": sum(
                bool(e.get("diagnostics", {}).get("triggers"))
                for e in runtime_events
                if e["event"] == "runtime_adaptation_check"
            ),
            "updates": sum(
                e["event"] == "runtime_plan_update_applied" for e in runtime_events
            ),
            "operator_changes": sum(
                e["event"] == "runtime_update" for e in runtime_events
            ),
            "rejected": sum(
                e["event"] == "runtime_value_rejected" for e in runtime_events
            ),
            "beam_width": runtime_adaptation["settings"].beam_width,
            "sampling_requests": 0,
            "topology_search": False,
        }
    if actual["status"] == "completed":
        stage("计算 F1", **row)
        aggregate = reference["basis"] == "pre_aggregate_rows"
        expected = reference["pre_aggregate"] if aggregate else reference["result"]
        observed = candidate_result(
            actual["artifact"], expected["columns"], aggregate=aggregate
        )
        metrics = score_non_free_text(observed, expected, workload["public_query_id"])
        save(root / "accuracy.json", metrics)
        row.update(
            f1=metrics["accuracy"],
            result_rows=metrics["result_rows"],
            reference_rows=metrics["reference_rows"],
            matched_rows=metrics["matched_rows"],
            reference_sha256=reference["answer_sha256"],
            accuracy_basis=reference["basis"],
        )
        if row.get("minimum_accuracy") is not None:
            row["observed_constraint_satisfied"] = row["f1"] >= row["minimum_accuracy"]
    return row


def execute_saved_fallback(
    workload,
    root,
    *,
    settings,
    models,
    teacher,
    database,
    history,
    minimum,
    reference,
    stage,
    runtime_adaptation=None,
):
    """Execute the best saved completion; keep paid sampling and Beam evidence."""
    from planning.kernels import _search_core

    previous = read(root / "summary.json")
    if (
        previous["query"] != workload["id"]
        or previous.get("minimum_accuracy") != minimum
    ):
        raise ValueError("Saved fallback inputs differ from this query or target")
    result = read(root / "search.json")
    assessments = result["assessments"]
    if len(result["complete"]) != len(assessments) or not assessments:
        raise ValueError("Saved search has no complete candidates for fallback")
    decision = _search_core.select_completions(
        [(0.0, 0.0)] * len(assessments),
        [a.get("time_prediction") for a in assessments],
        [a.get("accuracy") for a in assessments],
        minimum,
    )
    index = decision["selected_indices"][0]
    selected, assessment = result["complete"][index], assessments[index]
    save(root / "fallback-origin.json", previous)
    row = {
        **previous,
        "status": "planned",
        "accuracy_fallback": decision["constraint_satisfied"] is False,
        "constraint_satisfied": decision["constraint_satisfied"],
        "selection_basis": "highest_predicted_accuracy_fallback",
        "predicted_execution_seconds": assessment.get("time_prediction"),
        "predicted_f1": assessment.get("accuracy"),
        "search_reused": True,
        "retry_beam_seconds": 0.0,
        "reason": "No candidate meets A_min; executing highest predicted accuracy completion",
    }
    save(root / "plan.json", selected)
    destination = root / "fallback-sql"
    if destination.exists():
        shutil.rmtree(destination)
    with DuckDBEvaluator(
        runtime_path(settings, "build"),
        teacher.transport_models(models),
        workload["sql"],
        database=database,
        output=destination,
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
        return execute_selected(
            workload,
            root,
            evaluator,
            read(root / "query_ir.json"),
            selected,
            row,
            reference,
            stage,
            destination / "run-1/telemetry.jsonl",
            runtime_adaptation={
                **runtime_adaptation,
                "profile": read(root / "profile.json"),
            }
            if runtime_adaptation
            else None,
        )


def render(output, state, rows, workloads):
    from experiments.semcache.monitoring.files import publish

    def number(value, digits=2):
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"

    by_key = {(r["case"], r["query"]): r for r in rows}
    count = len(workloads)
    total = count * len(CASES)
    lines = [
        f"# 训练后 {count} 条查询测试",
        "",
        f"状态：{state['status']}；已有记录 {len(rows)}/{total}；待补执行 {sum(r.get('status') == 'infeasible' for r in rows)}；更新：{state.get('updated_at', '')}。",
        f"当前：{state.get('case', '—')} / {state.get('query', '—')} / {state.get('stage', '—')}。",
        "",
        f"第 {state.get('checkpoint_round', 3)} 轮最终模型；"
        + (
            "使用独立测试 SQL，测试结果不加入训练。"
            if state.get("heldout")
            else "使用训练查询，衡量同查询学习效果。"
        ),
        "模式二优先级；Beam 90；E2B、E4B、12B；CPU 16 线程，查询串行。"
        + (
            f"采样和执行的全部模型HTTP请求共享总并发上限{state['global_http_limit']}。"
            if state.get("global_http_limit")
            else "每模型并发16。"
        ),
        "依次测试 "
        + "、".join(f"{case}（下限 {minimum:.0%}）" for case, _, minimum in CASES)
        + "；在达到各组准确率下限的完整计划中选最快计划；不运行 max-quality。",
        "中间层按预测时间和准确率做 Pareto 非支配分层，不按准确率下限裁剪。最终有达标完整计划就选最快的；全部不达标则执行预测准确率最高的计划，并标记准确率回退，不视为达到原始下限。",
        (
            "prompt cache 开启，二次核对关闭；按最新指令，续跑及后续各组保留模型与 prompt 缓存。"
            if state.get("preserve_cache")
            else "prompt cache 开启，二次核对关闭；每组开始清一次模型与 prompt 缓存，组内连续复用。"
        ),
        "F1 使用对应查询已保存的 teacher 答案和非自由文本评分；聚合查询比较聚合前输入。每条执行后立即评分。",
        "调用列只列执行 HTTP 请求，不含采样；采样请求保存在各查询 sampling.json。",
        "appstore 若选中语义过滤先于关系连接的计划，仅保存计划并跳过执行；不计入平均 F1。",
        "",
    ]
    if state.get("runtime_adaptation"):
        lines += [
            f"动态调整已开启：切换上限K={state['runtime_adaptation']['max_switches']}，在预计处理进度i/(K+1)处各检查一次，每次最多提交一次计划更新，可同时改变多个未完成语义算子的实现；未触发的机会不补发。进度由实际完成的语义输入行或候选对及初始计划预计输入基数计算。统计使用当前实现下累计真实数据，包含当前chunk已完成的输入；只使用选择率、缓存复用和计划准确率触发；运行时Beam=12，固定逻辑结构。",
            "运行时不采样、不增加推理请求；使用实际处理行数、输出数、响应耗时、API缓存用量和实际请求前缀。搜索在低优先级后台线程执行，value model和缓存MLP使用GPU；只在chunk边界切换，执行线程不等待搜索。",
            "平均匹配长度按请求来源计算：HTTP请求只取API的cached_tokens，本地prompt cache命中只取缓存条目的完整prompt_tokens；两类请求分别计数后计算加权平均，不对同一个请求叠加两种值。",
            "",
        ]
    for case, _, _ in CASES:
        entries = [r for r in rows if r["case"] == case]
        scores = [r["f1"] for r in entries if r.get("f1") is not None]
        seconds = sum(r.get("execution_seconds") or 0 for r in entries)
        lines.append(
            f"- {case}：已处理 {len(entries)}/{count}；跳过 {sum(r.get('status') == 'skipped_plan' for r in entries)}；已评分 {len(scores)}；平均 F1 {number(sum(scores) / len(scores), 4) if scores else '—'}；执行合计 {seconds:.2f}s。"
        )
    lines += [
        "",
        "| 策略 | 查询 | 状态 | Beam(s) | 预测时间(s) | 预测准确率 | 执行(s) | F1 | E2B调用 | E4B调用 | 12B调用 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    if state.get("runtime_adaptation"):
        lines[-2] = lines[-2].rstrip("|") + "| 运行时检查 | 触发 | 切换 |"
        lines[-1] = lines[-1].rstrip("|") + "|---:|---:|---:|"
    for case, _, _ in CASES:
        for workload in workloads:
            key = workload["id"]
            row = by_key.get((case, key), {})
            active = (
                state.get("case") == case
                and state.get("query") == key
                and (not row or row.get("status") == "infeasible")
            )
            if active:
                row = dict(state.get("live", {}))
            status = row.get("status", "未开始")
            if status == "infeasible":
                status = "待回退执行"
            elif status == "skipped_plan":
                status = "跳过（过滤先于连接）"
            elif status == "completed" and row.get("accuracy_fallback"):
                status = "完成（准确率回退）"
            beam, execution = row.get("beam_seconds"), row.get("execution_seconds")
            if active:
                status = state.get("stage", "准备中")
                elapsed = time.time() - state.get("stage_started_at", time.time())
                if status == "采样与搜索中":
                    beam = elapsed
                if status == "执行中":
                    execution = elapsed
            counts = row.get("execution_http_requests_by_model", {})
            if active and status == "执行中":
                counts = state.get("live_counts", counts)
            cells = [
                case,
                workload["public_query_id"],
                status,
                number(beam),
                number(row.get("predicted_execution_seconds")),
                number(row.get("predicted_f1"), 4),
                number(execution),
                number(row.get("f1"), 4),
                *(str(counts.get(m, "—")) for m in MODELS),
            ]
            if state.get("runtime_adaptation"):
                metrics = row.get("runtime", {})
                cells.extend(
                    str(metrics.get(k, "—")) for k in ("checks", "triggered", "updates")
                )
            lines.append("| " + " | ".join(cells) + " |")
    if state.get("error"):
        lines += ["", "错误：" + state["error"]]
    publish(output / "测试结果.md", "\n".join(lines) + "\n")


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
    resource = (ROOT.parent / "experiments/results/semcache/model_training/.collection.lock").open("a")
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
        from planning.execution.adaptation_monitor import AdaptationSettings

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
        "model_checkpoint_override": str(checkpoint) if model_checkpoint is not None else None,
        "use_cache_features": settings.get("value_model", {}).get("use_cache_features", True),
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
            "appstore": "skip_selected_plan_if_semantic_filter_precedes_relational_join",
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
            render(output, state, rows, workloads)

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
            r["status"] == "skipped_plan"
            or (r["status"] == "completed" and r.get("f1") is not None)
            for r in rows
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


def digest_sql(sql):
    import hashlib

    return hashlib.sha256(sql.encode()).hexdigest()


def main():
    global CASES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=ROOT.parent / "experiments/results/semcache/model_training/feedback-training",
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


if __name__ == "__main__":
    main()
