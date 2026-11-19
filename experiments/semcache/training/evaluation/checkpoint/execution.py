"""Evaluate a saved value-model checkpoint in separate accuracy-target passes."""

import shutil
import time
from collections import Counter

from planning.backends.execution import DuckDBEvaluator
from planning.settings.configuration import (
    runtime_path,
)
from reference_answers.sql.aggregate import capture_plan
from training.artifacts import read, save
from training.references.scoring import candidate_result
from training.reporting.telemetry import MODELS

from experiments.system_comparison.accuracy.scoring.matching import score_non_free_text

CASES = (
    ("min-20", "fastest", 0.2),
    ("min-50", "fastest", 0.5),
    ("min-80", "fastest", 0.8),
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
    row["execution_telemetry"] = str(telemetry)
    save(root / "selection.json", row)
    stage("执行中", **row)
    runtime_arguments = {}
    if runtime_adaptation:
        from planning.backends.execution.evaluator import ExplicitPolicy
        from planning.execution.adaptation.service import RuntimeExecutionPolicy

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


def render(output, state, rows, workloads, *, cases=CASES):
    from experiments.semcache.monitoring.files import publish

    def number(value, digits=2):
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"

    by_key = {(r["case"], r["query"]): r for r in rows}
    count = len(workloads)
    total = count * len(cases)
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
            f"采样和执行的本地部署模型HTTP请求共享总并发上限{state['global_http_limit']}（外网Teacher独立并发上限{state.get('external_teacher_concurrency', 16)}，不占用本地上限）。"
            if state.get("global_http_limit")
            else f"每模型并发16（外网Teacher独立并发上限{state.get('external_teacher_concurrency', 16)}）。"
        ),
        "依次测试 "
        + "、".join(f"{case}（下限 {minimum:.0%}）" for case, _, minimum in cases)
        + "；在达到各组准确率下限的完整计划中选最快计划；不运行 max-quality。",
        "中间层按预测时间和准确率做 Pareto 非支配分层，不按准确率下限裁剪。最终有达标完整计划就选最快的；全部不达标则执行预测准确率最高的计划，并标记准确率回退，不视为达到原始下限。",
        (
            "prompt cache 开启，二次核对关闭；按最新指令，续跑及后续各组保留模型与 prompt 缓存。"
            if state.get("preserve_cache")
            else "prompt cache 开启，二次核对关闭；每组开始清一次模型与 prompt 缓存，组内连续复用。"
        ),
        "F1 使用对应查询已保存的 teacher 答案和非自由文本评分；聚合查询比较聚合前输入。每条执行后立即评分。",
        "调用列只列执行 HTTP 请求，不含采样；采样请求保存在各查询 sampling.json。",
        "",
    ]
    if state.get("runtime_adaptation"):
        lines += [
            f"动态调整已开启：切换上限K={state['runtime_adaptation']['max_switches']}，在预计处理进度i/(K+1)处各检查一次，每次最多提交一次计划更新，可同时改变多个未完成语义算子的实现；未触发的机会不补发。进度由实际完成的语义输入行或候选对及初始计划预计输入基数计算。统计使用当前实现下累计真实数据，包含当前chunk已完成的输入；只使用选择率、缓存复用和计划准确率触发；运行时Beam=12，固定逻辑结构。",
            "运行时不采样、不增加推理请求；使用实际处理行数、输出数、响应耗时、API缓存用量和实际请求前缀。搜索在低优先级后台线程执行，value model和缓存MLP使用GPU；只在chunk边界切换，执行线程不等待搜索。",
            "平均匹配长度按请求来源计算：HTTP请求只取API的cached_tokens，本地prompt cache命中只取缓存条目的完整prompt_tokens；两类请求分别计数后计算加权平均，不对同一个请求叠加两种值。",
            "",
        ]
    for case, _, _ in cases:
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
    for case, _, _ in cases:
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
