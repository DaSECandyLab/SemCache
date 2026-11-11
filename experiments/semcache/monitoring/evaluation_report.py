"""Heldout inference uses the same compact table and incremental telemetry as training."""

import time
from collections import Counter
from pathlib import Path

from training.query_manifest import ordered_heldout_manifest

from .collection_report import DEFAULT_MODEL_IDS, execution, query_headers
from .files import publish
from .progress import read
from .tables import table


def duration(value):
    return f"{value:.2f}" if isinstance(value, (int, float)) else "—"


def render_references(spec, state):
    if not spec.get("reference_output") or spec.get("reference_reuse_only"):
        return
    output = Path(spec["reference_output"])
    rows, complete = [], 0
    for index, item in enumerate(ordered_heldout_manifest(spec["manifest"]), 1):
        key = item["id"]
        folder = output / key
        record = read(folder / "reference.json") or {}
        result = record.get("result", {})
        status = result.get("status")
        complete += status == "completed"
        label = {"completed": "完成", "timeout": "超时", "failed": "失败"}.get(
            status, "未开始"
        )
        if not status and any(folder.glob("attempt-*")):
            label = "未完成"
        if (
            state.get("status") == "running"
            and state.get("current_query") == key
            and state.get("current_stage") == "参考执行"
        ):
            label = "生成中"
        seconds = (
            result.get("seconds")
            if status != "timeout"
            else result.get("time_lower_bound")
        )
        rows.append(
            [
                index,
                key.removeprefix("translated/semcache/"),
                label,
                ("≥" if status == "timeout" else "") + duration(seconds),
                f"[答案与执行数据]({key}/)" if folder.exists() else "—",
            ]
        )
    teacher = Path(
        spec.get("reference_teacher_config") or spec["settings"]["teacher_config"]
    ).name
    lines = [
        "# 公共查询参考答案生成",
        "",
        f"- 完成：{complete}/{len(rows)}；教师配置：`{teacher}`。",
        "- 每条查询只生成一份参考答案，fastest 与 accurate 共用；参考生成不使用测试候选或价值模型。",
        "- 完整原始数据；启发式结构、固定 GOTDF；末尾聚合保留聚合前输入。",
        "- 本轮参考生成时限："
        + (
            f"{spec['reference_timeout_seconds']:g} 秒"
            if spec.get("reference_timeout_seconds") is not None
            else "不设查询总时限，持续执行直至完成"
        )
        + "；每条答案的 SQL、数据库与教师契约保存在 reference.json。",
        "",
    ]
    table(lines, ["序号", "查询", "进度", "执行(s)", "数据"], rows, right=(0, 3))
    publish(output / "参考答案生成.md", "\n".join(lines) + "\n")


def render_planning(output, spec, state):
    """Report Beam artifacts and predictions without implying query execution."""
    output = Path(output)
    completed = {}
    for preference in spec["preferences"]:
        for row in (read(output / preference / "results.json") or {}).get(
            "queries", []
        ):
            completed[preference, row["query"]] = row
    rows = []
    finished = 0
    now = time.time()
    for index, preference in enumerate(spec["preferences"], 1):
        for item in ordered_heldout_manifest(spec["manifest"]):
            key = item["id"]
            active = (
                state.get("preference") == preference
                and state.get("current_query") == key
            )
            attempts = list((output / preference / key).glob("attempt-*"))
            folder = (
                max(attempts, key=lambda p: int(p.name.removeprefix("attempt-")))
                if attempts
                else None
            )
            entry = dict(completed.get((preference, key), {}))
            search = {}
            if folder:
                entry = {**(read(folder / "progress.json") or {}), **entry}
                entry.update(read(folder / "summary.json") or {})
                search = read(folder / "search.json") or {}
            status = entry.get("status")
            terminal = status in (
                "planned",
                "infeasible",
                "completed",
                "failed",
                "interrupted",
            )
            finished += terminal
            infeasible = (
                status == "infeasible"
                or entry.get("constraint_satisfied", search.get("constraint_satisfied"))
                is False
            )
            stage = (
                "本次搜索无达标计划"
                if infeasible
                else "计划已生成"
                if status in ("planned", "completed")
                else {"failed": "规划失败", "interrupted": "已中断"}.get(
                    status or "", entry.get("stage", "准备中" if active else "未开始")
                )
            )
            if active and not terminal:
                if state.get("status") in ("error", "stopped"):
                    stage = "规划失败" if state["status"] == "error" else "已停止"
                elif state.get("current_stage"):
                    stage = state["current_stage"]
            beam = entry.get("planning_seconds")
            if (
                beam is None
                and active
                and not terminal
                and entry.get("stage") == "搜索中"
                and entry.get("stage_started_at") is not None
            ):
                beam = now - entry["stage_started_at"]
            candidates = entry.get("candidates")
            if candidates is None and isinstance(search.get("complete"), list):
                candidates = len(search["complete"])
            search_status = (
                "本次搜索无达标计划"
                if infeasible
                else "已完成"
                if status in ("planned", "completed")
                else "已结束"
                if search
                else "搜索中"
                if active and stage == "搜索中"
                else "—"
            )
            if search.get("budget_exhausted"):
                search_status += "（达到搜索预算）"
            if search.get("action_budget_exhausted"):
                search_status += "（达到扩展预算）"
            plan = folder / "plan_spec.json" if folder else None
            if not plan or not plan.is_file():
                plan = (
                    Path(entry["plan_artifact"]) if entry.get("plan_artifact") else None
                )
            plan_link = "—"
            if plan and plan.is_file() and not infeasible:
                plan_link = f"[计划]({plan.relative_to(output) if plan.is_relative_to(output) else plan})"
            predicted_f1 = entry.get("predicted_f1")
            reason = (
                entry.get("infeasible_reason")
                or entry.get("search_reason")
                or entry.get("reason")
                or search.get("reason")
                or "—"
            )
            rows.append(
                [
                    key.removeprefix("translated/semcache/"),
                    preference,
                    index,
                    stage,
                    search_status,
                    duration(beam),
                    entry.get("expansions", search.get("expansions", "—")),
                    candidates if candidates is not None else "—",
                    entry["estimated_requests"]
                    if entry.get("estimated_requests") is not None
                    else "—",
                    duration(entry.get("predicted_execution_seconds")),
                    f"{predicted_f1:.4f}" if predicted_f1 is not None else "—",
                    "未执行",
                    plan_link,
                    reason,
                    f"[目录]({folder.relative_to(output)}/)" if folder else "—",
                ]
            )
    models = list(
        spec.get("candidate_models")
        or read(Path(spec["training_run"]) / "models.json")
        or DEFAULT_MODEL_IDS
    )
    lines = [
        f"# {len(spec['manifest'])} 条公共查询计划生成",
        "",
        f"- 状态：`{state['status']}`；规划结束 {finished}/{len(rows)}；更新时间：{state.get('updated_at', '—')}",
        "- 本轮只运行 Beam 搜索并保存选中计划，不执行完整查询、不生成完整参考答案、不计算实测 F1。",
        "- 请求量模式只预测语义处理记录／记录对总量，不预测执行秒数或F1；value模式预测时间和F1。两者都可进行有界采样，未知值显示 —。",
        f"- 训练目录：`{spec['training_run']}`",
        "- 候选模型：" + "、".join(f"`{model}`" for model in models) + "。",
        f"- 搜索配置：Beam 宽度 {spec.get('settings', {}).get('search', {}).get('beam_width', '—')}；时间单位为秒。",
        (
            "- 价值模型：请求量模式不加载 checkpoint。"
            if spec.get("plan_generation", "heuristic_requests") == "heuristic_requests"
            else f"- 价值模型 checkpoint：`{spec['checkpoint']}`"
        ),
        f"- 当前：`{state.get('current_query') or '—'}`；偏好：`{state.get('preference') or '—'}`；阶段：{state.get('current_stage') or '—'}。",
        "- 查询按 estate → movies → movies_sema → appstore → tpch 顺序生成计划，各数据集内按 Q 编号递增；结果不回流训练。",
    ]
    if spec.get("minimum_accuracy") is not None:
        lines.append(
            f"- 最低预测 F1：{spec['minimum_accuracy']:.1%}；只保存本次搜索中预测达标的选中计划。"
            "本次搜索无达标计划表示当前候选与搜索预算下未找到可行计划，不代表所有计划均不可行。"
        )
    if spec.get("fixed_layout"):
        lines.append(
            f"- 固定 layout：{spec['fixed_layout']}，各输入字段保持原始绑定顺序。"
            + (
                f"固定模型：`{models[0]}`；Beam 按预计处理记录／记录对总量保留合法结构。"
                if len(models) == 1
                else "模型仍从上述候选中选择。"
            )
        )
    if spec.get("after_evaluation"):
        lines.append(
            f"- 启动依赖：`{spec['after_evaluation']}`；前序状态：`{state.get('previous_status', '—')}`。"
        )
    if state.get("training_phase"):
        lines.append(
            f"- 等待训练：{state['training_phase']}，第 {state.get('training_round', '—')} 轮。"
        )
    if state.get("error"):
        lines.append(f"- 错误：{str(state['error']).replace(chr(10), ' ')}")
    lines += ["", "## 查询计划总表", ""]
    table(
        lines,
        [
            "查询",
            "偏好",
            "轮",
            "进度",
            "搜索状态",
            "Beam(s)",
            "扩展",
            "候选",
            "预计处理记录／对",
            "预测执行(s)",
            "预测 F1",
            "完整执行",
            "计划",
            "原因",
            "数据",
        ],
        rows,
        right=(2, 5, 6, 7, 8, 9, 10),
    )
    return publish(output / "测试进度.md", "\n".join(lines) + "\n")


def render(output, spec, state):
    if spec.get("planning_only"):
        return render_planning(output, spec, state)
    output = Path(output)
    render_references(spec, state)
    models = list(
        spec.get("candidate_models")
        or read(Path(spec["training_run"]) / "models.json")
        or DEFAULT_MODEL_IDS
    )
    preferences = spec["preferences"]
    rounds = "、".join(
        f"第 {index} 轮 {preference}" for index, preference in enumerate(preferences, 1)
    )
    reference_usage = "供本轮使用" if len(preferences) == 1 else "供各轮共用"
    completed = {}
    for preference in spec["preferences"]:
        for row in (read(output / preference / "results.json") or {}).get(
            "queries", []
        ):
            completed[preference, row["query"]] = row
    total = len(spec["manifest"]) * len(spec["preferences"])
    lines = [
        f"# {len(spec['manifest'])} 条公共查询测试",
        "",
        f"- 状态：`{state['status']}`；完成 {len(completed)}/{total}；更新时间：{state.get('updated_at', '—')}",
        f"- 训练目录：`{spec['training_run']}`",
        "- 候选模型：" + "、".join(f"`{model}`" for model in models) + "。",
        "- 搜索配置：Beam 宽度 "
        + str(spec.get("settings", {}).get("search", {}).get("beam_width", "—"))
        + "。",
        (
            "- 价值模型：请求量模式不加载 checkpoint。"
            if spec.get("plan_generation", "heuristic_requests") == "heuristic_requests"
            else f"- 价值模型 checkpoint：`{spec['checkpoint']}`"
        ),
        f"- 当前：`{state.get('current_query') or '—'}`；偏好：`{state.get('preference') or '—'}`",
        "- 顺序与其他系统一致：estate → movies → movies_sema → appstore → tpch，各数据集内按 Q 编号递增。",
        f"- {rounds}，每轮 {len(spec['manifest'])} 条；完整原始数据库，保留 SQL 自身筛选；每条选中计划执行一次，候选执行限时 {spec.get('timeout', 600):g} 秒，参考答案不设查询总时限。已完成历史结果保留当时的上限。",
        (
            "- 整轮开始前清空模型与 prompt 结果缓存一次；所有查询的搜索与执行共享连续缓存，查询之间不清空；结果不回流训练。"
            if spec.get("cache_reset_scope") == "evaluation"
            else "- 每查询开始前清缓存；搜索与正式执行保持同一缓存上下文；结果不回流训练。"
        ),
        f"- Beam 判定教师：`{spec.get('settings', {}).get('teacher_config', '配置中的 teacher')}`；参考与评分教师：`{Path(spec['reference_teacher_config']).name if spec.get('reference_teacher_config') else spec.get('settings', {}).get('teacher_config', '配置中的 teacher')}`。参考由启发式确定结构、固定 GOTDF，生成一次{reference_usage}；末尾聚合使用聚合前输出，未知 F1 显示 —。",
        "- 与真实训练收集表使用相同列。时间单位为秒；总计包含参考生成，评分仅为 F1 比较。各候选模型次数包含 Beam 和正式执行的 HTTP 发送，不含缓存命中和 teacher 参考。",
    ]
    if spec.get("execution_only"):
        lines.append(
            "- 本轮只测查询执行时间与请求数，不运行完整参考或准确率评分；准确率列留空。"
        )
    elif spec.get("scoring_policy") == "after_each_completed_query":
        lines.append(
            "- 每条查询执行成功后立即计算并刷新 F1，再执行下一条；超时或尚未成功评分时显示 —。"
        )
    if spec.get("reference_output"):
        lines.append(
            f"- 独立参考答案：[参考答案生成进度]({spec['reference_output']}/参考答案生成.md)，{reference_usage}。"
        )
    if spec.get("plan_generation") == "heuristic_value":
        lines.append(
            "- 计划策略：heuristic_value；关系 FILTER优先，其次语义 FILTER／语义 JOIN／关系 JOIN同组，然后关系 PROJECTION、语义 PROJECTION；value model选择合法结构、模型和layout，保留多条Beam路径。"
        )
    if spec.get("plan_generation", "heuristic_requests") == "heuristic_requests":
        lines.append(
            "- 计划策略：heuristic_requests；关系 FILTER、关系 JOIN、语义 FILTER、语义 JOIN、关系 PROJECTION、语义 PROJECTION依次分组；仅扩最高合法组，固定单模型和GOTDF，按预计处理记录／记录对总量保留多条Beam路径。"
        )
    if spec.get("reference_reuse_only"):
        lines.append(
            "- 仅复用现有完整参考答案；缺失参考时保留执行结果、F1 记为未知，不重新生成参考。"
        )
    if spec.get("frozen_reference_teacher"):
        historical = spec["frozen_reference_teacher"]["endpoint"]
        lines.append(
            f"- 历史参考答案生成设置：thinking={historical.get('thinking')}、"
            f"reasoning_effort={historical.get('reasoning_effort')}；"
            "这些设置只用于核对旧答案契约，本次 Beam 和文本评分均使用当前 teacher 配置。"
        )
    if spec.get("minimum_accuracy") is not None:
        lines.append(
            f"- 最低预测 F1：{spec['minimum_accuracy']:.1%}；仅在预测达标候选中按当前偏好选计划。"
            "没有预测达标候选时记录原因并跳过执行；表中 F1 是与已有参考答案比较的实测值。"
        )
    if spec.get("after_evaluation"):
        lines.append(
            f"- 启动依赖：`{spec['after_evaluation']}` 完成全部查询并释放任务锁；前序状态：`{state.get('previous_status', '—')}`。"
        )
    if state.get("training_phase"):
        lines.append(
            f"- 等待训练：{state['training_phase']}，第 {state.get('training_round', '—')} 轮。"
        )
    if state.get("error"):
        lines.append(f"- 错误：{str(state['error']).replace(chr(10), ' ')}")
    lines += ["", "## 查询总表", ""]
    rows = []
    now = time.time()
    for index, preference in enumerate(spec["preferences"], 1):
        for item in ordered_heldout_manifest(spec["manifest"]):
            key = item["id"]
            active = (
                state.get("preference") == preference
                and state.get("current_query") == key
            )
            attempts = list((output / preference / key).glob("attempt-*"))
            folder = (
                max(attempts, key=lambda p: int(p.name.removeprefix("attempt-")))
                if attempts
                else None
            )
            entry = dict(completed.get((preference, key), {}))
            if folder:
                entry = {**(read(folder / "progress.json") or {}), **entry}
                entry.update(read(folder / "summary.json") or {})
            identity = [
                key.removeprefix("translated/semcache/"),
                preference,
                index,
                "正序",
            ]
            if not entry and not active:
                rows.append(
                    [
                        *identity,
                        "未开始",
                        *["—"] * (len(query_headers(models)) - len(identity) - 1),
                    ]
                )
                continue
            sampling = read(folder / "sampling.json") if folder else None
            sample_models = (sampling or {}).get(
                "http_requests_by_model", entry.get("sampling_http_requests_by_model")
            )
            counts = Counter()
            live_seconds, live_status = None, None
            if folder and (folder / "sql/run-2").exists():
                live_status, counters, live_seconds = execution(folder / "sql/run-2")
                counts.update(counters["models"])
            final = bool(folder and (folder / "summary.json").exists())
            stage = (
                "完成"
                if final and entry.get("status") in ("completed", "timeout")
                else "失败"
                if final and entry.get("status") == "failed"
                else entry.get("stage", "准备中" if active else "未开始")
            )
            if active and state["status"] in ("error", "stopped"):
                stage = "失败" if state["status"] == "error" else "已停止"
            elif active and state.get("current_stage"):
                stage = state["current_stage"]
            elif entry.get("status") == "completed" and entry.get(
                "reference_status"
            ) in ("pending", "interrupted"):
                stage = "待参考"
            execution_state = {
                "completed": "成功",
                "timeout": "超时",
                "failed": "失败",
                "interrupted": "已中断",
            }.get(entry.get("status") or "", "执行中" if live_status else "未执行")
            if (
                not active
                and preference == state.get("preference")
                and key in state.get("pending_execution_queries", [])
                and entry.get("status") == "failed"
            ):
                stage = "待重跑"
                execution_state = "上次失败"
            seconds = entry.get("execution_seconds", live_seconds)
            if entry.get("status") == "timeout":
                seconds = entry.get("time_lower_bound", seconds)
            if entry.get("reference_status") == "failed":
                execution_state = "成功/参考失败"
            beam = entry.get("planning_seconds")
            if beam is None and active and entry.get("stage") == "搜索中":
                beam = now - entry["stage_started_at"]
            elapsed = entry.get("process_seconds")
            if elapsed is None and active and entry.get("started_at"):
                elapsed = now - entry["started_at"]
            accuracy = entry.get("accuracy")
            rows.append(
                [
                    *identity,
                    stage,
                    duration(beam),
                    execution_state,
                    ("≥" if entry.get("status") == "timeout" else "")
                    + duration(seconds),
                    duration(entry.get("execution_timeout_seconds", spec["timeout"])),
                    duration(entry.get("scoring_seconds")),
                    duration(elapsed),
                    f"{accuracy:.4f}" if accuracy is not None else "—",
                    entry.get("expansions", "—"),
                    entry.get("candidates", "—"),
                    f"[目录]({folder.relative_to(output)}/)" if folder else "—",
                    *[
                        (sample_models.get(model, 0) + counts[model])
                        if sample_models is not None
                        else "—"
                        for model in models
                    ],
                ]
            )
    headers = query_headers(models)
    table(lines, headers, rows, right=(2, 5, *range(7, 14), *range(15, len(headers))))
    return publish(output / "测试进度.md", "\n".join(lines) + "\n")
