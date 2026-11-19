"""Collect training references using SemCache, a teacher and fixed GOTDF."""

import json
import shutil
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from planning.settings.teacher import TeacherModel, default_teacher_model, load_teacher
from reference_answers.sql.validation import validate_result_labels
from reference_answers.storage.files import file_digest
from training.artifacts import save

PROTOCOL = "semcache-calibrated-teacher-gotdf-v3"


def timestamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def configured_teacher(path, settings):
    loaded = load_teacher(default_teacher_model(path), {}, path)
    endpoint = dict(loaded.endpoint)
    endpoint["max_tokens"] = settings["max_output_tokens"]
    return TeacherModel(
        loaded.model_id, endpoint, loaded.supported_kinds, loaded.api_key
    )


def validate_saved_answer(answer, workload, teacher):
    from reference_answers.sql.aggregate import capture_plan

    from experiments.system_comparison.accuracy.reporting.artifacts import result_digest

    expected_basis = (
        "pre_aggregate_rows" if capture_plan(workload["sql"]) else "final_output"
    )
    if (
        answer.get("status") != "completed"
        or answer.get("protocol") != PROTOCOL
        or answer.get("system") != "semcache"
        or answer.get("sql_sha256") != workload["sql_sha256"]
        or answer.get("teacher")
        != (teacher if isinstance(teacher, dict) else teacher.configuration())
        or answer.get("result_sha256") != result_digest(answer)
        or answer.get("accuracy_basis") != expected_basis
        or expected_basis == "final_output"
        and answer.get("pre_aggregate") is not None
    ):
        raise ValueError(f"Saved SemCache answer contract differs: {workload['id']}")
    validate_result_labels(answer, workload["output_contract"], context=workload["id"])
    if answer.get("accuracy_basis") == "pre_aggregate_rows":
        pre = answer.get("pre_aggregate")
        if not isinstance(pre, dict):
            raise ValueError("Aggregate reference lacks its actual input snapshot")
        snapshot = json.loads(Path(pre["path"]).read_text())
        if (
            result_digest(snapshot) != pre["result_sha256"]
            or snapshot["columns"] != pre["columns"]
            or len(snapshot["rows"]) != pre["row_count"]
            or pre.get("protocol") != "final_aggregate_input_v1"
        ):
            raise ValueError("Saved aggregate input differs")


def collect(
    root,
    workloads,
    databases,
    build,
    teacher,
    settings,
    contract,
    *,
    execute,
    teacher_routes=None,
):
    from experiments.system_comparison.accuracy.reporting.artifacts import result_digest

    routes = teacher_routes or {"original": teacher}
    historical_teachers = contract.get("query_teacher_configs", {})
    teacher_configs = contract.get(
        "teachers", {name: model.configuration() for name, model in routes.items()}
    )
    assignments = contract.get(
        "query_teachers", {w["id"]: next(iter(routes)) for w in workloads}
    )
    teacher_label = ", ".join(
        f"{name}: {model.model_id}" for name, model in routes.items()
    )
    report = {
        "status": "running",
        "system": "semcache",
        "protocol": PROTOCOL,
        "planner": "heuristic_requests",
        "layout": "GOTDF",
        "teacher": teacher_label,
        "started_at": timestamp(),
        "total": len(workloads),
        "completed": 0,
        "current_query": None,
        "query_workers": settings.get("query_workers", 1),
        "queries": [
            {
                "id": w["id"],
                "dataset": w["dataset"],
                "status": "pending",
                "teacher_route": assignments[w["id"]],
                "teacher_model": historical_teachers.get(
                    w["id"], teacher_configs[assignments[w["id"]]]
                )["model_id"],
            }
            for w in workloads
        ],
    }
    lock, stopping, cancelled = threading.RLock(), threading.Event(), threading.Event()

    def publish():
        with lock:
            report["active_queries"] = [
                row["id"] for row in report["queries"] if row["status"] == "running"
            ]
            report["current_query"] = (
                report["active_queries"][0]
                if len(report["active_queries"]) == 1
                else None
            )
            report["updated_at"] = timestamp()
            save(root / "progress.json", report)
            lines = [
                "# SemCache Teacher 训练参考答案",
                "",
                f"状态：{report['status']}；完成 {report['completed']}/{report['total']}；更新：{report['updated_at']}",
                f"Teacher：{teacher_label}；每查询请求并发上限：{settings['parallelism']}",
                "思考设置：使用各渠道 Teacher 配置文件中的设置。",
                f"查询并行数：{report['query_workers']}；不设置跨查询的 Teacher 请求总上限。",
                f"HTTP 429：每个请求最多重试 {settings.get('rate_limit_retries', 0)} 次，优先按 Retry-After 等待；不是二次答案核对。",
                "本地答案缓存与请求复用："
                + (
                    "开启；首次保存候选答案，第二次独立核对，不一致时追加调用并投票；"
                    "第三次及以后命中已确认条目时复用，条目被淘汰后重新确认。"
                    if settings.get("cache_enabled", True)
                    and settings.get("verify_on_second_use", True)
                    else "开启；精确复用已保存答案，不进行缓存答案核验或投票。"
                    if settings.get("cache_enabled", True)
                    else "关闭；不进行缓存答案核验或投票。"
                ),
                "计划：模式1 heuristic_requests Beam；layout 固定 GOTDF；固定 Teacher；"
                f"语义算子最多{settings['planning_contract']['sampling']['operator_sample_rows']}条，"
                f"关系算子最多{settings['planning_contract']['sampling']['relational_sample_rows']}个输入单位；"
                "按预计执行时间排序；相关子查询无法完整计时时，按预计语义请求数排序并保留耗时未知。",
                "使用 SemCache 原生执行与 Prompt；保存最终结果、实际聚合输入、QueryIR、PlanSpec 和原生遥测。",
                "SemCache 完整执行按单行／记录对请求，effective_batch_size=1；未套用 Sema 的四项合批协议。",
                "",
                "| 查询 | 数据集 | Teacher 渠道 | 结果行数 | HTTP 请求 | 耗时(s) | 错误 |",
                "|---|---|---|---:|---:|---:|---|",
            ]
            for row in report["queries"]:
                values = [
                    row["id"],
                    row["dataset"],
                    row["teacher_route"],
                    row.get("rows", "-"),
                    row.get("transport", {}).get("http_requests", "-"),
                    f"{row['seconds']:.2f}" if "seconds" in row else "-",
                    row.get("error", "-"),
                ]
                lines.append(
                    "| "
                    + " | ".join(
                        str(v).replace("|", "\\|").replace("\n", " ") for v in values
                    )
                    + " |"
                )
            target = root / "参考答案生成.md"
            temporary = target.with_suffix(".md.tmp")
            temporary.write_text("\n".join(lines) + "\n")
            temporary.replace(target)

    def monitor():
        while not stopping.wait(10):
            publish()

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        publish()

        def run_query(workload, row):
            if cancelled.is_set() or row["status"] in {"completed", "cached"}:
                return
            folder = root / workload["id"]
            selected = settings.get("selected_datasets")
            if (
                selected
                and workload["dataset"] not in selected
                and not (folder / "answer.json").exists()
            ):
                return
            query_teacher = routes.get(row["teacher_route"])
            expected_teacher = historical_teachers.get(
                workload["id"], teacher_configs[row["teacher_route"]]
            )
            with lock:
                row["status"] = "running"
            publish()
            answer_path = folder / "answer.json"
            started = time.monotonic()
            saved = json.loads(answer_path.read_text()) if answer_path.exists() else None
            if saved is not None and saved.get("verify_on_second_use", False) == settings["verify_on_second_use"]:
                answer = saved
                validate_saved_answer(answer, workload, expected_teacher)
                status = "cached"
            else:
                if (
                    query_teacher is None
                    or expected_teacher != query_teacher.configuration()
                ):
                    raise ValueError(
                        f"Historical teacher answer is missing: {workload['id']}"
                    )
                execution = folder / "execution"
                if execution.exists():
                    shutil.rmtree(execution)

                def progress(stats):
                    with lock:
                        row["transport"] = dict(stats)
                        row["seconds"] = time.monotonic() - started

                answer = execute(
                    workload,
                    databases[workload["dataset"]],
                    build,
                    query_teacher,
                    execution,
                    settings,
                    progress,
                )
                answer.update(
                    status="completed",
                    protocol=PROTOCOL,
                    system="semcache",
                    teacher=query_teacher.configuration(),
                    teacher_route=row["teacher_route"],
                    sql_sha256=workload["sql_sha256"],
                    ground_truth=False,
                    verify_on_second_use=settings["verify_on_second_use"],
                )
                answer["result_sha256"] = result_digest(answer)
                validate_saved_answer(answer, workload, query_teacher)
                save(answer_path, answer)
                status = "completed"
            for stale in ("error.txt", "root-cause.json"):
                (folder / stale).unlink(missing_ok=True)
            with lock:
                row.update(
                    status=status,
                    rows=len(answer["rows"]),
                    transport=answer["transport"],
                    seconds=answer["seconds"],
                    answer=str(answer_path),
                )
                report["completed"] += 1
            publish()
            print(
                f"{report['completed']}/{report['total']} {workload['id']}: {row['rows']} rows",
                flush=True,
            )

        def guarded_query(workload, row):
            try:
                run_query(workload, row)
            except Exception as error:
                from reference_answers.execution.errors import query_failure

                folder = root / workload["id"]
                failure = query_failure(folder, error)
                with lock:
                    row.update(status="failed", error=failure["reason"])
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / "error.txt").write_text(traceback.format_exc())
                    save(folder / "root-cause.json", failure)
                publish()
                if failure["reason"] != str(error):
                    raise RuntimeError(failure["reason"]) from error
                raise

        errors = []
        # Validate saved answers before issuing requests, including inactive channels.
        for workload, row in zip(workloads, report["queries"], strict=True):
            if (root / workload["id"] / "answer.json").exists():
                guarded_query(workload, row)

        def run_lane(name):
            failures = []
            for workload, row in zip(workloads, report["queries"], strict=True):
                if row["teacher_route"] == name:
                    try:
                        guarded_query(workload, row)
                    except Exception as error:
                        failures.append(error)
            if failures:
                raise failures[0]

        with ThreadPoolExecutor(
            max_workers=report["query_workers"], thread_name_prefix="reference-query"
        ) as pool:
            futures = (
                [pool.submit(run_lane, name) for name in routes]
                if len(routes) > 1
                else [
                    pool.submit(guarded_query, workload, row)
                    for workload, row in zip(workloads, report["queries"], strict=True)
                ]
            )
            try:
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as error:
                        errors.append(error)
            except BaseException:
                cancelled.set()
                for future in futures:
                    future.cancel()
                raise
        if errors:
            raise errors[0]
        for dataset, path in databases.items():
            if file_digest(path) != contract["database_sha256"][dataset]:
                raise ValueError(f"Source database changed: {dataset}")
        report.update(
            status="completed"
            if report["completed"] == report["total"]
            else "waiting_queries",
            current_query=None,
        )
    except BaseException as error:
        with lock:
            report["status"] = (
                "stopped" if isinstance(error, KeyboardInterrupt) else "failed"
            )
            report["error"] = str(error) or type(error).__name__
            for row in report["queries"]:
                if row["status"] == "running":
                    row.update(status=report["status"], error=report["error"])
        raise
    finally:
        stopping.set()
        thread.join()
        publish()
