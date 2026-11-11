"""Train a value model from repeated query execution feedback, without pretraining."""

import argparse
import copy
import fcntl
import json
import math
import os
import shutil
import signal
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from planning.execution.timeouts import TimeoutPolicy
from planning.learning.model import PlanValueModel
from planning.learning.training.data import dataset
from planning.learning.training.targets import TARGET_PROTOCOL
from planning.observations.cache import CacheHistory
from planning.prompt_layouts import BLOCK_ORDERS
from planning.settings.configuration import (
    ROOT,
    activate_native,
    load_config,
    runtime_path,
)
from planning.settings.model_config import discover_models
from planning.settings.parallelism import configure_cpu_resources
from planning.settings.teacher import default_teacher_model, load_teacher
from reference_answers.storage.files import file_digest

from training.artifacts import read, save
from training.calibration.calibrate import calibrate_models
from training.calibration.global_profile import GlobalCalibration
from training.collection.collector import collect_native, reset_context
from training.collection.feedback import collect_feedback_query, restore_completed_query
from training.collection.policy import DUPLICATE_PLAN_POLICY, ordered_workloads
from training.collection.recollection import start_recollection
from training.collection.rewards import execution_reward
from training.fitting import fit
from training.query_manifest import heldout_manifest, ordered_heldout_manifest
from training.references.answers import TrainingAnswers
from training.references.scoring import NON_FREE_TEXT_METRIC
from training.reporting.telemetry import live_execution_row, live_execution_text
from training.reporting.training import (
    enrich_rows,
    read_experience,
    render_table,
    saved_executions,
    with_global_calibration,
    write_training_report,
)
from training.workload_resume import (
    accept_workload_update,
    matching_replay,
    round_query_counts,
)

ACCURACY_MINIMUMS = (0.2, 0.5, 0.8)
SEARCHES_PER_QUERY = len(ACCURACY_MINIMUMS)


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def initial_checkpoint(path, settings):
    if path.exists():
        restored = PlanValueModel.load(path)
        if (restored.training_info or {}).get("initialization") != "random":
            raise ValueError("Feedback training requires random initial weights")
        return
    torch.manual_seed(settings["training"]["seed"])
    model = PlanValueModel(
        **settings["value_model"], time_scale=settings["learning"]["time_scale"]
    )
    model.training_info = {
        "initialization": "random",
        "epochs": 0,
        "pretraining": False,
        "target_protocol": TARGET_PROTOCOL,
        "purpose": "feedback training from random initial weights",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def run(
    output,
    references,
    *,
    workloads_path=None,
    baseline_protocol=None,
    global_http_limit=None,
    execution_timeout=None,
    resume_round=None,
    rounds=3,
    exclude_training_rounds=(),
    preserve_resume_cache=False,
    learning_rate=None,
    collect_only=False,
    collection_checkpoint=None,
    recollect_queries=(),
):
    output, references = Path(output).resolve(), Path(references).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if collect_only != (collection_checkpoint is not None):
        raise ValueError("Collection-only runs require an explicit existing checkpoint")
    recollect_queries = set(recollect_queries)
    if recollect_queries and (
        not collect_only or not (output / "protocol.json").is_file()
    ):
        raise ValueError(
            "Selective recollection requires an existing collection-only run"
        )
    if collection_checkpoint is not None:
        collection_checkpoint = Path(collection_checkpoint).resolve()
        if not collection_checkpoint.is_file():
            raise FileNotFoundError(collection_checkpoint)
    run_guard = (output / "pipeline.lock").open("a")
    fcntl.flock(run_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    resource_guard = (
        ROOT.parent / "experiments/results/semcache/model_training/.collection.lock"
    ).open("a")
    fcntl.flock(resource_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    baseline_digest = None
    if baseline_protocol:
        if Path(baseline_protocol).is_file():
            baseline = read(baseline_protocol)
            baseline_digest = file_digest(baseline_protocol)
        else:
            # The run freezes its full configuration; deleted baseline results
            # must not make an otherwise self-contained continuation unusable.
            baseline = read(output / "protocol.json")
            baseline_digest = baseline["baseline_protocol_sha256"]
        settings = copy.deepcopy(baseline["configuration"])
    else:
        settings = copy.deepcopy(load_config())
    if learning_rate is not None:
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive")
        settings["learning"]["learning_rate"] = learning_rate
    if type(rounds) is not int or rounds < 1:
        raise ValueError("Training rounds must be a positive integer")
    if preserve_resume_cache and resume_round is None:
        raise ValueError("Keeping serving state requires an explicit resume round")
    excluded_rounds = set(exclude_training_rounds)
    if any(
        type(number) is not int or not 1 <= number < rounds
        for number in excluded_rounds
    ):
        raise ValueError("Excluded training rounds must precede the last round")
    for number in excluded_rounds:
        if not (output / f"measured/round-{number}/model/value.pt").is_file():
            raise ValueError(
                "Only previously fitted rounds can be excluded from replay"
            )
    if execution_timeout is not None:
        if not math.isfinite(execution_timeout) or execution_timeout <= 0:
            raise ValueError("Execution timeout must be positive")
        # Apply the execution budget to every rollout without shortening it from earlier timings.
        settings["training"]["timeout"] = execution_timeout
        settings["training"]["minimum_timeout"] = execution_timeout
    if global_http_limit is not None:
        if not 1 <= global_http_limit <= 1024:
            raise ValueError("Global HTTP concurrency must be between 1 and 1024")
        os.environ["SEMCACHE_GLOBAL_HTTP_CONCURRENCY"] = str(global_http_limit)
        os.environ["SEMCACHE_GLOBAL_HTTP_AUDIT"] = str(
            output / "http-concurrency.jsonl"
        )
    settings["teacher_config"] = "teacher.opencode.json"
    settings["training"].update(
        pretrain_rounds=0,
        rounds=rounds,
        dataset=None,
        allow_random_initialization=True,
        feedback_rollouts_per_query=SEARCHES_PER_QUERY,
        feedback_accuracy_minimums=list(ACCURACY_MINIMUMS),
        teacher_config="teacher.opencode.json",
        feedback_cycle=[
            {
                "plan_generation": "heuristic_value",
                "operator_priority_mode": "heuristic_requests",
                "query_order": "shuffled",
            },
        ]
        + [
            {
                "plan_generation": "heuristic_value",
                "operator_priority_mode": "heuristic_value",
                "query_order": "shuffled",
            }
            for _ in range(rounds - 1)
        ],
    )
    if excluded_rounds:
        settings["training"]["replay_exclude_rounds"] = sorted(excluded_rounds)
    else:
        settings["training"].pop("replay_exclude_rounds", None)
    # Training never consumes or publishes prompt answers across requests.
    settings["response_cache"]["directory"] = None
    settings["response_cache"]["verify_on_second_use"] = False
    configure_cpu_resources(settings, configure_torch=True)
    activate_native(settings)
    teacher_config = ROOT / "config/teacher/teacher.opencode.json"
    teacher = load_teacher(default_teacher_model(teacher_config), {}, teacher_config)
    if (
        teacher.endpoint.get("thinking") != "disabled"
        or teacher.endpoint.get("reasoning_effort") != "none"
    ):
        raise ValueError("Beam training teacher must have thinking disabled")
    models, catalog = discover_models(
        settings["models"]["endpoints"],
        max_output_tokens=settings["models"]["max_output_tokens"],
    )
    if set(models) != {"gemma4-e2b", "gemma4-e4b", "gemma4-12b"}:
        raise ValueError("Feedback training requires exactly E2B, E4B and 12B")
    custom_workloads = read(workloads_path)["workloads"] if workloads_path else None
    manifest = (
        ordered_heldout_manifest(heldout_manifest()) if custom_workloads is None else []
    )
    workloads = []
    for item in manifest:
        sql = Path(item["path"]).read_text()
        short = item["id"].removeprefix("translated/semcache/")
        domain = short.split("/")[0]
        workloads.append(
            {
                **item,
                "sql": sql,
                "split": "train",
                "group": item["id"],
                "dataset": {"movies": "movies_nirvana"}.get(domain, domain),
                "public_query_id": short,
                "rows": None,
                "output_contract": {},
                "accuracy_metric": NON_FREE_TEXT_METRIC,
            }
        )
    if custom_workloads is not None:
        if not custom_workloads or any(
            w.get("split") != "train" for w in custom_workloads
        ):
            raise ValueError("Training manifest must contain training queries")
        workloads = [
            dict(w, public_query_id=w["id"], accuracy_metric=NON_FREE_TEXT_METRIC)
            for w in custom_workloads
        ]
        databases = runtime_path(settings, "source_databases")
    else:
        databases = {
            name: ROOT.parent / "benchmarks/databases/semcache" / f"{name}.duckdb"
            for name in sorted({w["dataset"] for w in workloads})
        }
    query_count = len(workloads)
    if recollect_queries - {w["id"] for w in workloads}:
        raise ValueError("Recollection query is absent from the frozen workload")
    if len({workload["id"] for workload in workloads}) != query_count:
        raise ValueError("Training workload IDs must be unique")
    if custom_workloads is not None:
        settings["training"]["queries_per_dataset"] = dict(
            sorted(Counter(workload["dataset"] for workload in workloads).items())
        )
    training_scope = (
        "training manifest; public test queries excluded"
        if custom_workloads is not None
        else "in-sample feedback; public queries intentionally used for training"
    )
    # A custom workload manifest may intentionally select a subset of the
    # frozen reference set (for example, recollecting one dataset). Keep the
    # strict full-set check for ordinary runs, while allowing the selected
    # workload IDs to be validated against the same frozen answers.
    answers = TrainingAnswers(
        references, workloads, databases, allow_subset=custom_workloads is not None
    )
    previous_protocol = read(output / "protocol.json") if recollect_queries else None
    reference_evidence = {
        w["id"]: (
            previous_protocol["answer_sha256"][w["id"]]
            if recollect_queries and w["id"] not in recollect_queries
            else answers.load(w)["answer_sha256"]
        )
        for w in workloads
    }
    frozen = {
        "experiment": "training-manifest"
        if custom_workloads is not None
        else "in-sample-feedback",
        "generalization_evaluation": False,
        "configuration": settings,
        "initialization": "random",
        "pretraining": False,
        "searches_per_query": SEARCHES_PER_QUERY,
        "minimum_accuracies": ACCURACY_MINIMUMS,
        "query_sampling_policy": "once_per_query_round_v1",
        "duplicate_plan_policy": DUPLICATE_PLAN_POLICY,
        "models": models,
        "teacher": teacher.configuration(),
        "teacher_config_sha256": file_digest(teacher_config),
        "target_protocol": TARGET_PROTOCOL,
        "layouts": BLOCK_ORDERS,
        "reference_root": str(references),
        "reference_contract_sha256": file_digest(references / "contract.json"),
        "answer_sha256": reference_evidence,
        "database_sha256": {
            key: file_digest(value) for key, value in databases.items()
        },
        "workload_sql_sha256": {w["id"]: w["sql_sha256"] for w in workloads},
        "reward": "(-observed_execution_seconds, observed_non_free_text_F1)",
        "cache_policy": "prompt answer cache disabled; reset model KV/history at round boundaries; continuous within each round",
        "round_query_counts": round_query_counts(output, query_count, rounds),
    }
    if collect_only:
        frozen.update(
            collect_only=True,
            collection_checkpoint=str(collection_checkpoint),
            collection_checkpoint_sha256=file_digest(collection_checkpoint),
        )
    if baseline_protocol:
        frozen["baseline_protocol_sha256"] = baseline_digest
    if workloads_path:
        frozen["training_manifest_sha256"] = file_digest(workloads_path)
    if global_http_limit is not None:
        frozen["global_http_limit"] = global_http_limit
    if execution_timeout is not None:
        frozen["execution_timeout_seconds"] = execution_timeout
    protocol_path = output / "protocol.json"
    # JSON normalization makes tuple-based layout constants compare consistently.
    frozen = json.loads(json.dumps(frozen))
    if protocol_path.exists() and read(protocol_path) != frozen:
        previous = read(protocol_path)
        upgraded = {
            **previous,
            "query_sampling_policy": "once_per_query_round_v1",
            "round_query_counts": frozen["round_query_counts"],
        }
        if recollect_queries:
            # New teacher answers apply only to replaced executions. Retained
            # records keep their original, explicit answer provenance.
            upgraded.update(
                answer_sha256=frozen["answer_sha256"],
                reference_contract_sha256=frozen["reference_contract_sha256"],
            )
        if collect_only:
            upgraded.update(
                collect_only=True,
                collection_checkpoint=str(collection_checkpoint),
                collection_checkpoint_sha256=file_digest(collection_checkpoint),
            )
            if upgraded == frozen and previous.get("collect_only") is not True:
                save(
                    output / "collection-only-policy.json",
                    {
                        "time": now(),
                        "collect_only": True,
                        "checkpoint": str(collection_checkpoint),
                        "checkpoint_sha256": file_digest(collection_checkpoint),
                        "previous_fitting_artifacts_used": False,
                        "existing_execution_records_preserved": True,
                    },
                )
        if learning_rate is not None and resume_round is not None:
            previous_rate = previous["configuration"]["learning"]["learning_rate"]
            if learning_rate < previous_rate:
                upgraded = copy.deepcopy(upgraded)
                upgraded["configuration"]["learning"]["learning_rate"] = learning_rate
                if upgraded == frozen:
                    save(
                        output / "learning-rate-update.json",
                        {
                            "time": now(),
                            "round": resume_round,
                            "previous_rate": previous_rate,
                            "learning_rate": learning_rate,
                            "weights": "best monitored weights in saved epoch checkpoint",
                            "optimizer_reset": True,
                            "warmup_restarted": False,
                            "completed_executions_preserved": True,
                        },
                    )
        duplicate_policy_upgrade = previous.get("duplicate_plan_policy") == (
            "random_unseen_completion_within_each_query_round"
        )
        if duplicate_policy_upgrade:
            upgraded["duplicate_plan_policy"] = DUPLICATE_PLAN_POLICY
        if upgraded != frozen:
            if not accept_workload_update(
                output,
                {**previous, "duplicate_plan_policy": DUPLICATE_PLAN_POLICY}
                if duplicate_policy_upgrade
                else previous,
                frozen,
                resume_round=resume_round,
            ):
                raise ValueError("Frozen feedback training inputs or settings changed")
        elif previous.get("query_sampling_policy") is None:
            save(
                output / "sampling-policy-update.json",
                {
                    "from": "independent_sampling_per_search",
                    "to": "once_per_query_round_v1",
                    "time": now(),
                    "preserved_executions": saved_executions(output),
                    "existing_completed_queries_are_not_rerun": True,
                },
            )
        if duplicate_policy_upgrade:
            save(
                output / "duplicate-policy-update.json",
                {
                    "time": now(),
                    "from": previous["duplicate_plan_policy"],
                    "to": DUPLICATE_PLAN_POLICY,
                    "applies_to": "newly selected rollouts; completed executions preserved",
                    "accuracy_floor_preference_unchanged": True,
                },
            )
    save(protocol_path, frozen)
    recollection_path = output / "recollection.json"
    recollection = None
    if recollect_queries:
        recollection = start_recollection(
            output, recollect_queries, rounds, collection_checkpoint, SEARCHES_PER_QUERY
        )
    save(output / "models.json", models)
    save(output / "model-catalog.json", catalog)
    save(
        output / "workloads.json",
        {
            "suite": "training-manifest"
            if custom_workloads is not None
            else "public-queries-intentionally-used-for-training",
            "workloads": workloads,
        },
    )
    save(
        output / "reference_answers_path.json",
        {"root": str(references), "system": "semcache"},
    )
    save(output / "cpu-resources.json", configure_cpu_resources(settings))
    initial = collection_checkpoint if collect_only else output / "initial/value.pt"
    if not collect_only:
        initial_checkpoint(initial, settings)
    total_executions = sum(frozen["round_query_counts"]) * SEARCHES_PER_QUERY
    restored_executions = saved_executions(output)
    state = {
        "status": "running",
        "phase": "initial_calibration",
        "pid": os.getpid(),
        "round": 0,
        "rounds": rounds,
        "query_count": query_count,
        "round_query_counts": frozen["round_query_counts"],
        "current_query": None,
        "completed_executions": restored_executions,
        "restored_executions": restored_executions,
        "total_executions": total_executions,
        "searches_per_query": SEARCHES_PER_QUERY,
        "minimum_accuracies": ACCURACY_MINIMUMS,
        "started_at": now(),
        "collect_only": collect_only,
    }
    (output / "feedback-training.pid").write_text(str(os.getpid()) + "\n")
    state_lock, stopping = threading.RLock(), threading.Event()

    def report(**values):
        with state_lock:
            if "completed_executions" in values:
                verified = saved_executions(output)
                values["verified_executions"] = verified
                values["completed_executions"] = verified
            state.update(values, updated_at=now())
            save(output / "pipeline-status.json", state)
            if not collect_only:
                write_training_report(output, state, settings)
            else:
                (output / "模型训练.md").write_text(
                    "# 仅收集数据\n\n模型拟合已停止。本次只收集查询执行数据，不更新模型参数。\n\n"
                    "[收集执行明细](执行明细.md)\n"
                )
            lines = [
                f"# {query_count}条查询 · 模型训练",
                "",
                f"状态：{state['status']}；阶段：{state['phase']}；轮次 {state['round']}/{rounds}；累计执行 {state['completed_executions']}/{total_executions}。",
                f"当前：{state.get('current_query') or '—'}；更新：{state['updated_at']}",
                "",
                (
                    f"只收集这{query_count}条查询的执行数据；固定使用已有检查点 {collection_checkpoint}，不初始化或拟合模型、不更新模型参数。"
                    if collect_only
                    else f"这{query_count}条查询作为训练集。随机初始化；无预训练拟合；仅执行必要的初始算子校准。"
                ),
                "首轮：关系过滤 > 关系join > 语义过滤 > 语义join > 关系projection > 语义projection。后续轮：关系过滤 > {关系join、语义过滤、语义join} > 关系projection > 语义projection。",
                "各轮均由value model在当前最高合法组中选择算子、顺序、模型和layout；Beam90、20%随机探索；模型E2B/E4B/12B；OTD/ODT/DOT。",
                "每条查询独立搜索三次，准确率下限依次20%、50%、80%，每次执行一个最快预计达标计划；没有预计达标候选时回退到最高预计准确率并明确标记。若与该查询本轮前面选中的计划重复，从未重复候选中随机挑选。每轮查询随机打乱。CPU16、查询串行。"
                + (
                    f"全部模型及teacher共享HTTP总并发上限{global_http_limit}。"
                    if global_http_limit
                    else "每模型并发16。"
                ),
                "训练标签按可比较中间状态合并，同组完整执行配对取平均时间和平均F1；模型不输入准确率下限，一次预测一对数；未展开算子使用已采样的选择率。",
                (
                    "后续拟合排除以下轮次的执行经验："
                    + "、".join(map(str, sorted(excluded_rounds)))
                    + "；保留已完成模型权重作为后续搜索和拟合的起点。"
                    if excluded_rounds
                    else "后续拟合合并当前查询清单内的历轮执行经验。"
                ),
                "训练期间关闭prompt回答缓存；耗时复用真实请求并持续合并到全局，准确率按模型/算子类型/OTD、ODT、DOT持续合并。",
                f"每个计划执行超时上限：{settings['training']['timeout']:g}秒。",
                "每条查询每轮只采样一次，三次独立Beam搜索共享算子统计；后续搜索使用当前缓存历史与全局校准，不重复计入采样请求。",
                "选择率采样每3个不同输入分别交给E2B/E4B/12B并行处理，合并为算子级选择率；初始20、最多50个输入单位，零输出到上限时估为1/50。每个HTTP只处理1个输入。",
                "参考为已保存的SemCache/OpenCode答案；F1忽略自由文本，聚合比较实际聚合前记录。Reward=(-实际执行秒数,+F1)，不包含Beam和采样；超时不伪造完整reward。",
                "训练准确率标签：同一查询同一轮的三个计划收集完后，仅最快完整执行计划的原始F1为0时设为0.2；没有10秒限制，其他零分保持0。表中F1展示训练标签，原始F1保存在raw_accuracy和quality.accuracy中；测试评分不应用此规则。",
                "",
            ]
            live = live_execution_row(output, state)
            active = live_execution_text(live)
            if (
                state.get("verified_executions", restored_executions)
                < restored_executions
            ):
                active = (
                    f"恢复校验：{state.get('verified_executions', 0)}/{restored_executions}；"
                    "累计执行按已保存结果统计，不重新执行已完成计划。"
                )
            if active:
                lines.insert(4, active)
            lines = with_global_calibration(lines, output)
            calibration_live = output / "calibration/progress.json"
            if calibration_live.exists():
                calibration_status = read(calibration_live)
                lines.extend(
                    [
                        f"初始校准：{calibration_status.get('phase', '—')}，{calibration_status.get('done', 0)}/{calibration_status.get('total', '—')}；仅校准统计，不训练网络。",
                        "",
                    ]
                )
            reward_rows = []
            for number in range(1, rounds + 1):
                directory = output / f"measured/round-{number}"
                round_meta = (
                    read(directory / "round.json")
                    if (directory / "round.json").exists()
                    else None
                )
                if round_meta is None:
                    continue
                epoch_file = directory / "model/epochs.jsonl"
                if epoch_file.exists() and not collect_only:
                    epochs = [
                        json.loads(line)
                        for line in epoch_file.read_text().splitlines(keepends=True)
                        if line.endswith("\n") and line.strip()
                    ]
                    if epochs:
                        last = epochs[-1]
                        lines.extend(
                            [
                                f"第{number}轮拟合：epoch {last['epoch']}；loss {last['value_loss']:.6f}。",
                                "",
                            ]
                        )
                for identity in round_meta["query_ids"]:
                    for index in range(1, SEARCHES_PER_QUERY + 1):
                        leaf = directory / "stream" / identity / f"search-{index}"
                        experience_path = leaf / "experience.json"
                        if not experience_path.exists():
                            continue
                        record = read_experience(experience_path)["records"][0]
                        reward = record.get("reward") or execution_reward(record)
                        selection = record["metadata"].get("selection_policy", "—")
                        reward_rows.append(
                            {
                                "round": number,
                                "query": identity,
                                "execution": index,
                                "selection": selection,
                                "minimum_accuracy": record["metadata"].get(
                                    "minimum_accuracy"
                                ),
                                "status": record["status"],
                                "reward": reward,
                                "artifact": str(leaf),
                            }
                        )
            reward_rows = enrich_rows(reward_rows)
            save(output / "rewards.json", reward_rows)
            display_rows = reward_rows + ([live] if live else [])
            lines.extend(render_table(display_rows))
            if state.get("error"):
                lines.extend(["", "错误：" + state["error"]])
            details = output / "执行明细.md.tmp"
            details.write_text("\n".join(lines) + "\n")
            details.replace(output / "执行明细.md")
            overview = [lines[0], "", lines[2], lines[3]]
            if active:
                overview.append(active)
            overview.extend(
                [
                    "",
                    "[查看完整执行明细](执行明细.md)",
                    "",
                    "[查看模型拟合进度](模型训练.md)",
                ]
            )
            if state.get("error"):
                overview.extend(["", "错误：" + state["error"]])
            temporary = output / "progress.md.tmp"
            temporary.write_text("\n".join(overview) + "\n")
            temporary.replace(output / "progress.md")
            save(output / "execution-details.json", display_rows)

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
                    prepared_statistics=None,
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
                        searches_per_query=SEARCHES_PER_QUERY,
                        prepared_statistics=prepared_statistics,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=ROOT.parent / "experiments/results/semcache/model_training/feedback-training",
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=ROOT.parent / "experiments/results/reference_answers/semcache/testing",
    )
    parser.add_argument("--workloads", type=Path)
    parser.add_argument("--baseline-protocol", type=Path)
    parser.add_argument("--global-http-limit", type=int)
    parser.add_argument(
        "--execution-timeout",
        type=float,
        help="Fixed per-plan execution timeout in seconds",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Lower the fitting learning rate when resuming saved feedback",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Collect execution data without fitting or writing model weights",
    )
    parser.add_argument(
        "--collection-checkpoint",
        type=Path,
        help="Existing fixed value checkpoint for collection-only search",
    )
    parser.add_argument(
        "--recollect-query",
        action="append",
        default=[],
        help="Overwrite only this query's executions in every round; collect-only",
    )
    parser.add_argument("--exclude-training-rounds", type=int, nargs="+", default=())
    parser.add_argument(
        "--preserve-resume-cache",
        action="store_true",
        help="Keep live KV/history when continuing the same round after a process update",
    )
    parser.add_argument(
        "--resume-round",
        type=int,
        help="Continue a changed query cohort after earlier fitted rounds",
    )
    args = parser.parse_args()
    run(
        args.run,
        args.references,
        workloads_path=args.workloads,
        baseline_protocol=args.baseline_protocol,
        global_http_limit=args.global_http_limit,
        execution_timeout=args.execution_timeout,
        resume_round=args.resume_round,
        rounds=args.rounds,
        exclude_training_rounds=args.exclude_training_rounds,
        preserve_resume_cache=args.preserve_resume_cache,
        learning_rate=args.learning_rate,
        collect_only=args.collect_only,
        collection_checkpoint=args.collection_checkpoint,
        recollect_queries=args.recollect_query,
    )


if __name__ == "__main__":
    main()
