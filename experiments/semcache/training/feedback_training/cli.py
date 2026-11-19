"""Progress rendering and CLI entrypoint for feedback training."""

import argparse
import json
from pathlib import Path

from planning.settings.configuration import (
    ROOT,
)
from training.artifacts import read, save
from training.collection.labeling.rewards import execution_reward
from training.reporting.telemetry import live_execution_row, live_execution_text
from training.reporting.training import (
    enrich_rows,
    read_experience,
    render_table,
    saved_executions,
    with_global_calibration,
    write_training_report,
)

from .preparation import now

ACCURACY_MINIMUMS = (0.2, 0.5, 0.8)
SEARCHES_PER_QUERY = len(ACCURACY_MINIMUMS)


def render_progress_report(ctx, **values):
    output = ctx.output
    settings = ctx.settings
    collect_only = ctx.collect_only
    collection_checkpoint = ctx.collection_checkpoint
    rounds = ctx.rounds
    query_count = ctx.query_count
    total_executions = ctx.total_executions
    restored_executions = ctx.restored_executions
    global_http_limit = ctx.global_http_limit
    excluded_rounds = ctx.excluded_rounds
    state = ctx.state
    state_lock = ctx.state_lock
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
                f"本地部署模型共享HTTP总并发上限{global_http_limit}（外网Teacher独立并发上限{settings['execution'].get('max_concurrent_external_teacher_requests', 16)}，不与本地模型共享）。"
                if global_http_limit
                else f"每模型并发16（外网Teacher独立并发上限{settings['execution'].get('max_concurrent_external_teacher_requests', 16)}）。"
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
        if state.get("verified_executions", restored_executions) < restored_executions:
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


def main(run_fn=None):
    if run_fn is None:
        from . import run as run_fn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=Path,
        default=ROOT.parent
        / "experiments/results/semcache/model_training/feedback-training",
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=ROOT.parent / "experiments/results/reference_answers/semcache/testing",
    )
    parser.add_argument(
        "--workloads",
        type=Path,
        default=ROOT.parent / "experiments/semcache/queries/workloads.json",
    )
    parser.add_argument(
        "--allow-heldout-training",
        action="store_true",
        help="Explicitly allow training on heldout benchmark queries",
    )
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
    workloads_arg = args.workloads
    if (
        args.allow_heldout_training
        and workloads_arg == ROOT.parent / "experiments/semcache/queries/workloads.json"
    ):
        workloads_arg = None
    run_fn(
        args.run,
        args.references,
        workloads_path=workloads_arg,
        allow_heldout_training=args.allow_heldout_training,
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
