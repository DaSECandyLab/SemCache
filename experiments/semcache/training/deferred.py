"""Start a manifest training run after reference collection and evaluation finish."""

import argparse
import fcntl
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from reference_answers.storage.files import file_digest

from training.artifacts import read, save


def readiness(references, evaluation, workloads):
    """Only completed artifacts count; a stopped job is never a successful dependency."""
    reference = read(references / "progress.json")
    test = read(evaluation / "job-status.json")
    expected = {w["id"]: w["sql_sha256"] for w in workloads}
    if reference.get("status") != "completed" or reference.get("completed") != len(expected):
        return False, "等待训练参考答案全部完成"
    contract = read(references / "contract.json")
    if contract.get("queries") != expected:
        raise ValueError("Collected reference queries differ from the scheduled training manifest")
    if not contract["settings"].get("verify_on_second_use"):
        raise ValueError("Training references must enable second-use verification")
    for identity in expected:
        answer = read(references / identity / "answer.json")
        if answer.get("status") != "completed" or not answer.get("verify_on_second_use"):
            raise ValueError(f"Reference answer was not regenerated under verification: {identity}")
    if test.get("status") not in {"completed", "finished_with_issues"} or test.get("completed") != test.get("total"):
        return False, "等待测试的三组全部结束（包括按要求跳过的计划）"
    return True, "两个前置任务已结束，可以训练"


def run(args):
    output = args.run.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "schedule.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        workloads = read(args.workloads)["workloads"]
        query_count = len(workloads)
        if not workloads or len({w["id"] for w in workloads}) != query_count or any(
            w.get("split") != "train" for w in workloads
        ):
            raise ValueError("Expected a nonempty manifest of unique training queries")
        rounds = 3
        planned_executions = query_count * rounds * 3
        manifest_hash = file_digest(args.workloads)
        baseline_hash = file_digest(args.baseline_protocol)
        request = {
            "workloads": str(args.workloads.resolve()),
            "workloads_sha256": manifest_hash,
            "baseline_protocol": str(args.baseline_protocol.resolve()),
            "baseline_protocol_sha256": baseline_hash,
            "references": str(args.references.resolve()),
            "evaluation": str(args.evaluation.resolve()),
            "global_http_limit": args.global_http_limit,
            "queries": query_count,
            "rounds": rounds,
            "planned_executions": planned_executions,
        }
        saved = output / "schedule-request.json"
        if saved.exists() and read(saved) != request:
            raise ValueError("Scheduled training inputs changed")
        save(saved, request)

        def publish(status, message, **extra):
            stamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
            save(output / "schedule-status.json", {
                "status": status, "message": message, "updated_at": stamp,
                "pid": os.getpid(), **extra,
            })
            temporary = output / "训练安排.md.tmp"
            temporary.write_text(
                f"# {query_count}条查询训练安排\n\n"
                f"状态：{status}；{message}。更新时间：{stamp}。\n\n"
                "等待训练参考答案和20%/50%/80%三组测试全部结束后自动启动。\n\n"
                "随机初始化，不做预训练；三轮。首轮模式一细优先级，后两轮模式二优先级；"
                "由value model选择同组算子、模型和layout。每轮查询随机打乱，20%随机探索。\n\n"
                "每条查询每轮采样一次、独立搜索三次，下限20%/50%/80%；重复计划随机换选。"
                f"合计计划执行{planned_executions}次。Beam90；E2B/E4B/12B；OTD/ODT/DOT及首字段选择。\n\n"
                f"CPU16线程，单查询执行，所有模型及teacher的HTTP请求共享总并发{args.global_http_limit}。"
                "训练prompt cache和二次核对关闭，每轮开始清除模型缓存。\n\n"
                "每轮收集结束后合并历轮经验拟合：最少96、最多256个epoch，"
                "连续16个epoch监控loss无累计3%改善则停止。其余模型、采样、标签及训练参数"
                "复用基线训练的冻结配置。\n\n"
                "[执行明细](执行明细.md) · [模型训练](模型训练.md)\n"
            )
            temporary.replace(output / "训练安排.md")

        child = None
        try:
            if (output / "pipeline-status.json").exists():
                raise ValueError("Training already has pipeline state; do not launch it twice")
            while True:
                if file_digest(args.workloads) != manifest_hash or file_digest(args.baseline_protocol) != baseline_hash:
                    raise ValueError("Scheduled manifest or baseline changed while waiting")
                ready, message = readiness(args.references, args.evaluation, workloads)
                if ready:
                    break
                publish("waiting", message)
                import time
                time.sleep(30)
            command = [
                sys.executable, "-u", "-m", "training.feedback_training",
                "--run", str(output), "--references", str(args.references.resolve()),
                "--workloads", str(args.workloads.resolve()),
                "--baseline-protocol", str(args.baseline_protocol.resolve()),
                "--global-http-limit", str(args.global_http_limit),
            ]
            with (output / "training-launcher.log").open("a") as log:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                publish("training", f"前置任务已结束，{query_count}条训练已启动", training_pid=child.pid)
                code = child.wait()
            publish("completed" if code == 0 else "failed", "训练结束" if code == 0 else "训练失败，查看training-launcher.log", training_pid=child.pid, returncode=code)
        except BaseException as error:
            publish("failed", str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "references", "evaluation", "workloads", "baseline-protocol"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--global-http-limit", type=int, default=16)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
