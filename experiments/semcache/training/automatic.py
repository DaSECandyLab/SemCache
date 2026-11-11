"""Collect pending references while training one independent dataset at a time."""

import argparse
import fcntl
import os
import signal
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import psutil
from planning.settings.configuration import (
    DEFAULT_CONFIG,
    ROOT,
    load_config,
    runtime_path,
    teacher_path,
)

from training.artifacts import read, save
from training.references.answers import TrainingAnswers

ORDER = ("movies_nirvana", "estate", "appstore", "tpch", "movies_sema")


def live_run(directory):
    """Match the actual command and run path rather than trusting a stale PID."""
    for name, token in (("launcher.pid", "launcher.py"), ("pipeline.pid", "planning")):
        path = directory / name
        if not path.exists():
            continue
        try:
            process = psutil.Process(int(path.read_text()))
            command = process.cmdline()
            if process.status() == psutil.STATUS_ZOMBIE:
                continue
            if token == "planning":
                if "planning" in command and str(directory) in command:
                    return True
            elif any(Path(arg).name == token for arg in command):
                state = directory / "pipeline-status.json"
                if state.exists() and read(state).get("status") == "complete":
                    continue
                # The launcher's fixed dataset/run directory is recorded in its
                # own PID file; verify its command also names this dataset/run.
                if str(directory) in command or (
                    "--dataset" in command and directory.name in command
                ):
                    return True
        except (OSError, ValueError, psutil.Error):
            continue
    return False


def run(reference, directory, config_path):
    if Path(config_path).resolve() != DEFAULT_CONFIG.resolve():
        raise ValueError("Automatic dataset launches use config/system/config.json")
    settings = load_config(config_path)
    manifest = read(ROOT / settings["training"]["workload_manifest"])
    workloads = manifest["workloads"]
    groups = {name: [w for w in workloads if w["dataset"] == name] for name in ORDER}
    bank = TrainingAnswers(
        reference, workloads, runtime_path(settings, "source_databases")
    )
    directory.mkdir(parents=True, exist_ok=True)
    stopping = threading.Event()
    collection = {"status": "starting", "dataset": None}
    collection_lock = threading.Lock()

    def collect():
        try:
            for name in ("tpch", "movies_sema"):
                if stopping.is_set():
                    return
                missing = [
                    w
                    for w in groups[name]
                    if not (reference / w["id"] / "answer.json").exists()
                ]
                if not missing:
                    continue
                with collection_lock:
                    collection.update(status="running", dataset=name)
                command = [
                    str(runtime_path(settings, "python")),
                    "-u",
                    "-m",
                    "reference_answers.semcache",
                    "--output",
                    str(reference),
                    "--workloads",
                    str(reference / "inputs/workloads.json"),
                    "--datasets",
                    name,
                    "--teacher-config",
                    str(teacher_path(settings, config_path)),
                    "--parallelism",
                    str(settings["reference_answers"]["semcache"]["parallelism"]),
                    "--query-workers",
                    "2",
                    "--rate-limit-retries",
                    "2",
                ]
                with (reference / "run.log").open("a") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    with collection_lock:
                        collection["pid"] = process.pid
                    try:
                        while process.poll() is None and not stopping.wait(2):
                            pass
                        if stopping.is_set() and process.poll() is None:
                            os.killpg(process.pid, signal.SIGINT)
                            return
                        if process.returncode:
                            raise RuntimeError(
                                f"Reference collection for {name} exited with status {process.returncode}; see {reference / 'run.log'}"
                            )
                    finally:
                        with collection_lock:
                            collection.pop("pid", None)
            with collection_lock:
                collection.update(status="completed", dataset=None)
        except BaseException as error:
            with collection_lock:
                collection.update(status="failed", error=str(error))

    with (directory / "automatic.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (directory / "automatic.pid").write_text(str(os.getpid()) + "\n")
        collector = threading.Thread(target=collect, daemon=True)
        collector.start()
        verified, failed, attempted = set(), {}, set()
        state = {}

        def publish(status="running", error=None):
            with collection_lock:
                lane = dict(collection)
            payload = {
                "status": status,
                "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "reference_root": str(reference),
                "training_root": str(directory),
                "references_completed": len(verified),
                "total_queries": len(workloads),
                "collection": lane,
                "datasets": state,
                "schedule": ["pretrain", "heuristic_value", "value", "value"],
                "training_order": list(ORDER),
                "training_queries_parallel": 1,
                "simultaneous_training_datasets": 1,
            }
            if error is not None:
                payload["error"] = error
            save(directory / "automatic.json", payload)
            save(directory / "sequence.json", payload)
            lines = [
                "# 自动收集与分数据集训练",
                "",
                f"状态：{status}；参考答案 {len(verified)}/{len(workloads)}。",
                f"答案收集：{lane['status']}；当前数据集：{lane.get('dataset') or '-'}。",
                "",
                "收集与训练同时进行；同一时间训练一个数据集。每个数据集答案齐全并核验后即可训练。",
                "各数据集独立预训练及三轮训练，分别保存value model。第一轮启发式约束结构、value model选layout与模型；后两轮纯value model。",
                "20%随机探索；各轮打乱查询；每条查询搜索并执行两次：先从前沿随机选，再重新搜索选最高预测质量；若第一次已选最高质量，第二次改选最快。每模型并发16、CPU16线程、查询串行。",
                "",
                "| 数据集 | 参考答案 | 训练状态 | 阶段 | 查询进度 |",
                "|---|---:|---|---|---:|",
            ]
            for name in ORDER:
                row = state[name]
                link = (
                    f"[{name}]({name}/预训练数据收集.md)"
                    if (directory / name / "预训练数据收集.md").exists()
                    else name
                )
                lines.append(
                    f"| {link} | {row['references']}/{row['total']} | {row['status']} | {row.get('phase', '-')} | {row.get('completed_queries', '-')} |"
                )
                if row.get("error"):
                    lines.append(f"\n{name}：{row['error']}\n")
            if lane.get("error"):
                lines.append("\n答案收集错误：" + lane["error"])
            if error is not None:
                lines.append("\n调度错误：" + error)
            lines.extend(["", f"更新时间：{payload['updated_at']}"])
            (directory / "训练流程.md").write_text("\n".join(lines) + "\n")

        try:
            while not stopping.is_set():
                for w in workloads:
                    if (
                        w["id"] not in verified
                        and (reference / w["id"] / "answer.json").is_file()
                    ):
                        bank.load(w)
                        verified.add(w["id"])
                active = []
                for name in ORDER:
                    path = directory / name
                    count = sum(w["id"] in verified for w in groups[name])
                    row = {"references": count, "total": len(groups[name])}
                    progress = (
                        read(path / "pipeline-status.json")
                        if (path / "pipeline-status.json").exists()
                        else {}
                    )
                    row.update(
                        {
                            k: progress[k]
                            for k in ("phase", "round", "completed_queries")
                            if k in progress
                        }
                    )
                    if live_run(path):
                        active.append(name)
                        row["status"] = "training"
                    elif (
                        progress.get("status") == "complete"
                        and (path / "value.pt").is_file()
                        and (path / "model.json").is_file()
                    ):
                        row["status"] = "completed"
                    elif name in attempted or name in failed:
                        failed[name] = (
                            progress.get("error")
                            or failed.get(name)
                            or f"Training stopped before completion; see {path / 'launcher.log'}"
                        )
                        row.update(status="failed", error=failed[name])
                    else:
                        row["status"] = (
                            "ready"
                            if count == len(groups[name])
                            else "waiting_references"
                        )
                    state[name] = row
                if len(active) > 1:
                    raise RuntimeError(
                        "Multiple training datasets are active; refusing to start another"
                    )
                if not active:
                    ready = next(
                        (name for name in ORDER if state[name]["status"] == "ready"),
                        None,
                    )
                    if ready:
                        path = directory / ready
                        command = [
                            str(runtime_path(settings, "python")),
                            str(ROOT.parent / "experiments/semcache/training/launcher.py"),
                        ]
                        command += (
                            ["resume", str(path)]
                            if (path / "protocol.json").exists()
                            else ["--dataset", ready]
                        )
                        with (directory / "automatic-launch.log").open("a") as log:
                            subprocess.run(
                                command,
                                cwd=ROOT,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                check=True,
                                timeout=30,
                            )
                        attempted.add(ready)
                        stopping.wait(3)
                        continue
                terminal = all(
                    row["status"] in ("completed", "failed") for row in state.values()
                )
                with collection_lock:
                    collection_done = collection["status"] in ("completed", "failed")
                publish(
                    "failed"
                    if terminal and failed
                    else "completed"
                    if terminal
                    else "running"
                )
                if terminal or (
                    collection_done
                    and not active
                    and not any(row["status"] == "ready" for row in state.values())
                ):
                    if not terminal:
                        publish("failed")
                    return
                stopping.wait(10)
        except BaseException as error:
            if state:
                publish(
                    "stopped" if isinstance(error, KeyboardInterrupt) else "failed",
                    str(error),
                )
            raise
        finally:
            stopping.set()
            collector.join(timeout=5)


def main():
    def interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--references", type=Path, required=True)
    args = parser.parse_args()
    settings = load_config(args.config)
    directory = runtime_path(settings, "run_root") / "train"
    run(args.references.resolve(), directory.resolve(), args.config.resolve())


if __name__ == "__main__":
    main()
