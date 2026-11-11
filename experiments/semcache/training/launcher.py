#!/usr/bin/env python3
"""SemCache 实验训练后台启动器，负责采集、续训、停止和报告编排。"""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "SemCache"
sys.path.insert(0, str(ROOT / "optimizer"))
sys.path.insert(0, str(ROOT.parent / "experiments/semcache"))
from planning.settings.configuration import (  # noqa: E402
    DEFAULT_CONFIG,
    activate_native,
    load_config,
    runtime_path,
    teacher_path,
)

CONFIG_PATH = DEFAULT_CONFIG
CONFIG = load_config()
RUN_ROOT = runtime_path(CONFIG, "run_root")
RUN_DIR = RUN_ROOT / "train"
if CONFIG["training"].get("dataset") is not None:
    RUN_DIR /= CONFIG["training"]["dataset"]


@contextmanager
def run_directory(run, *, resume=False):
    """Reuse one directory; never overwrite an active run or follow symlinks."""
    if run.is_symlink():
        raise ValueError(f"训练目录不能是符号链接：{run}")
    run.mkdir(parents=True, exist_ok=True)
    with (run / "reset.lock").open("a") as startup:
        fcntl.flock(startup, fcntl.LOCK_EX | fcntl.LOCK_NB)
        launcher = run / "launcher.pid"
        if launcher.exists() and is_launcher(int(launcher.read_text())):
            raise RuntimeError(f"训练仍在运行：{run}")
        if not resume:
            locks = {"sequence.lock", "monitor.lock", "reset.lock"}
            with ExitStack() as stack:
                for name in ("sequence.lock", "monitor.lock"):
                    guard = stack.enter_context((run / name).open("a"))
                    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
                for path in run.iterdir():
                    if path.name in locks:
                        continue
                    if path.is_dir() and not path.is_symlink():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
        yield


def start_background(run):
    """父进程返回终端，子进程独立运行；终端断开不影响任务。"""
    sys.stdout.flush()
    sys.stderr.flush()
    pid = os.fork()
    if pid:
        (run / "launcher.pid").write_text(f"{pid}\n")
        print(
            f"后台 PID: {pid}\n目录: {run}\n采集: {run / '预训练数据收集.md'}\n真实采集: {run / '真实训练数据收集.md'}\n训练: {run / '模型训练.md'}\n日志: {run / 'train.log'}"
        )
        return True
    os.setsid()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    with open(os.devnull, "rb") as source, (run / "launcher.log").open("a") as log:
        os.dup2(source.fileno(), 0)
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
    return False


def stop(process, signum):
    """先正常退出，超过退出等待时间仍未结束才终止整个子进程组。"""
    if process is None or process.poll() is not None:
        return

    def signal_group(value):
        if hasattr(process, "signal_group"):
            process.signal_group(value)
        else:
            os.killpg(process.pid, value)

    signal_group(signum)
    try:
        process.wait(timeout=CONFIG["runtime"]["shutdown_timeout_seconds"])
    except subprocess.TimeoutExpired:
        signal_group(signal.SIGKILL)
        process.wait()


def resume_configuration(run):
    """恢复冻结采集参数；重试与每轮模型训练策略使用当前配置。"""
    frozen = json.loads((run / "protocol.json").read_text())["configuration"]
    frozen["training"]["network_retry"] = CONFIG["training"].get("network_retry", {})
    for key in ("round_update", "checkpoint_epochs"):
        if key in CONFIG["training"]:
            frozen["training"][key] = CONFIG["training"][key]
    frozen["inference"]["gpu_batch_size"] = CONFIG["inference"]["gpu_batch_size"]
    frozen["search"]["workers"] = CONFIG["search"]["workers"]
    if "round_update" in frozen["training"]:
        frozen["training"].pop("final_update", None)
    return frozen


def record_native_build(run, config):
    """记录本次实际二进制；构建目录始终允许被原地重建。"""
    build = runtime_path(config, "build")
    files = [
        build / "src/libduckdb.so",
        build / "extension/semantic/semantic.duckdb_extension",
        build / "extension/beam_search/beam_search.duckdb_extension",
    ]
    files.extend(
        sorted(
            (runtime_path(config, "native_bindings") / "planning/kernels").glob(
                "_search_core*.so"
            )
        )
    )
    artifacts = {}
    for path in files:
        with path.open("rb") as stream:
            artifacts[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    path = run / "native-build.json"
    path.write_text(
        json.dumps(
            {
                "sha256": artifacts,
                "recorded_at": datetime.now().astimezone().isoformat(),
                "recorded_on_resume": (run / "protocol.json").exists(),
                "build_policy": "single in-place build; overwritten before every native build",
            },
            indent=2,
        )
        + "\n"
    )


def run_training(run, config, config_path=CONFIG_PATH, *, training=None):
    from planning.settings.parallelism import configure_cpu_resources

    configure_cpu_resources(config)
    paths = [
        str(runtime_path(config, "native_bindings")),
        str(ROOT / "optimizer"),
        str(ROOT.parent / "experiments/semcache"),
        str(ROOT.parent),
    ]
    os.environ["PYTHONPATH"] = os.pathsep.join(
        paths + [os.environ.get("PYTHONPATH", "")]
    )
    sys.path[:0] = paths
    if training is None:
        activate_native(config)
    else:
        training.validate()
    interval = config["monitoring"]["refresh_seconds"]

    def launch(module, arguments, log):
        with (run / log).open("a") as stream:
            return subprocess.Popen(
                [str(runtime_path(config, "python")), "-u", "-m", module, *arguments],
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

    monitor = None
    try:
        arguments = ["benchmark", "--run", str(run)]
        if config_path != CONFIG_PATH:
            arguments += [
                "--config",
                str(config_path),
                "--teacher-config",
                str(teacher_path(config)),
            ]
        if training is None:
            training = launch("planning", arguments, "train.log")
        monitor = launch(
            "experiments.semcache.monitoring.monitor",
            ["--run", str(run), "--interval", str(interval), "--reports"],
            "monitor.log",
        )
        while True:
            try:
                return training.wait(timeout=interval)
            except subprocess.TimeoutExpired:
                if monitor.poll() is not None and training.poll() is None:
                    raise RuntimeError(f"监控已退出，请查看 {run / 'monitor.log'}")
    finally:
        stop(training, signal.SIGINT)
        stop(monitor, signal.SIGTERM)
        # 启动器可以来自系统 Python；最终报告与训练一样使用配置的环境。
        monitor = launch(
            "experiments.semcache.monitoring.monitor",
            ["--run", str(run), "--once", "--reports"],
            "monitor.log",
        )
        try:
            if monitor.wait():
                raise RuntimeError(f"最终报告刷新失败，请查看 {run / 'monitor.log'}")
        finally:
            stop(monitor, signal.SIGTERM)


def is_launcher(pid):
    """核对 PID 的实际脚本路径，跳过已退出或被其他程序复用的 PID。"""
    if pid <= 1 or pid == os.getpid():
        return False
    proc = Path(f"/proc/{pid}")
    try:
        if proc.joinpath("stat").read_text().rsplit(")", 1)[1].split()[0] == "Z":
            return False
        for arg in proc.joinpath("cmdline").read_bytes().split(b"\0")[1:]:
            path = Path(os.fsdecode(arg))
            if path.name in ("launcher.py", "pretrain.py"):
                return (proc / "cwd" / path).resolve().parent == ROOT
    except (OSError, ValueError, IndexError):
        pass
    return False


def stop_runs():
    """通知本脚本启动的所有存活任务退出，由管理进程清理训练和监控。"""
    found = False
    # Stop the scheduler first so it cannot resume a dataset just stopped below.
    automatic = RUN_ROOT / "train/automatic.pid"
    if automatic.exists():
        try:
            pid = int(automatic.read_text())
            command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if b"training.automatic" in command:
                os.kill(pid, signal.SIGTERM)
                print("已请求停止自动答案收集与分数据集训练调度")
                found = True
        except (ProcessLookupError, ValueError, FileNotFoundError):
            pass
    for path in RUN_ROOT.rglob("feedback-training.pid"):
        try:
            pid = int(path.read_text())
            command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if (
                b"training.feedback_training" in command
                and Path(f"/proc/{pid}/cwd").resolve() == ROOT
            ):
                os.kill(pid, signal.SIGINT)
                print(f"已请求停止反馈训练：{path.parent}")
                found = True
        except (ProcessLookupError, ValueError, FileNotFoundError):
            pass
    for path in RUN_ROOT.rglob("launcher.pid"):
        try:
            pid = int(path.read_text().strip())
            if is_launcher(pid):
                os.kill(pid, signal.SIGTERM)
                print(f"已请求停止：{path.parent}（PID {pid}），正在清理训练和监控进程")
                found = True
        except (ProcessLookupError, ValueError, FileNotFoundError):
            continue
    if not found:
        print("没有找到正在运行的训练任务")
    return 0


def attach_training(run):
    """Replace supervision only; retain the running controller and its state."""
    sys.path.insert(0, str(ROOT.parent))
    from experiments.semcache.monitoring.process import AttachedTraining

    config = json.loads((run / "protocol.json").read_text())["configuration"]
    if config["monitoring"]["refresh_seconds"] <= 0:
        raise ValueError("refresh_seconds must be positive")
    launcher = run / "launcher.pid"
    if launcher.exists() and is_launcher(int(launcher.read_text())):
        raise RuntimeError(f"原启动器仍在运行，请先完成监督交接：{run}")
    # Fail before taking responsibility for the pipeline if the old monitor is alive.
    with (run / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with AttachedTraining(run, ROOT, runtime_path(config, "python")) as training:
        return (
            0 if start_background(run) else run_training(run, config, training=training)
        )


def main():
    usage = (
        "直接运行以启动，--dataset 数据集 独立训练，--run 目录 指定新运行目录，stop 终止，resume 目录 继续，"
        "attach 目录 接管已有训练监控，references 目录 生成参考答案"
    )
    if sys.argv[1:] in (["--help"], ["-h"]):
        argparse.ArgumentParser(description=usage).print_help()
        return 0
    if sys.argv[1:2] == ["references"]:
        parser = argparse.ArgumentParser(
            description="SemCache 启发式计划与 GOTDF 生成训练教师答案"
        )
        parser.add_argument("run", type=Path, help="含 workloads.json 的目录")
        parser.add_argument("--output", type=Path, help="答案目录；续跑时指定已有目录")
        args = parser.parse_args(sys.argv[2:])
        run = args.run.resolve()
        manifest = run / "workloads.json"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        output = (
            args.output.resolve()
            if args.output is not None
            else ROOT
            / "experiments/results/reference_answers/semcache/training"
            / "answers-magpie"
        )
        from training.artifacts import save

        save(
            run / "reference_answers_path.json",
            {"root": str(output), "system": "semcache"},
        )
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [
                str(ROOT / "optimizer"),
                str(ROOT.parent / "experiments/semcache"),
                str(runtime_path(CONFIG, "native_bindings")),
                str(ROOT.parent),
                *filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep)),
            ]
        )
        os.execv(
            str(runtime_path(CONFIG, "python")),
            [
                str(runtime_path(CONFIG, "python")),
                "-u",
                "-m",
                "reference_answers.semcache",
                "--workloads",
                str(manifest),
                "--output",
                str(output),
            ],
        )
    if sys.argv[1:] in (["stop"], ["--stop"]):
        return stop_runs()
    if len(sys.argv) == 3 and sys.argv[1] == "attach":
        return attach_training(Path(sys.argv[2]).resolve())
    resume = len(sys.argv) == 3 and sys.argv[1] == "resume"
    named_run = len(sys.argv) == 3 and sys.argv[1] == "--run"
    named_dataset = len(sys.argv) == 3 and sys.argv[1] == "--dataset"
    if len(sys.argv) != 1 and not resume and not named_run and not named_dataset:
        raise ValueError(usage)
    run = Path(sys.argv[2]).resolve() if resume or named_run else RUN_DIR
    config = resume_configuration(run) if resume else CONFIG
    selected_dataset = (
        sys.argv[2] if named_dataset else config["training"].get("dataset")
    )
    if not resume and selected_dataset is not None:
        from training.datasets import scoped_configuration

        config = scoped_configuration(config, selected_dataset, CONFIG_PATH)
        if not named_run:
            run = RUN_ROOT / "train" / selected_dataset
    if (run / "launcher.pid").exists() and is_launcher(
        int((run / "launcher.pid").read_text())
    ):
        raise RuntimeError(f"训练仍在运行：{run}")
    if config["monitoring"]["refresh_seconds"] <= 0:
        raise ValueError("refresh_seconds must be positive")
    for path in (
        runtime_path(config, "python"),
        teacher_path(config),
        CONFIG_PATH,
        runtime_path(config, "build") / "src/libduckdb.so",
        runtime_path(config, "build") / "extension/semantic/semantic.duckdb_extension",
        runtime_path(config, "build")
        / "extension/beam_search/beam_search.duckdb_extension",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not list(
        (runtime_path(config, "native_bindings") / "planning/kernels").glob(
            "_search_core*.so"
        )
    ):
        raise FileNotFoundError(
            f"缺少配套原生绑定：{runtime_path(config, 'native_bindings')}"
        )
    with run_directory(run, resume=resume):
        record_native_build(run, config)
        config_path = CONFIG_PATH
        if resume:
            config_path = run / "resume-config.json"
            config_path.write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n"
            )
        elif selected_dataset is not None:
            config_path = run / "training-config.json"
            config_path.write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n"
            )
        return 0 if start_background(run) else run_training(run, config, config_path)


def interrupt(_signum, _frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupt)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
