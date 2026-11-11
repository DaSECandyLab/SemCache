"""Wait for the frozen reference bank, then build and run configured training."""

import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import psutil
from planning.settings.configuration import ROOT, load_config, runtime_path

from training.artifacts import save
from training.references.answers import TrainingAnswers


def validate_native_training(query):
    """Check training-only search policies without samples or model requests."""
    from planning.search.beam import ParetoBeamSearch, SearchConfig
    from planning.state import SearchProblem

    class FixedScores:
        value_available = True

        def scores(self, problem, states):
            return [(-float(len(state.actions)), 0.75) for state in states]

    checks = {}
    for mode in ("heuristic_value", "value"):
        search = ParetoBeamSearch(
            FixedScores(),
            SearchConfig(
                plan_generation=mode,
                training=True,
                exploration_fraction=0.2,
                beam_width=5,
                max_expansions=4096,
                max_seconds=60,
                seed=17,
            ),
        )
        result = search.run(SearchProblem(query))
        if not result["complete"] or not any(
            row["random_exploration"] for row in result["trace"]
        ):
            raise RuntimeError(f"Native training search check failed: {mode}")
        checks[mode] = {
            "first_layer_candidates": result["trace"][0]["enumerated"],
            "completions": len(result["complete"]),
            "exploration_slots": [row["random_exploration"] for row in result["trace"]],
        }
    if (
        checks["value"]["first_layer_candidates"]
        <= checks["heuristic_value"]["first_layer_candidates"]
    ):
        raise RuntimeError("Pure value search still applies heuristic tiers")
    return checks


def run(reference, directory, config_path):
    settings = load_config(config_path)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "sequence.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (directory / "sequence.pid").write_text(str(os.getpid()) + "\n")
        manifest = json.loads((directory / "workloads.json").read_text())
        state = {
            "status": "waiting_references",
            "reference_root": str(reference),
            "training_root": str(directory),
            "total_queries": len(manifest["workloads"]),
            "schedule": ["pretrain", "heuristic_value", "value", "value"],
            "exploration_fraction": settings["training"]["exploration_fraction"],
            "query_order": "shuffled_each_round",
            "execution_selection": "two_search_two_execution_v1",
        }

        def publish(**changes):
            state.update(changes)
            state["updated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
            save(directory / "sequence.json", state)
            (directory / "训练流程.md").write_text(
                "# 训练流程\n\n"
                f"状态：{state['status']}；答案 {state.get('references_completed', 0)}/{state['total_queries']}\n\n"
                "答案收集完成并逐条核验后，执行一次预训练，再执行三轮真实反馈训练。\n"
                "第一轮：模式2约束结构，value model选择模型与layout；第二、三轮：纯value model。\n"
                "每轮重新打乱查询，Beam保留20%随机探索；每条查询分别搜索和执行两次，先前沿随机，再选最高预测质量，首次已最高则第二次选最快。\n"
                "资源：每模型并发16，CPU／执行及模型训练线程16，查询串行。\n\n"
                f"进度：{state.get('error', '')}\n"
            )

        try:
            publish()
            while True:
                progress = json.loads((reference / "progress.json").read_text())
                publish(references_completed=progress["completed"])
                if progress["status"] == "completed":
                    if progress["completed"] != len(manifest["workloads"]):
                        raise ValueError(
                            "Reference completion count differs from workload"
                        )
                    break
                if progress["status"] != "running":
                    raise RuntimeError(f"Reference collection is {progress['status']}")
                pid = int((reference / "launcher.pid").read_text())
                if (
                    not psutil.pid_exists(pid)
                    or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
                ):
                    raise RuntimeError("Reference collector exited before completing")
                time.sleep(10)
            answers = TrainingAnswers(
                reference,
                manifest["workloads"],
                runtime_path(settings, "source_databases"),
            )
            for workload in manifest["workloads"]:
                answers.load(workload)
            # Wait for the collector to unload its native extensions before
            # replacing the single production build in place.
            pid = int((reference / "launcher.pid").read_text())
            while psutil.pid_exists(pid):
                process = psutil.Process(pid)
                if (
                    process.status() == psutil.STATUS_ZOMBIE
                    or "reference_answers.semcache" not in process.cmdline()
                ):
                    break
                time.sleep(1)
            publish(status="building", references_verified=len(manifest["workloads"]))
            with (directory / "build.log").open("a") as log:
                for build, targets in (
                    (
                        runtime_path(settings, "build"),
                        ["beam_search_loadable_extension"],
                    ),
                    (runtime_path(settings, "native_bindings").parent, []),
                ):
                    command = [
                        "cmake",
                        "--build",
                        str(build),
                        "--parallel",
                        str(settings["statistics"]["build_parallelism"]),
                    ]
                    if targets:
                        command += ["--target", *targets]
                    subprocess.run(
                        command,
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            query = json.loads((directory / "native-preflight-query.json").read_text())
            save(
                directory / "native-training-preflight.json",
                validate_native_training(query),
            )
            publish(status="training")
            from train import record_native_build, run_training

            record_native_build(directory, settings)
            code = run_training(directory, settings, config_path)
            if code:
                raise RuntimeError(f"Training exited with status {code}")
            publish(status="completed")
        except BaseException as error:
            publish(status="failed", error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.references.resolve(), args.run.resolve(), args.config.resolve())


if __name__ == "__main__":
    main()
