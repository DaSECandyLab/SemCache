"""Append and execute MOAR only in an existing frozen comparison campaign."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

from experiments.core.io import digest, save
from experiments.core.paths import ROOT
from experiments.system_comparison.scheduling.campaign import execute


def prepare(run, references, agent_config):
    config = json.loads((run / "campaign.json").read_text())
    frozen = json.loads((run / "inputs/manifest.json").read_text())
    variant = {"name": "moar-pool", "system": "moar"}
    if not any(v["name"] == variant["name"] for v in config["variants"]):
        config["variants"].append(variant)
    config["reference_directory"] = str(references.resolve())
    config["moar_agent_config"] = str(agent_config.resolve())
    config["moar_agent_config_sha256"] = digest(agent_config)
    tasks = []
    for query in config["query_ids"]:
        source = ROOT / "benchmarks/queries/translated/nirvana" / (query + ".py")
        target = run / "queries/moar" / (query + ".py")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        frozen["query_files"]["moar/" + query] = {
            "path": str(target), "sha256": digest(target), "source": str(source.relative_to(ROOT)),
            "translation": "MOAR DocETL pipeline generated from existing frozen-equivalent task definitions; CTE aliases simplified",
        }
        base = frozen["jobs"][query]
        # These source databases have identical full inputs for the Python
        # systems. MOAR does not need to manufacture a different data copy.
        base["input_databases"]["moar"] = dict(base["input_databases"]["nirvana"])
        task = {"task_id": "moar-pool/" + query, "variant": "moar-pool", "system": "moar",
                "query_id": query, "folder": "tasks/moar-pool/" + query + "/attempt-1"}
        tasks.append(task)
        if not any(t["task_id"] == task["task_id"] for t in config["tasks"]):
            config["tasks"].append(task)
    config["moar"] = {"optimizer": "native MCTS", "max_iterations": 20, "max_threads": 16,
                      "max_concurrent_agents": 3, "max_requests_in_flight": 16, "batch_size": 1,
                      "model_pool": ["semantic-e2b", "semantic-e4b", "semantic-large"], "selection": "best()",
                      "capture_aggregate_inputs": True, "preserve_model_cache": True}
    teacher = json.loads(agent_config.read_text())
    config["moar"]["agent"] = {"config": config["moar_agent_config"], "sha256": config["moar_agent_config_sha256"],
                              "model": teacher["model"], "base_url": teacher["base_url"],
                              "reasoning_effort": teacher.get("reasoning_effort"), "thinking": teacher.get("thinking")}
    save(run / "campaign.json", config)
    save(run / "inputs/manifest.json", frozen)
    return config, frozen, tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--agent-config", type=Path, default=ROOT / "SemCache/config/teacher/teacher.opencode.json")
    parser.add_argument("--retry-failed", action="store_true", help="Overwrite incomplete MOAR attempts, retaining completed outputs")
    args = parser.parse_args()
    run = args.run.resolve()
    with (run / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config, frozen, tasks = prepare(run, args.references, args.agent_config)
        if args.retry_failed:
            for task in tasks:
                folder = run / task["folder"]
                if folder.exists():
                    path = folder / "summary.json"
                    if not path.exists() or json.loads(path.read_text()).get("status") != "completed":
                        shutil.rmtree(folder)
        if args.prepare_only:
            print(f"Prepared {len(tasks)} genuine MOAR tasks")
            return
        save(run / "moar-launch.json", {"pid": os.getpid(), "started_unix": time.time(), "entrypoint": __name__})
        execute(run, {**config, "tasks": tasks}, frozen, persist_campaign=False,
                cpu_set=config["model_lanes"][0]["cpu_set"], write_shared_metadata=False,
                reset_variant_caches=False, final_phase="moar_finished")


if __name__ == "__main__":
    main()
