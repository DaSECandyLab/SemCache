"""Training monitoring, with an opt-in slow-query interrupt guard."""

import argparse
import fcntl
import json
import os
import shutil
import signal
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

TZ = ZoneInfo("Asia/Shanghai")


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def alive(pid, run):
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        base = Path("/proc") / str(pid)
        state = (base / "stat").read_text().rsplit(")", 1)[1].split()[0]
        command = (base / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        return state not in ("Z", "X") and "planning" in command and str(run) in command
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False


def open_telemetry(pid):
    """Live private sample-stage logs belong to the controller, not run/sql."""
    paths = set()
    if not isinstance(pid, int) or pid <= 0:
        return paths
    try:
        descriptors = list((Path("/proc") / str(pid) / "fd").iterdir())
    except OSError:
        return paths
    for fd in descriptors:
        try:
            target = Path(os.readlink(fd))
            if target.name == "telemetry.jsonl" and target.is_file():
                paths.add(target)
        except OSError:
            continue
    return paths


class SlowQueryWatch:
    """Query-local elapsed guard, opt-in and independent of artifact activity."""

    def __init__(self, seconds):
        self.seconds = seconds
        self.key = self.started = None

    def check(self, status, now):
        if (
            status.get("status") != "running"
            or status.get("phase") != "estimated"
            or not status.get("current_query")
        ):
            self.key = self.started = None
            return None
        key = (status.get("pid"), status.get("round"), status["current_query"])
        if key != self.key:
            age = 0
            if status.get("updated_at"):
                try:
                    age = max(
                        0,
                        time.time()
                        - datetime.fromisoformat(status["updated_at"]).timestamp(),
                    )
                except (TypeError, ValueError):
                    pass
            self.key, self.started = key, now - age
        elapsed = now - self.started
        return elapsed if self.seconds > 0 and elapsed >= self.seconds else None


def same_running_query(run, status):
    current = read(run / "pipeline-status.json")
    return (
        isinstance(current, dict)
        and current.get("status") == "running"
        and current.get("phase") == "estimated"
        and bool(current.get("current_query"))
        and all(
            current.get(key) == status.get(key)
            for key in (
                "status",
                "phase",
                "pid",
                "round",
                "current_query",
                "completed_queries",
                "updated_at",
            )
        )
    )


def stop_slow_query(run, status, elapsed):
    """Capture evidence then interrupt only the verified live training PID."""
    pid = status.get("pid")
    record = {
        "reason": "query_elapsed_limit",
        "query": status.get("current_query"),
        "phase": status.get("phase"),
        "round": status.get("round"),
        "global_round": status.get("global_round"),
        "total_rounds": status.get("total_rounds"),
        "query_order": status.get("query_order"),
        "pid": pid,
        "elapsed_seconds": elapsed,
        "signal_sent": False,
        "updated_at": datetime.now(TZ).isoformat(timespec="seconds"),
    }
    if not same_running_query(run, status):
        record["reason"] = "query_changed"
        return record
    if not alive(pid, run):
        record["reason"] = "controller_not_verified"
        return record
    diagnostic = run / "slow-query-diagnostics"
    diagnostic.mkdir(exist_ok=True)
    base = Path("/proc") / str(pid)
    for name in ("status", "stat", "wchan"):
        try:
            (diagnostic / f"process-{name}.txt").write_text((base / name).read_text())
        except (OSError, PermissionError):
            pass
    # Only open model telemetry, never arbitrary process descriptors/data.
    paths = open_telemetry(pid)
    for index, path in enumerate(sorted(paths)):
        try:
            shutil.copyfile(path, diagnostic / f"telemetry-{index}.jsonl")
        except OSError:
            pass
    record["telemetry_sources"] = [str(p) for p in sorted(paths)]
    save(run / "slow-query-stop.json", record)
    # Capture can overlap a query transition in the same live controller.
    if not same_running_query(run, status):
        record["reason"] = "query_changed"
    elif alive(pid, run):
        try:
            os.kill(pid, signal.SIGINT)
            record["signal_sent"] = True
        except ProcessLookupError:
            pass
    save(run / "slow-query-stop.json", record)
    return record


def healthcheck(item):
    name, config = item
    checked = datetime.now(TZ).isoformat(timespec="seconds")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(
            config["base_url"].rstrip("/") + "/v1/models", timeout=5
        ) as response:
            identities = [entry["id"] for entry in json.load(response)["data"]]
        return name, {
            "healthy": config["model"] in identities,
            "served_models": identities,
            "checked_at": checked,
        }
    except (OSError, ValueError, KeyError) as error:
        return name, {"healthy": False, "error": str(error), "checked_at": checked}


def training_summary(report):
    if not report:
        return None
    return {
        "epochs": report.get("epochs"),
        "best_epoch": report.get("best_epoch"),
        "train_queries": report.get("queries"),
        "states": report.get("states"),
        "censored_terminal_states": report.get("censored_terminal_states"),
        "source": report.get("source"),
        "objective": report.get("objective"),
        "training_value_loss": report.get("training", {}).get("value_loss"),
        "validation_value_loss": report.get("validation", {}).get("value_loss"),
        "validation_queries": report.get("validation", {}).get("queries", 0),
    }


def snapshot(run, status, health, stalled_after):
    now = time.time()
    artifacts = [
        run / "pipeline-status.json",
        run / "pipeline.log",
        run / "train.log",
    ]
    # Per-epoch records provide the fitting heartbeat.
    artifacts.extend(run.glob("*/round-*/epochs.jsonl"))
    artifacts.extend(run.glob("*/round-*/model/epochs.jsonl"))
    artifacts.extend(run.glob("*/round-*/model/training.json"))
    query = status.get("current_query")
    if query:
        for context in ("stream", "cold", "warm"):
            for episode in ("", "search-*/"):
                for name in ("telemetry.jsonl", "profile.json"):
                    artifacts.extend(
                        run.glob(
                            f"*/round-*/{context}/{query}/{episode}sql/run-*/{name}"
                        )
                    )
    live = alive(status.get("pid"), run)
    if status.get("phase") == "value_update":
        # A round's fit writes no pipeline artifacts while it trains; one epoch
        # may legitimately take minutes, so do not call that a stall.
        stalled_after = max(stalled_after, 1800)
    if live:
        artifacts.extend(open_telemetry(status["pid"]))
    latest = max(
        (path.stat().st_mtime for path in artifacts if path.exists()), default=now
    )
    inactive = max(0, now - latest)
    alerts = []
    state = status.get("status")
    if state == "running" and not live:
        alerts.append("Controller exited while pipeline status is still running")
    if state in ("error", "stopped", "interrupted"):
        alerts.append("Pipeline stopped: " + str(status.get("error") or state))
    if (
        state == "running"
        and inactive > stalled_after
        and status.get("phase") != "waiting_shared_resources"
    ):
        alerts.append(
            "No pipeline, execution telemetry, or training log progress within the configured interval"
        )
    for name, info in health.items():
        if not info["healthy"]:
            alerts.append(
                "Model endpoint unavailable or served identity differs: " + name
            )
    if status.get("failed_queries", 0):
        alerts.append(
            "Current collection attempt has failed queries; inspect failures.json"
        )
    checkpoints = list(run.glob("*/round-*/model/value.pt"))
    checkpoint = (
        max(checkpoints, key=lambda p: p.stat().st_mtime)
        if checkpoints
        else run / "value.pt"
    )
    report = read(checkpoint.parent / "training.json")
    trained = report.get("queries") if report else None
    result = {
        "updated_at": datetime.now(TZ).isoformat(timespec="seconds"),
        "monitor_pid": os.getpid(),
        "monitor_status": "monitoring",
        "pipeline_status": state,
        "phase": status.get("phase"),
        "round": status.get("round"),
        "global_round": status.get("global_round"),
        "total_rounds": status.get("total_rounds"),
        "query_order": status.get("query_order"),
        "controller_pid": status.get("pid"),
        "controller_alive": live,
        "completed_queries": status.get("completed_queries", 0),
        "completed_cold_queries": status.get("completed_queries", 0),
        "completed_warm_queries": status.get("completed_warm_queries", 0),
        "total_queries": status.get("total_queries"),
        "records": status.get("records", 0),
        "failed_queries": status.get("failed_queries", 0),
        "current_query": query,
        "current_dataset": status.get("current_dataset"),
        "trained_feedback_queries": trained,
        "seconds_without_artifact_progress": round(inactive),
        "checkpoint": str(checkpoint) if checkpoint.exists() else None,
        "checkpoint_updated_at": datetime.fromtimestamp(
            checkpoint.stat().st_mtime, TZ
        ).isoformat(timespec="seconds")
        if checkpoint.exists()
        else None,
        "model_services": health,
        "training_metrics": training_summary(report),
        "alerts": alerts,
    }
    if state in ("complete", "complete_with_failures"):
        result["monitor_status"] = "finished"
    return result


def main():
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root / "SemCache/optimizer"))
    sys.path.insert(0, str(project_root / "experiments/semcache"))
    from planning.settings.configuration import config_from_argv

    config_path, settings = config_from_argv()
    monitoring = settings["monitoring"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=config_path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=monitoring["refresh_seconds"])
    parser.add_argument(
        "--health-interval", type=float, default=monitoring["health_interval"]
    )
    parser.add_argument(
        "--stalled-after", type=float, default=monitoring["stalled_after"]
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--reports", action="store_true", help="Refresh progress and loss Markdown"
    )
    parser.add_argument(
        "--stop-query-after",
        type=float,
        default=monitoring["stop_query_after"],
        help="Opt-in: interrupt estimated collection if one query exceeds these seconds; 0 disables",
    )
    args = parser.parse_args()
    if min(args.interval, args.health_interval, args.stalled_after) <= 0:
        parser.error("Monitoring intervals must be positive")
    if args.stop_query_after < 0:
        parser.error("Slow-query stop interval cannot be negative")
    checked_reports = {}
    watch = SlowQueryWatch(args.stop_query_after)
    stopped = False
    run = args.run.resolve()
    with (run / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (run / "monitor.pid").write_text(str(os.getpid()) + "\n")
        health, last_event, last_health = {}, None, -float("inf")
        while True:
            status = read(run / "pipeline-status.json")
            if status is None:
                print("Waiting for pipeline-status.json", flush=True)
                if args.once:
                    raise FileNotFoundError(run / "pipeline-status.json")
                time.sleep(args.interval)
                continue
            if time.monotonic() - last_health >= args.health_interval:
                models = read(run / "models.json") or {}
                with ThreadPoolExecutor(max_workers=3) as pool:
                    health = dict(pool.map(healthcheck, models.items()))
                last_health = time.monotonic()
            # Network health checks may outlast the observed query.
            status = read(run / "pipeline-status.json")
            if not isinstance(status, dict):
                continue
            elapsed = watch.check(status, time.monotonic())
            if elapsed is not None and not stopped:
                stopped = stop_slow_query(run, status, elapsed)["signal_sent"]
                if stopped:
                    print(
                        "Slow query interrupted; evidence saved in slow-query-diagnostics",
                        flush=True,
                    )
            current = snapshot(run, status, health, args.stalled_after)
            started = watch.started
            current["query_elapsed_seconds"] = (
                round(time.monotonic() - started) if started is not None else None
            )
            current["query_stop_after_seconds"] = args.stop_query_after or None
            if stopped:
                current["alerts"].append(
                    "Slow query stop requested; inspect slow-query-stop.json"
                )
                if not current["controller_alive"]:
                    current["monitor_status"] = "stopped_slow_query"
            save(run / "monitor-status.json", current)
            if args.reports:
                from .progress import refresh_reports

                checked_reports = refresh_reports(run, checked_reports)
            event = {
                key: current[key]
                for key in (
                    "pipeline_status",
                    "phase",
                    "completed_queries",
                    "total_queries",
                    "records",
                    "failed_queries",
                    "current_query",
                    "current_dataset",
                    "trained_feedback_queries",
                    "checkpoint_updated_at",
                    "alerts",
                )
            }
            event["model_health"] = {
                name: item["healthy"] for name, item in health.items()
            }
            if event != last_event:
                event["updated_at"] = current["updated_at"]
                with (run / "monitor-events.jsonl").open("a") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                print(json.dumps(event, ensure_ascii=False), flush=True)
                last_event = {
                    key: value for key, value in event.items() if key != "updated_at"
                }
            if args.once or current["monitor_status"] in (
                "finished",
                "stopped_slow_query",
            ):
                return
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
