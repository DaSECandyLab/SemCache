"""Read in-flight SQL execution telemetry and physical model request counts."""

import json
import time
from functools import lru_cache
from pathlib import Path

from training.artifacts import read

MODELS = ("gemma4-e2b", "gemma4-e4b", "gemma4-12b")


class TelemetryCounts:
    """Consume only appended complete lines; retries are physical sends too."""

    def __init__(self):
        self.identity = None
        self.offset = 0
        self.sent = dict.fromkeys(MODELS, 0)
        self.completed = 0

    def read(self, path):
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity != self.identity or stat.st_size < self.offset:
            self.__init__()
            self.identity = identity
        with path.open("rb") as stream:
            stream.seek(self.offset)
            while line := stream.readline():
                if not line.endswith(b"\n"):
                    break
                self.offset = stream.tell()
                if not line.strip():
                    continue
                event = json.loads(line)
                kind = event.get("event")
                model = event.get("model_id")
                if kind in ("sampling_llm_submit", "sampling_llm_retransmit"):
                    if model in self.sent:
                        self.sent[model] += 1
                elif kind == "semantic_row":
                    self.completed += 1
        return dict(self.sent), self.completed, stat.st_mtime


@lru_cache(maxsize=8)
def _telemetry(path):
    return TelemetryCounts()


def live_execution_row(output, state):
    if (
        state.get("status") != "running"
        or state.get("phase") != "measured"
        or not state.get("current_query")
    ):
        return None
    leaf = (
        Path(output)
        / f"measured/round-{state['round']}/stream"
        / state["current_query"]
        / f"search-{state.get('rollout_index', 1)}"
    )
    if not leaf.exists() or (leaf / "experience.json").exists():
        return None
    selected_path = leaf / "selected-plan.json"
    selected = read(selected_path) if selected_path.exists() else {}
    selection = selected.get("rollout_selection", {})
    scores = selection.get("predicted_scores")
    timing_path = leaf / "timing.json"
    timing = read(timing_path) if timing_path.exists() else {}
    beam = timing.get("phases", {}).get("beam_search", {}).get("wall_seconds")
    sql = leaf / "sql/run-2/query.sql"
    telemetry = sql.parent / "telemetry.jsonl"
    profile = sql.parent / "profile.json"
    status = "searching"
    elapsed = age = None
    counts = dict.fromkeys(MODELS)
    completed = 0
    if sql.exists():
        status = "finalizing" if profile.exists() else "running"
        end = profile.stat().st_mtime if profile.exists() else time.time()
        elapsed = max(0.0, end - sql.stat().st_mtime)
        if telemetry.exists():
            counts, completed, updated = _telemetry(str(telemetry)).read(telemetry)
            age = max(0.0, time.time() - updated)
    elif selected:
        status = "preparing"
    return {
        "round": state["round"],
        "query": state["current_query"],
        "execution": state.get("rollout_index", 1),
        "selection": selection.get("policy", state.get("selection_policy", "—")),
        "minimum_accuracy": selection.get(
            "minimum_accuracy", state.get("minimum_accuracy")
        ),
        "duplicate_replaced": selection.get("duplicate_replaced", False),
        "replacement_scope": selection.get("replacement_scope"),
        "accuracy_fallback": selection.get("accuracy_fallback", False),
        "status": status,
        "artifact": str(leaf),
        "beam_seconds": beam,
        "predicted_seconds": -scores[0] if scores is not None else None,
        "predicted_accuracy": scores[1] if scores is not None else None,
        "predicted_scores": scores,
        "reward": {"latency_seconds": None, "accuracy": None, "complete": False},
        "elapsed_seconds": elapsed,
        "telemetry_age_seconds": age,
        "completed_responses": completed,
        "model_http_requests": {
            model: {"sampling": None, "execution": counts[model]} for model in MODELS
        },
    }


def live_execution_text(row):
    if row is None:
        return None
    query = row["query"].removeprefix("translated/semcache/")
    label = {
        "searching": "采样与搜索中",
        "preparing": "准备执行中",
        "running": "执行中",
        "finalizing": "保存结果中",
    }[row["status"]]
    minimum = row.get("minimum_accuracy")
    target = "" if minimum is None else f"，准确率下限{minimum:.0%}"
    pieces = [f"当前执行：{query}，第 {row['execution']} 次{target}，{label}"]
    if row["elapsed_seconds"] is not None:
        pieces.append(f"已执行约 {row['elapsed_seconds']:.0f} 秒")
        counts = [row["model_http_requests"][m]["execution"] for m in MODELS]
        pieces.append(
            "HTTP请求 E2B/E4B/12B："
            + "/".join("—" if n is None else str(n) for n in counts)
        )
        if row["telemetry_age_seconds"] is not None:
            pieces.append(f"执行日志 {row['telemetry_age_seconds']:.0f} 秒前更新")
    return "；".join(pieces) + "。"
