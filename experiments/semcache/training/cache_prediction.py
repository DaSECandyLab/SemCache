"""Fit operator mean prefix-match lengths from saved features and public usage."""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import signal
import sqlite3
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from planning.execution.cache_features import operator_context, semantic_context
from planning.execution.cache_prefix import prefix_context

from training.artifacts import read, save
from training.cache_prediction_paths import artifact_path, prepare_layout

# Versions 16 and 18 share the same 105 numeric values and cache encoding;
# their differences concern the value network's remaining-operator encoder.
CACHE_FEATURE_VERSIONS = {16, 18}


class SkippedObservation(ValueError):
    """An unexecuted timeout has no observed cache-token label."""


def episode_telemetry(leaf, source):
    """Merged episodes own their copied execution, not the old absolute path."""
    source = Path(source)
    if "sql" not in source.parts:
        raise ValueError("Cache sidecar requires a SQL execution artifact")
    local = (
        leaf.joinpath(*source.parts[source.parts.index("sql") :]) / "telemetry.jsonl"
    )
    return local if local.is_file() else source / "telemetry.jsonl"


def collection_status(run):
    path = run / "pipeline-status.json"
    return read(path if path.exists() else run / "status.json")


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def decode(value, valid=True):
    if not valid or not number(value) or value > 700:
        return None
    return math.expm1(value)


def reuse_features(node):
    """Recover the original numeric features from the frozen model input.

    Retain all existing reductions so a later study need not rerun queries.
    Empty previous-source sets represent known zero reuse. Missing values stay
    unknown and cannot become a misleading zero training label.
    """
    numeric, history = node["numeric"], node.get("l_existing")
    previous = {
        str(source): decode(value)
        for source, value in node.get("l_previous", {}).items()
    }
    previous_valid = all(v is not None for v in previous.values())
    values = list(previous.values()) if previous_valid else []
    existing = isinstance(history, list) and len(history) == 18
    return {
        "l_self": decode(numeric[-3], numeric[-1] == 1),
        "l_previous": max(values, default=0.0) if previous_valid else None,
        "l_existing": decode(history[1], history[13] == 1) if existing else None,
        "l_previous_by_source": previous,
        "l_previous_sum": math.fsum(values) if previous_valid else None,
        "l_existing_mean": decode(history[0], history[12] == 1) if existing else None,
        "l_existing_sources": round(decode(history[2], history[14] == 1))
        if existing and history[14] == 1
        else None,
        "l_self_reliability": numeric[-2],
        "encoded_cache_features": {
            "self": numeric[-3:],
            "previous": node.get("l_previous", {}),
            "existing": history,
        },
        **operator_context(node),
    }


def extract_episode(leaf, run):
    payload = read(leaf / "experience.json")
    if payload.get("feature_version") not in CACHE_FEATURE_VERSIONS:
        raise ValueError("Cache feature layout changed; update the explicit decoder")
    (record,) = payload["records"]
    query = read(leaf / "query_ir.json")
    features = record["trajectory"][-1]["features"]["nodes"]
    actions = {a["operator_id"]: a for a in record["plan"]["actions"]}
    sources = record["metadata"]["timing_artifacts"]
    if len(sources) != 1:
        raise ValueError("Cache sidecar requires one physical execution per episode")
    telemetry = episode_telemetry(leaf, sources[0])
    if not telemetry.is_file() and record["metadata"].get("execution_skipped"):
        raise SkippedObservation("execution_skipped_without_request_telemetry")
    # Only LF separates JSONL records. Unicode line-separator characters can
    # occur inside quoted input data and must not split one response in two.
    with telemetry.open() as stream:
        identity, rows, requests = extract_observations(
            stream,
            leaf,
            run,
            record,
            query,
            features,
            actions,
            payload["feature_version"],
        )
    profile_path = leaf / "profile.json"
    profile = read(profile_path) if profile_path.exists() else {}
    for row in rows:
        op = next(
            op
            for op in query["semantic_operators"]
            if op["operator_id"] == row["operator_id"]
        )
        row.update(semantic_context(op, actions[row["operator_id"]], profile))
        row.update(
            prefix_context(
                features[str(op["node_id"])],
                op,
                actions[row["operator_id"]],
                profile,
                row,
            )
        )
    return identity, rows, requests


def extract_observations(
    stream, leaf, run, record, query, features, actions, feature_version
):
    counters, requests = {}, []
    for line in stream:
        if not line.strip():
            continue
        event = json.loads(line)
        # Cache-answer hits/reuses have no public usage of a new HTTP response.
        if (
            event.get("event") != "semantic_row"
            or event.get("parse_status") != "validated"
        ):
            continue
        oid = event["operator_id"]
        action = actions[oid]
        if (
            event.get("model_id") != action["model_id"]
            or event.get("config_generation", 0) != 0
        ):
            raise ValueError(
                "Static training observation differs from frozen model/generation"
            )
        for name, order in action["layout"].items():
            if event.get(name) != order:
                raise ValueError(
                    "Static training response differs from frozen prompt layout"
                )
        bank = counters.setdefault(oid, Counter())
        bank["responses"] += 1
        tokens, prompt = event.get("cached_input_tokens"), event.get("input_tokens")
        usage_valid = number(tokens) and number(prompt) and tokens <= prompt
        if usage_valid:
            bank["usage_responses"] += 1
            bank["cached_tokens"] += tokens
            bank["prompt_tokens"] += prompt
        if number(event.get("latency_ms")):
            bank["latency_ms"] += event["latency_ms"]
        requests.append(
            {
                "operator_id": oid,
                "request_seq": event["request_seq"],
                "cached_tokens": tokens if usage_valid else None,
                "prompt_tokens": prompt if number(prompt) else None,
                "latency_ms": event.get("latency_ms"),
                "usage_known": usage_valid,
                "parse_status": event.get("parse_status"),
            }
        )
    relative = str(leaf.relative_to(run))
    episode_id = hashlib.sha256(relative.encode()).hexdigest()
    rows = []
    for op in query["semantic_operators"]:
        oid = op["operator_id"]
        node = features.get(str(op["node_id"]))
        if node is None or len(node["numeric"]) != 105:
            raise ValueError("Frozen final state has no semantic node features")
        stats = counters.get(oid, Counter())
        count, covered = stats["responses"], stats["usage_responses"]
        known = covered == count and count > 0
        f = reuse_features(node)
        row = {
            "sample_id": f"{episode_id}:{oid}",
            "episode": relative,
            "source_feature_version": feature_version,
            "round": int(leaf.parents[2].name.removeprefix("round-")),
            "query": record["metadata"]["workload"],
            "dataset": record["dataset"],
            "search": int(leaf.name.removeprefix("search-")),
            "operator_id": oid,
            "operator_kind": op["kind"],
            "model_id": actions[oid]["model_id"],
            "layout": actions[oid]["layout"],
            "query_status": record["status"],
            **f,
            "http_response_observations": count,
            "usage_known_responses": covered,
            "usage_coverage": covered / count if count else None,
            "cached_tokens_sum": stats["cached_tokens"] if known else None,
            "mean_cached_tokens": stats["cached_tokens"] / count if known else None,
            "prompt_tokens_sum": stats["prompt_tokens"] if known else None,
            "training_eligible": known
            and record["status"] == "completed"
            and all(
                f[name] is not None for name in ("l_self", "l_previous", "l_existing")
            ),
            "feature_scope": "frozen pre-execution operator mean prefix lengths in characters",
            "label_scope": "mean public cached tokens over consumed HTTP responses",
            "chunk_features_available": False,
        }
        rows.append(row)
    ids = {r["operator_id"]: r["sample_id"] for r in rows}
    for request in requests:
        request["sample_id"] = ids[request["operator_id"]]
    return episode_id, rows, requests


def run_sidecar(args):
    run = args.run.resolve()
    output = args.output.resolve() if args.output else run / "cache_prediction"
    output.mkdir(parents=True, exist_ok=True)
    prepare_layout(output)
    guard = (artifact_path(output, "collector.lock")).open("a")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    db = sqlite3.connect(artifact_path(output, "observations.sqlite"))
    db.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS episodes (id TEXT PRIMARY KEY, source TEXT, collected TEXT);
        CREATE TABLE IF NOT EXISTS operators (id TEXT PRIMARY KEY, episode TEXT, payload TEXT);
        CREATE TABLE IF NOT EXISTS requests (sample TEXT, sequence INTEGER, payload TEXT, PRIMARY KEY(sample,sequence));
        CREATE TABLE IF NOT EXISTS errors (source TEXT PRIMARY KEY, message TEXT);
        CREATE TABLE IF NOT EXISTS skipped (source TEXT PRIMARY KEY, reason TEXT);
    """)
    saved_fit = (
        read(artifact_path(output, "mlp-fit.json"))
        if (artifact_path(output, "mlp-fit.json")).exists()
        else {}
    )
    fit_options = {
        key: saved_fit[key]
        for key in (
            "feature_mode",
            "hidden_sizes",
            "target_transform",
            "loss_kind",
            "scale_floor",
            "huber_beta",
        )
        if key in saved_fit
    }
    if "architecture" in saved_fit:
        fit_options["hidden_sizes"] = saved_fit["architecture"][1:-1]
    architecture = saved_fit.get("architecture", [128, 64, 32, 1])
    network_name = " → ".join(map(str, architecture))
    config = {
        "model": "MLP",
        "architecture": architecture,
        "input_features": [
            "GAT operator numeric",
            "history numeric",
            "upstream summaries",
        ],
        "reductions": {
            "l_self": "saved per-request self-reuse mean",
            "l_previous": "max of saved previous-operator edge lengths",
            "l_existing": "saved strongest compatible history prefix length",
        },
        "all_original_cache_features_retained": True,
        "target": "operator mean matched prefix length per consumed HTTP response",
        "target_unit": "tokens",
        "api_field": "usage.prompt_tokens_details.cached_tokens",
        "missing_usage": "unknown, never substituted with zero",
        "fit_uses": "completed operators with full usage coverage and valid frozen features",
        "online_adaptation_enabled": False,
        "value_model_changed": False,
        "source_feature_versions": sorted(CACHE_FEATURE_VERSIONS),
        "query_execution_enabled": False,
        "excluded_training_rounds": sorted(args.exclude_training_rounds),
    }
    save(artifact_path(output, "config.json"), config)
    started = time.monotonic()
    fit_history = []
    save(
        artifact_path(output, "progress.json"),
        {"status": "extracting", "source": str(run), "episodes_scanned": 0},
    )

    def training_epoch(metrics):
        metrics = {**metrics, "elapsed_seconds": time.monotonic() - started}
        fit_history.append(metrics)
        save(artifact_path(output, "training-history.json"), fit_history)
        save(artifact_path(output, "progress.json"), {"status": "fitting", **metrics})
        lines = [
            "# 平均前缀匹配长度 MLP 训练\n",
            f"来源：`{run}`\n",
            "输入：算子 GAT 数值特征、历史特征和上游摘要，包含三个缓存统计量。目标：算子实际请求平均匹配前缀长度（token）。\n",
            f"网络：{network_name}；按查询划分训练集与验证集。\n",
            "| Epoch | 训练损失 | 验证 MAE(token) | 最佳验证 MAE(token) | 从开始耗时(s) |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
        lines.extend(
            f"| {m['epoch']} | {m['training_loss']:.6f} | {m['monitor_mae_tokens']:.6f} | {m['best_monitor_mae_tokens']:.6f} | {m['elapsed_seconds']:.2f} |"
            for m in fit_history
        )
        temporary = artifact_path(output, "模型训练.md.tmp")
        temporary.write_text("\n".join(lines) + "\n")
        temporary.replace(artifact_path(output, "模型训练.md"))

    manifest = run / "workloads.json"
    workloads = (
        {w["id"] for w in read(manifest)["workloads"]} if manifest.exists() else None
    )
    last_fit_samples = 0
    if (artifact_path(output, "mlp-fit.json")).exists():
        last_fit_samples = read(artifact_path(output, "mlp-fit.json")).get(
            "total_eligible_samples", 0
        )
    try:
        while not stop.is_set():
            new = 0
            markers = sorted(
                (run / "measured").glob("round-*/stream/*/search-*/collection.json")
            )
            markers = [
                m
                for m in markers
                if workloads is None or m.parent.parent.name in workloads
            ]
            for scanned, marker in enumerate(markers, 1):
                leaf = marker.parent
                relative = str(leaf.relative_to(run))
                if db.execute(
                    "SELECT 1 FROM episodes WHERE source=?", (relative,)
                ).fetchone():
                    continue
                if not (leaf / "experience.json").exists():
                    continue
                try:
                    episode, operators, requests = extract_episode(leaf, run)
                    with db:
                        for row in operators:
                            db.execute(
                                "INSERT OR REPLACE INTO operators VALUES(?,?,?)",
                                (
                                    row["sample_id"],
                                    episode,
                                    json.dumps(
                                        row, ensure_ascii=False, allow_nan=False
                                    ),
                                ),
                            )
                        for row in requests:
                            db.execute(
                                "INSERT OR REPLACE INTO requests VALUES(?,?,?)",
                                (
                                    row["sample_id"],
                                    row["request_seq"],
                                    json.dumps(row, allow_nan=False),
                                ),
                            )
                        db.execute(
                            "INSERT INTO episodes VALUES(?,?,?)",
                            (
                                episode,
                                relative,
                                datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                            ),
                        )
                        db.execute("DELETE FROM errors WHERE source=?", (relative,))
                        db.execute("DELETE FROM skipped WHERE source=?", (relative,))
                    new += 1
                except SkippedObservation as error:
                    with db:
                        db.execute(
                            "INSERT OR REPLACE INTO skipped VALUES(?,?)",
                            (relative, str(error)),
                        )
                        db.execute("DELETE FROM errors WHERE source=?", (relative,))
                except (ValueError, KeyError, OSError, TypeError) as error:
                    with db:
                        db.execute(
                            "INSERT OR REPLACE INTO errors VALUES(?,?)",
                            (relative, str(error)),
                        )
                if scanned % 5 == 0 or scanned == len(markers):
                    save(
                        artifact_path(output, "progress.json"),
                        {
                            "status": "extracting",
                            "episodes_scanned": scanned,
                            "episodes_total": len(markers),
                            "elapsed_seconds": time.monotonic() - started,
                            "extraction_errors": db.execute(
                                "SELECT count(*) FROM errors"
                            ).fetchone()[0],
                        },
                    )
            rows = [
                json.loads(v)
                for (v,) in db.execute(
                    "SELECT payload FROM operators ORDER BY episode,id"
                )
            ]
            eligible = [
                r
                for r in rows
                if r["training_eligible"]
                and r["round"] not in args.exclude_training_rounds
            ]
            if new or not (artifact_path(output, "operator_samples.jsonl")).exists():
                tmp = artifact_path(output, "operator_samples.jsonl.tmp")
                with tmp.open("w") as stream:
                    for row in rows:
                        stream.write(
                            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                        )
                tmp.replace(artifact_path(output, "operator_samples.jsonl"))
                columns = [
                    "sample_id",
                    "round",
                    "query",
                    "search",
                    "operator_id",
                    "model_id",
                    "operator_kind",
                    "l_self",
                    "l_previous",
                    "l_existing",
                    "mean_cached_tokens",
                    "cached_tokens_sum",
                    "http_response_observations",
                    "usage_coverage",
                    "training_eligible",
                ]
                tmp = artifact_path(output, "operator_samples.csv.tmp")
                with tmp.open("w", newline="") as stream:
                    writer = csv.DictWriter(
                        stream, fieldnames=columns, extrasaction="ignore"
                    )
                    writer.writeheader()
                    writer.writerows(rows)
                tmp.replace(artifact_path(output, "operator_samples.csv"))
            pipeline = collection_status(run)
            finished = pipeline["status"] in {"completed", "stopped"}
            if (
                args.fit
                and len(eligible) >= 16
                and (
                    len(eligible) >= last_fit_samples + 24
                    or (finished or args.once)
                    and len(eligible) > last_fit_samples
                )
            ):
                # Separate CPU process: this cannot modify the running trainer's
                # RNG, GPU memory, torch thread settings or model parameters.
                import torch
                from planning.execution.cache_prediction import fit_cache_predictor

                torch.set_num_threads(1)
                model = fit_cache_predictor(
                    eligible, on_epoch=training_epoch, **fit_options
                )
                model.save(artifact_path(output, "cache_tokens_mlp.pt"))
                last_fit_samples = len(eligible)
                save(
                    artifact_path(output, "mlp-fit.json"),
                    {**model.metadata, "total_eligible_samples": len(eligible)},
                )
            state = {
                "status": "completed" if finished or args.once else "collecting",
                "pid": os.getpid(),
                "episodes": db.execute("SELECT count(*) FROM episodes").fetchone()[0],
                "operator_samples": len(rows),
                "eligible_samples": len(eligible),
                "http_response_records": db.execute(
                    "SELECT count(*) FROM requests"
                ).fetchone()[0],
                "extraction_errors": db.execute(
                    "SELECT count(*) FROM errors"
                ).fetchone()[0],
                "skipped_unexecuted_plans": db.execute(
                    "SELECT count(*) FROM skipped"
                ).fetchone()[0],
                "training_status": pipeline["status"],
                "updated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "elapsed_seconds": time.monotonic() - started,
                "checkpoint": str(artifact_path(output, "cache_tokens_mlp.pt"))
                if (artifact_path(output, "cache_tokens_mlp.pt")).exists()
                else None,
                "model": network_name + " MLP",
                "value_model_untouched": True,
            }
            if (artifact_path(output, "mlp-fit.json")).exists():
                fit = read(artifact_path(output, "mlp-fit.json"))
                state.update(
                    epochs=fit["epochs"],
                    training_samples=fit["training_samples"],
                    validation_samples=fit["validation_samples"],
                    validation_mae_tokens=fit["mae_tokens"],
                    checkpoint_sha256=hashlib.sha256(
                        (artifact_path(output, "cache_tokens_mlp.pt")).read_bytes()
                    ).hexdigest(),
                )
                history_path = artifact_path(output, "training-history.json")
                if history_path.exists():
                    state["elapsed_seconds"] = max(
                        state["elapsed_seconds"],
                        read(history_path)[-1]["elapsed_seconds"],
                    )
                document = artifact_path(output, "模型训练.md")
                if document.exists():
                    text = document.read_text().split("\n## 最终结果", 1)[0]
                    text += (
                        "\n## 最终结果\n\n"
                        f"状态：已完成；训练 {fit['epochs']} epoch。检查点：`cache_tokens_mlp.pt`。\n\n"
                        f"有效算子样本 {len(eligible)}；训练 {fit['training_samples']}；验证 {fit['validation_samples']}。"
                        f"验证 log-MSE：{fit['log_mse']:.6f}；验证平均绝对误差：{fit['mae_tokens']:.4f} token。\n\n"
                        f"已提取 {state['episodes']} 条计划；排除 {state['skipped_unexecuted_plans']} 条未执行且没有请求遥测的超时记录；"
                        f"提取错误 {state['extraction_errors']}。缺失 usage、未完整执行或缺失特征的算子不参与拟合。\n"
                    )
                    temporary = artifact_path(output, "模型训练.md.tmp")
                    temporary.write_text(text)
                    temporary.replace(document)
            save(artifact_path(output, "progress.json"), state)
            (artifact_path(output, "数据收集.md")).write_text(
                "# 缓存token预测：数据与独立MLP\n\n"
                f"状态：{state['status']}；执行记录{state['episodes']}条；算子样本{len(rows)}条；可拟合{len(eligible)}条；HTTP响应{state['http_response_records']}条；提取错误{state['extraction_errors']}条。\n\n"
                "MLP：GAT算子数值与缓存特征 → 64 → 32 → 平均前缀匹配长度。仅CPU独立拟合，不改value model、不发送LLM请求。\n\n"
                "每个样本对应一次计划执行中的一个语义算子。输入为已冻结的L_self、L_previous、L_existing三个统计量，标签为该算子实际HTTP响应的平均前缀匹配长度（token），取API公开usage中的cached_tokens总量除以响应数，不是整份计划的总量。当前静态实验没有chunk级特征。\n\n"
                "L_previous取前序算子匹配长度的最大值，L_existing取历史中最强匹配长度；原始逐来源数据和mean/sum等口径也保留。缺失usage或特征保持未知，不补零。超时记录保留，但不用于当前拟合。\n\n"
                "[算子样本CSV](data/operator_samples.csv) · [完整特征JSONL](data/operator_samples.jsonl) · [逐请求及算子数据库](data/observations.sqlite)\n"
            )
            if args.once or finished:
                break
            stop.wait(args.interval)
    finally:
        db.close()
        guard.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--exclude-training-rounds", type=int, nargs="+", default=())
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval must be at least one second")
    if any(number < 1 for number in args.exclude_training_rounds):
        parser.error("--exclude-training-rounds must contain positive round numbers")
    run_sidecar(args)


if __name__ == "__main__":
    main()
