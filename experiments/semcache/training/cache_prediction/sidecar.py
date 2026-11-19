"""Fit operator mean prefix-match lengths from saved features and public usage."""

import argparse
import csv
import fcntl
import hashlib
import json
import os
import signal
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from training.artifacts import read, save
from training.cache_prediction.paths import artifact_path, prepare_layout

from .extraction import (
    SkippedObservation,
    collection_status,
    extract_episode,
)

# Versions 16 and 18 share the same 105 numeric values and cache encoding;
# their differences concern the value network's remaining-operator encoder.
CACHE_FEATURE_VERSIONS = {16, 18}


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
                from planning.execution.cache.prediction import fit_cache_predictor

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
