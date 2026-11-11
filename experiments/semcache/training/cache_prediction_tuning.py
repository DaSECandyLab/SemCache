"""Select cache predictor capacity on inner query folds, preserving outer queries."""

import argparse
import csv
import hashlib
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import torch
from planning.execution.cache_features import operator_context, semantic_context
from planning.execution.cache_fit import fit_cache_predictor, validation_queries
from planning.execution.cache_prediction import CacheTokenMLP
from planning.execution.cache_prefix import prefix_context

from training.artifacts import read, save
from training.cache_prediction_paths import artifact_path, prepare_layout

CANDIDATES = tuple(
    {
        "feature_mode": "operator_tokens",
        "hidden_sizes": [64, 32],
        "target_transform": "log1p",
        "loss_kind": "huber",
        **variant,
    }
    for variant in (
        {"id": "tokens-huber-current"},
        {"id": "tokens-huber-01", "huber_beta": 0.1},
        {
            "id": "tokens-absolute-linear",
            "target_transform": "linear",
            "loss_kind": "absolute",
        },
        {"id": "tokens-huber-scale01", "scale_floor": 0.1},
        {"id": "tokens-huber01-scale01", "huber_beta": 0.1, "scale_floor": 0.1},
        {"id": "tokens-huber01-scale001", "huber_beta": 0.1, "scale_floor": 0.01},
        {
            "id": "tokens-large-huber01-scale01",
            "hidden_sizes": [128, 64],
            "huber_beta": 0.1,
            "scale_floor": 0.1,
        },
        {
            "id": "tokens-large-huber005-scale01",
            "hidden_sizes": [128, 64],
            "huber_beta": 0.05,
            "scale_floor": 0.1,
        },
    )
)


def augment_rows(run, output):
    rows = [
        json.loads(line)
        for line in (artifact_path(output, "operator_samples.jsonl")).open()
    ]
    episodes = defaultdict(list)
    for row in rows:
        episodes[row["episode"]].append(row)
    for scanned, (episode, operators) in enumerate(episodes.items(), 1):
        leaf = run / episode
        (record,) = read(leaf / "experience.json")["records"]
        nodes = record["trajectory"][-1]["features"]["nodes"]
        query = read(leaf / "query_ir.json")
        profile = read(leaf / "profile.json")
        actions = {a["operator_id"]: a for a in record["plan"]["actions"]}
        ops = {op["operator_id"]: op for op in query["semantic_operators"]}
        ids = {
            op["operator_id"]: str(op["node_id"]) for op in query["semantic_operators"]
        }
        for row in operators:
            row.update(operator_context(nodes[ids[row["operator_id"]]]))
            row.update(
                semantic_context(
                    ops[row["operator_id"]], actions[row["operator_id"]], profile
                )
            )
            row.update(
                prefix_context(
                    nodes[ids[row["operator_id"]]],
                    ops[row["operator_id"]],
                    actions[row["operator_id"]],
                    profile,
                    row,
                )
            )
        if scanned % 10 == 0:
            save(
                artifact_path(output, "progress.json"),
                {
                    "status": "refreshing_features",
                    "episodes_scanned": scanned,
                    "episodes_total": len(episodes),
                },
            )
    with sqlite3.connect(artifact_path(output, "observations.sqlite")) as db:
        for row in rows:
            db.execute(
                "UPDATE operators SET payload=? WHERE id=?",
                (
                    json.dumps(row, ensure_ascii=False, allow_nan=False),
                    row["sample_id"],
                ),
            )
    temporary = artifact_path(output, "operator_samples.jsonl.tmp")
    with temporary.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(artifact_path(output, "operator_samples.jsonl"))
    return [row for row in rows if row["training_eligible"]]


def run_tuning(run):
    output = run / "cache_prediction"
    prepare_layout(output)
    torch.set_num_threads(1)
    started = time.monotonic()
    rows = augment_rows(run, output)
    outer = validation_queries(rows)
    development = [r for r in rows if r["query"] not in outer]
    groups = sorted(
        {r["query"] for r in development},
        key=lambda q: hashlib.sha256(f"cache-inner:{q}".encode()).digest(),
    )
    folds = [set(groups[i::3]) for i in range(3)]
    comparisons = []
    for candidate in CANDIDATES:
        scores = []
        for fold, heldout in enumerate(folds):
            save(
                artifact_path(output, "progress.json"),
                {
                    "status": "inner_validation",
                    "candidate": candidate["id"],
                    "fold": fold + 1,
                    "folds": 3,
                    "completed_candidates": len(comparisons),
                    "candidates": len(CANDIDATES),
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
            config = {k: v for k, v in candidate.items() if k != "id"}
            config.setdefault("target_transform", "linear")
            model = fit_cache_predictor(
                development, **config, holdout_queries=heldout, epochs=1200
            )
            scores.append(model.metadata["mae_tokens"])
        result = {
            **candidate,
            "inner_fold_mae_tokens": scores,
            "inner_mean_mae_tokens": sum(scores) / len(scores),
        }
        comparisons.append(result)
        save(artifact_path(output, "capacity-comparison.json"), comparisons)
        print(json.dumps(result), flush=True)
    selected = min(comparisons, key=lambda c: c["inner_mean_mae_tokens"])
    config = {
        k: selected[k]
        for k in (
            "feature_mode",
            "hidden_sizes",
            "target_transform",
            "loss_kind",
            "scale_floor",
            "huber_beta",
        )
        if k in selected
    }
    history = []
    config.setdefault("target_transform", "linear")

    def on_epoch(metrics):
        history.append({**metrics, "elapsed_seconds": time.monotonic() - started})
        if metrics["epoch"] % 10 == 0:
            save(
                artifact_path(output, "progress.json"),
                {"status": "final_fitting", **history[-1]},
            )
            save(artifact_path(output, "training-history.json"), history)

    model = fit_cache_predictor(
        rows, **config, on_epoch=on_epoch, holdout_queries=outer
    )
    model.metadata.update(
        total_eligible_samples=len(rows),
        selection_protocol="three inner query folds within training queries; outer validation excluded from capacity selection",
        inner_cv_mae_tokens=selected["inner_mean_mae_tokens"],
    )
    model.save(artifact_path(output, "cache_tokens_mlp.pt"))
    save(artifact_path(output, "mlp-fit.json"), model.metadata)
    save(artifact_path(output, "training-history.json"), history)
    reloaded = CacheTokenMLP.load(artifact_path(output, "cache_tokens_mlp.pt"))
    predicted = reloaded.predict(rows)
    actuals = [r for r in rows if r["query"] in outer]
    outer_predictions = [p for p, row in zip(predicted, rows) if row["query"] in outer]
    checked_mae = sum(
        abs(p - r["mean_cached_tokens"]) for p, r in zip(outer_predictions, actuals)
    ) / len(actuals)
    assert abs(checked_mae - model.metadata["mae_tokens"]) < 1e-4
    baseline = read(artifact_path(output, "baseline/mlp-fit.json"))
    comparison = {
        "old_mae_tokens": baseline["mae_tokens"],
        "new_mae_tokens": checked_mae,
        "improvement_fraction": 1 - checked_mae / baseline["mae_tokens"],
        "selected": selected,
        "same_outer_validation_queries": sorted(outer)
        == baseline["validation_queries"],
        "validation_samples": len(actuals),
        "wape": model.metadata["wape"],
        "source": str(run),
        "saved_checkpoint_reload_verified": True,
    }
    save(artifact_path(output, "optimization-result.json"), comparison)
    with (artifact_path(output, "predictions.csv")).open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "sample_id",
                "query",
                "split",
                "actual_mean_prefix_tokens",
                "predicted_mean_prefix_tokens",
                "absolute_error",
            ]
        )
        for row, prediction in zip(rows, predicted):
            writer.writerow(
                [
                    row["sample_id"],
                    row["query"],
                    "validation" if row["query"] in outer else "training",
                    row["mean_cached_tokens"],
                    prediction,
                    abs(prediction - row["mean_cached_tokens"]),
                ]
            )
    header = [
        "# 平均前缀匹配长度 MLP 训练",
        "",
        f"数据来源：`{run}`",
        "",
        f"状态：已完成；执行 {model.metadata['epochs']} epoch；最佳检查点 epoch {model.metadata['best_epoch']}。",
        f"网络：{' → '.join(map(str, model.metadata['architecture']))}；参数 {sum(p.numel() for p in model.parameters())}。",
        "",
        "输入包含 GAT 算子的类型、模型、layout、静态前缀长度、输入基数、宽度、选择率、校准准确率及耗时、可靠性和有效性掩码，以及缓存来源统计。全部来自执行前冻结特征。",
        "另加入实际语义参数的长度、重复率、频率分布，以及本地 tokenizer 计算的静态 prompt token 长度和复用比例；不调用推理服务。文本哈希输入未被最终方案选用。",
        "目标：每个算子真实请求的平均前缀匹配长度（token）。不使用待预测算子的实际 usage、实际耗时或实际输出作为输入。",
        "",
        f"训练 {model.metadata['training_samples']} 个样本；验证 {model.metadata['validation_samples']} 个样本，按查询隔离。容量选择仅用训练查询内部三折，外层验证集保持原划分。",
        f"验证 MAE：{baseline['mae_tokens']:.4f} → {checked_mae:.4f} token；降低 {comparison['improvement_fraction']:.2%}；WAPE {model.metadata['wape']:.2%}。",
        "",
        "| 配置 | 训练查询内部三折平均 MAE(token) |",
        "| --- | ---: |",
    ]
    header.extend(
        f"| {c['id']} | {c['inner_mean_mae_tokens']:.4f} |" for c in comparisons
    )
    header.extend(
        [
            "",
            "| Epoch | 训练损失 | 验证 MAE(token) | 最佳验证 MAE(token) | 从开始耗时(s) |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    header.extend(
        f"| {m['epoch']} | {m['training_loss']:.6f} | {m['monitor_mae_tokens']:.4f} | {m['best_monitor_mae_tokens']:.4f} | {m['elapsed_seconds']:.2f} |"
        for m in history
    )
    (artifact_path(output, "模型训练.md")).write_text("\n".join(header) + "\n")
    status = {
        "status": "completed",
        "epochs": model.metadata["epochs"],
        "best_epoch": model.metadata["best_epoch"],
        "architecture": model.metadata["architecture"],
        "eligible_samples": len(rows),
        "training_samples": model.metadata["training_samples"],
        "validation_samples": len(actuals),
        "validation_mae_tokens": checked_mae,
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint": str(artifact_path(output, "cache_tokens_mlp.pt")),
        "checkpoint_sha256": hashlib.sha256(
            (artifact_path(output, "cache_tokens_mlp.pt")).read_bytes()
        ).hexdigest(),
    }
    save(artifact_path(output, "progress.json"), status)
    save(artifact_path(output, "progress.json"), status)
    settings = read(artifact_path(output, "config.json"))
    settings.update(
        architecture=model.metadata["architecture"],
        input_features=[
            "GAT operator numeric",
            "history numeric",
            "upstream summaries",
            "semantic argument distributions",
            "static token lengths and reuse interactions",
        ],
        feature_mode=model.feature_mode,
        target_transform=model.target_transform,
    )
    save(artifact_path(output, "config.json"), settings)
    print(json.dumps(comparison), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    run_tuning(parser.parse_args().run.resolve())
