"""One artifact layout shared by cache data collection and MLP fitting."""

from pathlib import Path

GROUPS = {
    "data": {
        "observations.sqlite",
        "operator_samples.jsonl",
        "operator_samples.csv",
        "数据收集.md",
    },
    "training": {"mlp-fit.json", "training-history.json", "collector.lock"},
    "diagnostics": {
        "capacity-comparison.json",
        "error-analysis.json",
        "optimization-result.json",
        "predictions.csv",
        "tree-diagnostic.json",
        "verification.json",
    },
}


def artifact_path(root, filename):
    root = Path(root)
    if filename.startswith("baseline/"):
        return root / "archive" / filename
    base = filename.removesuffix(".tmp")
    group = next((group for group, names in GROUPS.items() if base in names), None)
    return root / group / filename if group else root / filename


def prepare_layout(root):
    for group in (*GROUPS, "archive"):
        (Path(root) / group).mkdir(parents=True, exist_ok=True)
