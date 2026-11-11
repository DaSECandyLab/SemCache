"""Workspace paths shared by experiment tools; independent of cwd."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QUERY_ROOT = ROOT / "benchmarks/queries"
TRANSLATED = QUERY_ROOT / "translated"


def experiment_sources(root=ROOT):
    """Enumerate executable helpers recursively, excluding tests and artifacts."""
    return sorted(
        path
        for package in ("core", "system_comparison", "validation")
        for path in (root / "experiments" / package).rglob("*.py")
    )
