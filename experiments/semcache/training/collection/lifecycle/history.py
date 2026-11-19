"""Store audit-only cache history once; model features stay in each trajectory."""

import json
from pathlib import Path

from planning.state import digest


def save_history_context(output, history):
    identity = digest(history)
    directory = Path(output).resolve() / "cache-history"
    directory.mkdir(exist_ok=True)
    artifact = directory / (identity + ".json")
    if not artifact.exists():
        temporary = artifact.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(history, ensure_ascii=False, allow_nan=False) + "\n"
        )
        temporary.replace(artifact)
    return {"sha256": identity, "entries": len(history), "artifact": str(artifact)}
