"""Read JSON artifacts and publish complete files through atomic replacement."""

import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n"
    )
    temporary.replace(path)
