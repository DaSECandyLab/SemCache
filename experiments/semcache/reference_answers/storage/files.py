"""Content hashes for frozen source and answer artifacts."""

import hashlib
from pathlib import Path


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
