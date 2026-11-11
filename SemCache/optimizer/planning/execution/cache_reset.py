"""Clear reusable prompt answers between query clients at a round boundary."""

import fcntl
import os
import re
from pathlib import Path

from planning.settings.configuration import response_cache_settings


def clear_prompt_answers(*, directory=None):
    """No client may remain active: its L1 is released with the HTTP client.

    Keep shard lock inodes intact and remove only files owned by the native
    prompt cache. Other offline statistics and saved answers are not here.
    """
    directory = response_cache_settings()["directory"] if directory is None else str(directory)
    result = {
        "status": "completed",
        "directory": directory,
        "entries_removed": 0,
        "temporary_files_removed": 0,
        "allocated_bytes_removed": 0,
    }
    if directory is None or not Path(directory).exists():
        return result
    with os.scandir(directory) as entries:
        shards = sorted(
            Path(entry.path)
            for entry in entries
            if entry.name.isascii()
            and entry.name.isdecimal()
            and entry.is_dir(follow_symlinks=False)
        )
    for shard in shards:
        descriptor = os.open(
            shard / ".lock",
            os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(descriptor, "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with os.scandir(shard) as entries:
                for entry in entries:
                    temporary = entry.name == ".tmp"
                    if (
                        not temporary
                        and re.fullmatch(r"[0-9a-f]{64}\.entry", entry.name) is None
                    ):
                        continue
                    info = entry.stat(follow_symlinks=False)
                    os.unlink(entry.path)
                    key = "temporary_files_removed" if temporary else "entries_removed"
                    result[key] += 1
                    result["allocated_bytes_removed"] += info.st_blocks * 512
    return result
