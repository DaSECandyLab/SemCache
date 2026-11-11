"""Publish changed reports atomically and coordinate their single writer."""

import fcntl
import os
from contextlib import contextmanager


def publish(path, content):
    if path.exists() and path.read_text() == content:
        return path
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content)
    temporary.replace(path)
    return path


@contextmanager
def report_writer(run, *, blocking=False):
    with (run / "reports.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
        else:
            yield True
