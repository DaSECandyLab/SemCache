"""Train a value model from repeated query execution feedback, without pretraining."""

from .cli import main as main
from .preparation import (
    initial_checkpoint as initial_checkpoint,
)
from .preparation import (
    now as now,
)
from .preparation import (
    prepare_feedback_run,
)
from .rounds import run_feedback_rounds


def run(*args, **kwargs):
    ctx = prepare_feedback_run(*args, **kwargs)
    return run_feedback_rounds(ctx)
