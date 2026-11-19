"""Heldout selected text SQL evaluation after training; no labels enter replay."""

from .execution import run_evaluation
from .setup import (
    freeze_evaluation_contract as freeze_evaluation_contract,
)
from .setup import (
    original_query as original_query,
)
from .setup import (
    prepare_evaluation,
)


def evaluate(*args, **kwargs):
    return run_evaluation(prepare_evaluation(*args, **kwargs))
