"""Fit value heads over frozen experience, with checkpoint recovery and monitoring."""

from .loop import run_training_loop
from .preparation import prepare_training


def train(
    model,
    data,
    *,
    source,
    config=None,
    progress=None,
    progress_epoch=None,
    checkpoint_dir=None,
    resume_history=None,
    resume_state=None,
    stop_after_epoch=None,
    extend_epoch_budget=False,
):
    """Fit value heads using the configured preparation and training loop."""
    session = prepare_training(
        model,
        data,
        source=source,
        config=config,
        progress=progress,
        progress_epoch=progress_epoch,
        checkpoint_dir=checkpoint_dir,
        resume_history=resume_history,
        resume_state=resume_state,
        stop_after_epoch=stop_after_epoch,
        extend_epoch_budget=extend_epoch_budget,
    )
    return run_training_loop(session)
