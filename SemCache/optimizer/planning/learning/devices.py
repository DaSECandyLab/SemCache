"""Resolve the value model's inference device independently from LLM serving."""

import torch


def load_inference_model(checkpoint, settings=None):
    if checkpoint is None:
        return None
    from planning.settings.configuration import load_config

    from .model import PlanValueModel

    settings = settings or load_config()
    options = settings["inference"]
    requested = options.get("device", "auto")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "auto"
        else torch.device(requested)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA value-model inference requested, but CUDA is unavailable"
        )
    size = options.get("gpu_batch_size", 64)
    if type(size) is not int or size < 1:
        raise ValueError("inference.gpu_batch_size must be a positive integer")
    model = PlanValueModel.load(checkpoint, device=device)
    if (
        not model.value_available
        and settings.get("training", {}).get("allow_random_initialization") is True
        and (model.training_info or {}).get("initialization") == "random"
        and (model.training_info or {}).get("epochs") == 0
    ):
        # Explicit from-scratch experiments may score with random heads. This
        # enables inference only; the saved checkpoint still records no fitted
        # heads, and ordinary training/inference keeps its supervision guard.
        model.value_available = True
    model.inference_batch_size = size
    return model
