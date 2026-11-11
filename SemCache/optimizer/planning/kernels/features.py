"""One owned numeric batch for PyTorch; metadata and gradients stay in Python."""

import torch

from planning.learning.features.schema import HISTORY_NUMERIC, NODE_NUMERIC

from . import _search_core


def pack(features, reference):
    buffers = _search_core.FeatureBuffers(features, NODE_NUMERIC, HISTORY_NUMERIC)
    count = buffers.offsets[-1]
    if count == 0:
        return [None for _ in features]
    numeric = torch.frombuffer(buffers.numeric, dtype=torch.float64).reshape(
        count, NODE_NUMERIC
    )
    history = torch.frombuffer(buffers.history, dtype=torch.float64).reshape(
        count, HISTORY_NUMERIC
    )
    numeric = numeric.to(device=reference.device, dtype=reference.dtype)
    history = history.to(device=reference.device, dtype=reference.dtype)
    return [
        {
            key: (numeric[i], history[i])
            for key, i in zip(graph["nodes"], range(start, end))
        }
        for graph, start, end in zip(features, buffers.offsets, buffers.offsets[1:])
    ]
