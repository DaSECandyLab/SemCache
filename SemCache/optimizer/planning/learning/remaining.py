"""Plan-conditioned, permutation-invariant encoding of unfinished work."""

import math

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

REMAINING_ENCODING = "plan_attention_sum_v1"


def pad_remaining(groups, reference, width):
    """Keep operator vectors until the current plan supplies attention weights."""
    lengths = [len(group) for group in groups]
    maximum = max(1, max(lengths, default=0))
    if any(lengths):
        vectors = pad_sequence(groups, batch_first=True)
    else:
        vectors = reference.new_zeros((len(groups), maximum, width))
    mask = torch.arange(maximum, device=reference.device).unsqueeze(0) < torch.tensor(
        lengths, device=reference.device
    ).unsqueeze(1)
    return vectors, mask


class RemainingReadout(nn.Module):
    """Attend to relevant operators and retain total unfinished workload."""

    def __init__(self, width, plan_width):
        super().__init__()
        self.query = nn.Linear(plan_width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.scale = math.sqrt(width)
        self.output_size = width * 2

    def forward(self, plan, operators, mask):
        scores = (
            self.key(operators) * self.query(plan).unsqueeze(1)
        ).sum(-1) / self.scale
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        # Empty sets have finite, exactly zero summaries, including gradients.
        weights = scores.softmax(-1) * mask.to(scores.dtype)
        attention = (operators * weights.unsqueeze(-1)).sum(1)
        total = (operators * mask.unsqueeze(-1)).sum(1)
        return torch.cat((attention, total), dim=-1)
