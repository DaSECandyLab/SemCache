"""Pool plan nodes without changing GAT messages or remaining-work encoding."""

import torch
from torch import nn


class PlanReadout(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.output_size = width * 2

    def forward(self, values, mask, roots):
        valid = mask.unsqueeze(-1)
        total = (values * valid).sum(1)
        mean = total / mask.sum(1).clamp_min(1).unsqueeze(-1)
        root = (values * roots.unsqueeze(-1)).sum(1) / roots.sum(1).clamp_min(
            1
        ).unsqueeze(-1)
        return torch.cat((mean, root), dim=-1)
