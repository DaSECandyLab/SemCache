"""GAT-PE message-passing layers and neighborhood construction."""

import torch
from torch import nn


def mlp(inputs, width, outputs=None):
    return nn.Sequential(
        nn.Linear(inputs, width), nn.GELU(), nn.Linear(width, outputs or width)
    )


def neighborhood(key, row):
    """Direct inputs + compatible cache-source ancestors + self, once each."""
    children = row["children"]
    result = []
    for source in sorted(set(children) | set(row["l_previous"]) | {key}):
        plan, cache = source in children, source != key and source in row["l_previous"]
        edge_type = 3 if source == key else 2 if plan and cache else 0 if plan else 1
        position = (
            0
            if plan and len(children) == 1
            else 1 + children.index(source)
            if plan
            else 3
        )
        relation = [float(edge_type == i) for i in range(4)] + [
            float(position == i) for i in range(4)
        ]
        result.append((source, relation, row["l_previous"][source] if cache else None))
    return result


class AttentionLayer(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads, self.head_width = heads, width // heads
        self.norm = nn.LayerNorm(width)
        self.attention = nn.Linear(width * 2, width, bias=False)
        self.relation_attention = nn.Linear(8 + 1 + width, width, bias=False)
        self.attention_vector = nn.Parameter(torch.empty(heads, self.head_width))
        nn.init.xavier_uniform_(self.attention_vector)
        self.value = nn.Linear(width, width)
        self.plan_value = nn.Linear(8, width, bias=False)
        self.cache_value = nn.Linear(1 + width, width, bias=False)
        self.output = nn.Linear(width, width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = mlp(width, width * 2, width)
        self.dropout = nn.Dropout(dropout)

    def read(self, target, sources, relations):
        values = torch.stack(sources)
        p, c = relations[..., :8], relations[..., 8:]
        pair = torch.cat((target.expand(len(sources), -1), values), dim=-1)
        hidden = self.attention(pair) + self.relation_attention(relations)
        hidden = torch.nn.functional.leaky_relu(hidden, 0.2).view(
            -1, self.heads, self.head_width
        )
        scores = torch.einsum("nhd,hd->hn", hidden, self.attention_vector)
        messages = self.value(values) + self.plan_value(p) + self.cache_value(c)
        messages = messages.view(-1, self.heads, self.head_width)
        aggregate = torch.einsum(
            "hn,nhd->hd", self.dropout(scores.softmax(-1)), messages
        ).flatten()
        updated = self.norm(target + self.dropout(self.output(aggregate)))
        return self.ffn_norm(updated + self.dropout(self.ffn(updated)))

    def read_batch(self, target, sources, relations, mask):
        """Evaluate disjoint graphs together while preserving per-node edges."""
        # target: [batch, nodes, width]; sources: [batch, nodes, degree, width]
        p, c = relations[..., :8], relations[..., 8:]
        width = target.shape[-1]
        weight = self.attention.weight
        hidden = (
            torch.nn.functional.linear(
                target, weight[:, :width], self.attention.bias
            ).unsqueeze(2)
            + torch.nn.functional.linear(sources, weight[:, width:])
            + self.relation_attention(relations)
        )
        hidden = torch.nn.functional.leaky_relu(hidden, 0.2).view(
            *hidden.shape[:3], self.heads, self.head_width
        )
        # AttentionLayer.read normalizes over a target's incoming sources.
        scores = torch.einsum("bnqhk,hk->bhnq", hidden, self.attention_vector)
        scores = scores.masked_fill(~mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        weights = self.dropout(scores.softmax(-1)).permute(0, 2, 3, 1).unsqueeze(-1)
        messages = self.value(sources) + self.plan_value(p) + self.cache_value(c)
        messages = messages.view(*messages.shape[:3], self.heads, self.head_width)
        aggregate = (weights * messages).sum(2).flatten(-2)
        updated = self.norm(target + self.dropout(self.output(aggregate)))
        return self.ffn_norm(updated + self.dropout(self.ffn(updated)))
