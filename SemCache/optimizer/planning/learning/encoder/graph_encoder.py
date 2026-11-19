"""GAT-PE (§4.2): directed attention, edge messages and Mean + Root readout.

Reference: Lyu et al., PVLDB 18(13), 5716–5730, §6. No positional
encoding or Super Node. Incremental evaluation caches only unchanged inputs
and incoming neighborhoods; training never reuses detached activations.
"""

import math
import threading
from collections import OrderedDict
from contextlib import contextmanager
from typing import cast

import torch
from torch import nn

from planning.learning.features.schema import (
    BUCKETS,
    FEATURE_GROUPS,
    HISTORY_NUMERIC,
    NODE_NUMERIC,
    WORK_NUMERIC,
)

from ...state import digest
from ..batches.graph import project_node_inputs
from .batch_forward import (
    edge_features,
    forward_batch,
    links,
    padded_topology,
    remaining_batch,
)
from .layers import AttentionLayer, mlp
from .native_inference import forward_native_batch, prepare_native_batch
from .readout import PlanReadout
from .remaining import RemainingReadout

_STATIC_INPUT_LOCK = threading.Lock()


class GraphEncoder(nn.Module):
    def __init__(
        self,
        width,
        heads,
        layers,
        dropout,
        feature_groups=FEATURE_GROUPS,
    ):
        super().__init__()
        self.width, self.feature_groups = width, frozenset(feature_groups)
        self.words = nn.Embedding(BUCKETS, width)
        self.fields = nn.Embedding(BUCKETS, width)
        self.descriptions = nn.ModuleList(
            [nn.GRU(width, width, batch_first=True) for _ in range(2)]
        )
        # Concatenate history statistics into x_o before the shared affine node projection W_plan.
        self.node = nn.Linear(NODE_NUMERIC + HISTORY_NUMERIC + width * 5, width)
        self.work = mlp(WORK_NUMERIC + width * 3, width)
        self.cache_relation = nn.Linear(1, width)
        self.layers = nn.ModuleList(
            [AttentionLayer(width, heads, dropout) for _ in range(layers)]
        )
        self.final_norm = nn.LayerNorm(width)
        self.readout = PlanReadout(width)
        self.remaining_readout = RemainingReadout(width, self.readout.output_size)
        self.output_size = self.readout.output_size + self.remaining_readout.output_size + 1
        self.memo = OrderedDict()
        self._description_memo = None
        self._inference_static_memo = None
        self._inference_static_owner = None
        self._inference_static_active = False
        self.cache_hits = self.cache_misses = 0

    def train(self, mode=True):
        if self._inference_static_memo is not None:
            self._check_static_inference()
            if mode:
                raise RuntimeError("Cannot train during an inference session")
            return super().train(mode)
        self.memo.clear()
        return super().train(mode)

    def _apply(self, fn, recurse=True):
        if self._inference_static_memo is not None:
            raise RuntimeError("Cannot move encoder weights during inference")
        self.memo.clear()
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        if self._inference_static_memo is not None:
            raise RuntimeError("Cannot load encoder weights during inference")
        self.memo.clear()
        return super()._load_from_state_dict(*args, **kwargs)

    def tensor(self, values):
        return self.words.weight.new_tensor(values)

    def mean(self, values):
        return (
            torch.stack(values).mean(0)
            if values
            else self.words.weight.new_zeros(self.width)
        )

    @contextmanager
    def shared_descriptions(self):
        if self._inference_static_memo is not None:
            self._check_static_inference()
        if self._description_memo is not None:
            raise RuntimeError("Description batches must not overlap")
        self._description_memo = (
            self._inference_static_memo["descriptions"]
            if self._inference_static_memo is not None
            else {}
        )
        try:
            yield
        finally:
            self._description_memo = None
            self._bound_static_inputs()

    @contextmanager
    def inference_static_inputs(self, *, reuse=False):
        """Own frozen input caches, or explicitly borrow a session's caches."""
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Static inference reuse requires eval and no gradients")
        with _STATIC_INPUT_LOCK:
            borrowed = self._inference_static_memo is not None
            if borrowed:
                assert self._inference_static_memo is not None
                self._check_static_inference()
                if not reuse or self._inference_static_active:
                    raise RuntimeError("Static inference batches must not overlap")
                self._inference_static_active = True
                # Keep every learned projection/GRU batch identical to an
                # isolated scoring call. Cross-call activation reuse changes
                # rounding and can reorder nearly tied Beam candidates.
                for name in (
                    "descriptions",
                    "work",
                    "node_templates",
                    "node_inputs",
                    "node_identities",
                ):
                    self._inference_static_memo[name].clear()
            else:
                self._inference_static_owner = threading.get_ident()
                self._inference_static_memo = {
                    "descriptions": {},
                    "work": {},
                    "work_rows": {},
                    "work_templates": {},
                    "node_templates": {},
                    "node_inputs": {},
                    "node_identities": {},
                    "links": {},
                    "topology": {},
                }
        try:
            yield
        finally:
            if borrowed:
                self._inference_static_active = False
                self._bound_static_inputs()
            else:
                self._inference_static_memo = None
                self._inference_static_owner = None

    def _check_static_inference(self):
        if self._inference_static_owner != threading.get_ident():
            raise RuntimeError("Inference caches belong to another thread")
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("Static inference reuse requires eval and no gradients")

    def _bound_static_inputs(self):
        # Clear related key spaces together: node/work IDs are allocated from
        # their template dictionaries and cannot survive a template reset.
        memo = self._inference_static_memo
        if memo is None:
            return
        for names, limit in (
            (("descriptions",), 8192),
            (("work",), 8192),
            (("work_rows", "work_templates", "work"), 65536),
            (("node_templates", "node_inputs", "node_identities"), 65536),
            (("links", "topology"), 8192),
        ):
            if any(len(memo[name]) > limit for name in names):
                for name in names:
                    memo[name].clear()

    def task(self, description):
        kind, ids = description
        if "task" not in self.feature_groups or not kind or not ids:
            return self.mean([])
        key = (kind, tuple(ids))
        if self._description_memo is not None and key in self._description_memo:
            return self._description_memo[key]
        _, hidden = self.descriptions[kind - 1](
            self.words(self.tensor(ids).long()).unsqueeze(0)
        )
        result = hidden[0, 0]
        if self._description_memo is not None:
            self._description_memo[key] = result
        return result

    def cached(self, key, compute, enabled):
        if not enabled:
            return compute()
        if key in self.memo:
            self.memo.move_to_end(key)
            self.cache_hits += 1
            return self.memo[key]
        self.cache_misses += 1
        value = compute()
        self.memo[key] = value
        if len(self.memo) > 16384:
            self.memo.popitem(last=False)
        return value

    def node_input(self, row, packed=None):
        return project_node_inputs(self, [row], None if packed is None else [packed])[0]

    def forward(self, features, *, incremental=True, packed=None):
        cache = incremental and not self.training and not torch.is_grad_enabled()
        rows = features["nodes"]
        # Signatures exist only to key the memo; hashing every node (and every
        # layer's neighborhood) is pure overhead while caching is disabled, which
        # is exactly the training case.
        signatures = (
            {
                key: digest(
                    {
                        k: v
                        for k, v in row.items()
                        if k not in ("children", "l_previous")
                    }
                )
                for key, row in rows.items()
            }
            if cache
            else {}
        )
        vectors = {
            key: self.cached(
                ("input", signatures.get(key)),
                lambda row=row, key=key: self.node_input(
                    row, None if packed is None else packed[key]
                ),
                cache,
            )
            for key, row in rows.items()
        }
        links = {key: self.links(key, row) for key, row in rows.items()}
        prepared_edges = {}

        def edges_for(key):
            # Prepare every edge in one linear call only when a GAT update is
            # needed. An unchanged incremental graph bypasses this entirely.
            if not prepared_edges:
                flat = [link for current in links.values() for link in current]
                encoded = self.edge_features(
                    self.tensor([p for _, p, _ in flat]),
                    self.tensor([c if c is not None else 0 for _, _, c in flat]),
                    self.tensor([c is not None for _, _, c in flat]),
                )
                prepared_edges.update(
                    zip(
                        rows,
                        encoded.split([len(current) for current in links.values()]),
                    )
                )
            return prepared_edges[key]

        for index, module in enumerate(self.layers):
            layer = cast(AttentionLayer, module)
            next_vectors, next_signatures = {}, {}
            for key, row in rows.items():
                current = links[key]
                if cache:
                    signature = digest(
                        [
                            signatures[key],
                            [(signatures[s], p, c) for s, p, c in current],
                        ]
                    )
                    next_signatures[key] = signature
                else:
                    signature = None
                next_vectors[key] = self.cached(
                    ("layer", index, signature),
                    lambda key=key, current=current, layer=layer, vectors=vectors: (
                        layer.read(
                            vectors[key],
                            [vectors[s] for s, _, _ in current],
                            edges_for(key),
                        )
                    ),
                    cache,
                )
            vectors, signatures = next_vectors, next_signatures
        keys = list(vectors)
        children = {child for row in rows.values() for child in row["children"]}
        values = (
            torch.stack([self.final_norm(value) for value in vectors.values()])
            if keys
            else self.words.weight.new_zeros((1, self.width))
        )
        mask = torch.tensor(
            [[bool(keys)] * len(values)], dtype=torch.bool, device=values.device
        )
        roots = torch.tensor(
            [[key not in children for key in keys] or [False]],
            dtype=torch.bool,
            device=values.device,
        )
        plan = self.readout(values.unsqueeze(0), mask, roots)[0]
        rows = list(features["work"].values())
        remaining, mask = self.remaining_batch([rows])
        work = self.remaining_readout(plan.unsqueeze(0), remaining, mask)[0]
        return torch.cat(
            (plan, work, self.tensor([math.log1p(len(rows))]))
        )

    def remaining_batch(self, graph_work):
        return remaining_batch(self, graph_work)

    def padded_topology(self, keys, mapping, links_rows, max_nodes, max_degree):
        return padded_topology(self, keys, mapping, links_rows, max_nodes, max_degree)

    def links(self, key, row):
        return links(self, key, row)

    def edge_features(self, relations, reuse, present):
        return edge_features(self, relations, reuse, present)

    def forward_batch(self, feature_batch, *, incremental=False, packed=None):
        return forward_batch(
            self, feature_batch, incremental=incremental, packed=packed
        )

    def prepare_native_batch(self, batch):
        return prepare_native_batch(self, batch)

    def forward_native_batch(self, batch, buffers, start, end, node_vectors):
        return forward_native_batch(self, batch, buffers, start, end, node_vectors)
