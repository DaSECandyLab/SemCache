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

from ..state import digest
from .graph_batch import node_inputs, project_node_inputs, remaining_inputs
from .native_inference import forward_native_batch, prepare_native_batch
from .readout import PlanReadout
from .remaining import RemainingReadout, pad_remaining

_STATIC_INPUT_LOCK = threading.Lock()


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
        pair = torch.cat((target.unsqueeze(2).expand_as(sources), sources), dim=-1)
        hidden = self.attention(pair) + self.relation_attention(relations)
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

    def remaining_batch(self, graph_work):
        if self._inference_static_memo is None:
            return remaining_inputs(self, graph_work)
        # Cache only each plan-independent operator token. Attention depends
        # on the current GAT output and must be recomputed for every state.
        cache = self._inference_static_memo["work"]
        row_keys = self._inference_static_memo["work_rows"]
        templates = self._inference_static_memo["work_templates"]
        groups, missing = [], {}
        for group in graph_work:
            identifiers = []
            for row in group:
                identity = id(row)
                previous = row_keys.get(identity)
                if previous is None:
                    template = (
                        row["description"][0],
                        tuple(row["description"][1]),
                        tuple(tuple(side) for side in row["fields"]),
                        tuple(row["numeric"]),
                    )
                    identifier = templates.setdefault(template, len(templates))
                    row_keys[identity] = (row, identifier)
                else:
                    identifier = previous[1]
                identifiers.append(identifier)
                if identifier not in cache:
                    missing.setdefault(identifier, row)
            groups.append(identifiers)
        if missing:
            encoded, _ = remaining_inputs(self, [list(missing.values())])
            cache.update(zip(missing, encoded[0].unbind(0)))
        vectors = [
            torch.stack([cache[key] for key in group])
            if group else self.words.weight.new_zeros((0, self.width))
            for group in groups
        ]
        result = pad_remaining(vectors, self.words.weight, self.width)
        if len(cache) > 8192:
            cache.clear()
        if len(row_keys) > 65536 or len(templates) > 65536:
            row_keys.clear()
            templates.clear()
            cache.clear()
        return result

    def padded_topology(self, keys, mapping, links, max_nodes, max_degree):
        assert self._inference_static_memo is not None
        cache = self._inference_static_memo["topology"]
        signature = (tuple(keys), tuple(map(id, links)), max_nodes, max_degree)
        previous = cache.get(signature)
        if previous is not None:
            return previous[1]
        buffers = ([], [], [], [], [])
        sources, relations, reuse, present, masks = buffers
        for node_index in range(max_nodes):
            current = links[node_index] if node_index < len(keys) else []
            padding = max_degree - len(current)
            sources.append(
                [mapping[source] for source, _, _ in current] + [0] * padding
            )
            relations.append(
                [relation for _, relation, _ in current] + [[0.0] * 8] * padding
            )
            reuse.append(
                [cached if cached is not None else 0.0 for _, _, cached in current]
                + [0.0] * padding
            )
            present.append(
                [cached is not None for _, _, cached in current] + [False] * padding
            )
            masks.append([True] * len(current) + [node_index >= len(keys)] * padding)
        if len(cache) >= 8192:
            cache.clear()
        # Keep link objects alive for the ID-based topology key.
        cache[signature] = (links, buffers)
        return buffers

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

    def links(self, key, row):
        if "structure" not in self.feature_groups:
            return [(key, [0.0] * 8, None)]
        if "cache" not in self.feature_groups:
            row = {**row, "l_previous": {}}
        if self._inference_static_memo is not None:
            cache = self._inference_static_memo["links"]
            signature = (key, tuple(row["children"]), tuple(row["l_previous"].items()))
            if signature not in cache:
                cache[signature] = neighborhood(key, row)
            return cache[signature]
        return neighborhood(key, row)

    def edge_features(self, relations, reuse, present):
        """§4.2.3 [r_P, b_C, Linear_C(match)], shared by all GAT layers.

        Unknown strength retains b_C but masks the numeric encoding. Absent
        cache relations, including self edges, have a strictly zero r_C.
        """
        exists = relations[..., 1:2] + relations[..., 2:3]
        match = self.cache_relation(reuse.unsqueeze(-1))
        match = match * present.unsqueeze(-1) * exists
        return torch.cat((relations, exists, match), dim=-1)

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

    def forward_batch(self, feature_batch, *, incremental=False, packed=None):
        """Encode independent plan graphs in one padded tensor batch.

        Training previously called ``forward`` once per state.  This method
        keeps every graph's edges disjoint, so attention is vectorized over
        states without changing the graph or feature semantics.
        """
        if not feature_batch:
            return self.words.weight.new_zeros((0, self.output_size))
        if packed is None:
            packed = [None] * len(feature_batch)
        if len(packed) != len(feature_batch):
            raise ValueError("Packed feature count does not match feature batch")
        with self.shared_descriptions():
            assert self._description_memo is not None
            # Descriptions are immutable inputs, but embeddings remain attached
            # to this update's autograd graph. Encode unique sequences once per
            # kind, never reuse activations after an optimizer step.
            if "task" in self.feature_groups:
                descriptions = {}
                for graph in feature_batch:
                    for row in (*graph["nodes"].values(), *graph["work"].values()):
                        kind, ids = row["description"]
                        if (
                            kind
                            and ids
                            and (kind, tuple(ids)) not in self._description_memo
                        ):
                            descriptions.setdefault((kind, tuple(ids)), None)
                for kind in (1, 2):
                    keys = [key for key in descriptions if key[0] == kind]
                    if not keys:
                        continue
                    sequences = [self.words(self.tensor(ids).long()) for _, ids in keys]
                    packed_tasks = torch.nn.utils.rnn.pack_sequence(
                        sequences, enforce_sorted=False
                    )
                    _, hidden = self.descriptions[kind - 1](packed_tasks)
                    for key, vector in zip(keys, hidden[0]):
                        self._description_memo[key] = vector
            graph_keys = []
            graph_maps = []
            input_rows, input_packed = [], []
            graph_links = []
            graph_work = []
            for graph, graph_packed in zip(feature_batch, packed):
                keys = list(graph["nodes"])
                graph_keys.append(keys)
                mapping = {key: index for index, key in enumerate(keys)}
                graph_maps.append(mapping)
                input_rows.extend(graph["nodes"][key] for key in keys)
                input_packed.extend(
                    None if graph_packed is None else graph_packed[key] for key in keys
                )
                links = []
                for key in keys:
                    links.append(self.links(key, graph["nodes"][key]))
                graph_links.append(links)
                graph_work.append(list(graph["work"].values()))

            vectors = torch.nn.utils.rnn.pad_sequence(
                list(
                    node_inputs(self, input_rows, input_packed).split(
                        [len(keys) for keys in graph_keys]
                    )
                ),
                batch_first=True,
            )
            batch_size, max_nodes, _ = vectors.shape
            # Assemble immutable graph topology on CPU and transfer each dense buffer once.
            node_mask = torch.tensor(
                [[i < len(keys) for i in range(max_nodes)] for keys in graph_keys],
                dtype=torch.bool,
                device=vectors.device,
            )
            max_degree = max(
                max((len(item) for item in links), default=1) for links in graph_links
            )
            source_rows, relation_rows, reuse_rows, present_rows, mask_rows = (
                [],
                [],
                [],
                [],
                [],
            )
            for keys, mapping, links in zip(graph_keys, graph_maps, graph_links):
                if self._inference_static_memo is not None:
                    buffers = self.padded_topology(
                        keys, mapping, links, max_nodes, max_degree
                    )
                    for target, rows in zip(
                        (
                            source_rows,
                            relation_rows,
                            reuse_rows,
                            present_rows,
                            mask_rows,
                        ),
                        buffers,
                    ):
                        target.extend(rows)
                    continue
                for node_index in range(max_nodes):
                    current = links[node_index] if node_index < len(keys) else []
                    padding = max_degree - len(current)
                    source_rows.append(
                        [mapping[source] for source, _, _ in current] + [0] * padding
                    )
                    relation_rows.append(
                        [relation for _, relation, _ in current] + [[0.0] * 8] * padding
                    )
                    reuse_rows.append(
                        [
                            cached if cached is not None else 0.0
                            for _, _, cached in current
                        ]
                        + [0.0] * padding
                    )
                    present_rows.append(
                        [cached is not None for _, _, cached in current]
                        + [False] * padding
                    )
                    # A padded target needs dummy edges to keep softmax finite.
                    mask_rows.append(
                        [True] * len(current) + [node_index >= len(keys)] * padding
                    )
            shape = (batch_size, max_nodes, max_degree)
            source_indices = torch.tensor(
                source_rows, dtype=torch.long, device=vectors.device
            ).reshape(shape)
            relations = self.tensor(relation_rows).reshape(*shape, 8)
            reuse = self.tensor(reuse_rows).reshape(shape)
            reuse_present = torch.tensor(
                present_rows, dtype=torch.bool, device=vectors.device
            ).reshape(shape)
            edge_mask = torch.tensor(
                mask_rows, dtype=torch.bool, device=vectors.device
            ).reshape(shape)
            relations = self.edge_features(relations, reuse, reuse_present)
            for module in self.layers:
                layer = cast(AttentionLayer, module)
                sources = vectors.gather(
                    1,
                    source_indices.unsqueeze(-1)
                    .expand(-1, -1, -1, self.width)
                    .reshape(batch_size, max_nodes * max_degree, self.width),
                ).reshape(batch_size, max_nodes, max_degree, self.width)
                vectors = layer.read_batch(vectors, sources, relations, edge_mask)
                vectors = vectors * node_mask.unsqueeze(-1)
            root_rows = []
            for keys, graph in zip(graph_keys, feature_batch):
                children = {
                    child
                    for row in graph["nodes"].values()
                    for child in row["children"]
                }
                root_rows.append(
                    [key not in children for key in keys]
                    + [False] * (max_nodes - len(keys))
                )
            root_mask = torch.tensor(root_rows, dtype=torch.bool, device=vectors.device)
            plan = self.readout(self.final_norm(vectors), node_mask, root_mask)
            remaining, work_mask = self.remaining_batch(graph_work)
            work = self.remaining_readout(plan, remaining, work_mask)
            sizes = vectors.new_tensor(
                [math.log1p(len(rows)) for rows in graph_work]
            ).unsqueeze(-1)
            return torch.cat((plan, work, sizes), dim=-1)

    def prepare_native_batch(self, batch):
        return prepare_native_batch(self, batch)

    def forward_native_batch(self, batch, buffers, start, end, node_vectors):
        return forward_native_batch(self, batch, buffers, start, end, node_vectors)
