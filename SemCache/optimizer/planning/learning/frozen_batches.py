"""Reuse immutable training inputs, rebuilding learned activations every update."""

import math
from collections import OrderedDict

import torch
from torch.nn.functional import embedding_bag
from torch.nn.utils.rnn import PackedSequence, pack_sequence

from planning.learning.features.schema import (
    HISTORY_NUMERIC,
    IMPLEMENTATION_SIZE,
    KINDS,
    NODE_NUMERIC,
    WORK_NUMERIC,
)


class FrozenGraphBatch:
    def __init__(self, encoder, graphs):
        self.tensors = []
        reference = encoder.words.weight

        def tensor(values, dtype=None):
            value = torch.tensor(
                values, device=reference.device, dtype=dtype or reference.dtype
            )
            self.tensors.append(value)
            return value

        def fields(groups):
            ids, offsets = [], [0]
            for group in groups:
                ids.extend(group)
                offsets.append(len(ids))
            return tensor(ids, torch.long), tensor(offsets, torch.long), len(groups)

        nodes = [row for g in graphs for row in g["nodes"].values()]
        work = [row for g in graphs for row in g["work"].values()]
        self.batch_size, self.node_count, self.work_count = (
            len(graphs),
            len(nodes),
            len(work),
        )
        self.max_nodes = max(1, max(len(g["nodes"]) for g in graphs))
        self.numeric = tensor([r["numeric"] for r in nodes]).reshape(-1, NODE_NUMERIC)
        self.history = tensor(
            [r["l_existing"] or [0.0] * HISTORY_NUMERIC for r in nodes]
        ).reshape(-1, HISTORY_NUMERIC)
        boundary = len(KINDS) + IMPLEMENTATION_SIZE
        if "implementation" not in encoder.feature_groups:
            self.numeric[:, len(KINDS) : boundary] = 0
        if "statistics" not in encoder.feature_groups:
            self.numeric[:, boundary:-3] = 0
        if "cache" not in encoder.feature_groups:
            self.numeric[:, -3:] = 0
            self.history.zero_()
        if "history" not in encoder.feature_groups:
            self.history[:, 3:6] = 0
            self.history[:, 9:12] = 0
            self.history[:, 15:18] = 0
        self.fields = fields([ids for r in nodes for ids in r["fields"]])
        self.first = fields([ids for r in nodes for ids in r["first_fields"]])
        self.work_fields = fields([ids for r in work for ids in r["fields"]])
        self.work_numeric = tensor([r["numeric"] for r in work]).reshape(
            -1, WORK_NUMERIC
        )
        if "statistics" not in encoder.feature_groups:
            self.work_numeric[:, len(KINDS) :] = 0

        descriptions = {}
        if "task" in encoder.feature_groups:
            for graph in graphs:
                for row in (*graph["nodes"].values(), *graph["work"].values()):
                    kind, ids = row["description"]
                    if kind in (1, 2) and ids:
                        descriptions.setdefault((kind, tuple(ids)), None)
        self.descriptions, positions = [], {}
        for kind in (1, 2):
            keys = [key for key in descriptions if key[0] == kind]
            if not keys:
                continue
            # Pack integer IDs once; embedding and GRU outputs are recomputed
            # with gradients on every update, using the same packed ordering.
            packed = pack_sequence(
                [torch.tensor(ids, dtype=torch.long) for _, ids in keys],
                enforce_sorted=False,
            ).to(reference.device)
            self.tensors.extend(v for v in packed if isinstance(v, torch.Tensor))
            self.descriptions.append((kind, packed))
            for key in keys:
                positions[key] = len(positions) + 1
        self.node_descriptions = tensor(
            [
                positions.get((r["description"][0], tuple(r["description"][1])), 0)
                for r in nodes
            ],
            torch.long,
        )
        self.work_descriptions = tensor(
            [
                positions.get((r["description"][0], tuple(r["description"][1])), 0)
                for r in work
            ],
            torch.long,
        )

        links = [
            [encoder.links(key, row) for key, row in graph["nodes"].items()]
            for graph in graphs
        ]
        self.max_degree = max(
            1, max((len(item) for group in links for item in group), default=1)
        )
        node_indices, node_masks, roots = [], [], []
        sources, relations, reuse, present, masks = [], [], [], [], []
        work_indices, work_masks = [], []
        self.max_work = max(1, max(len(g["work"]) for g in graphs))
        node_offset = work_offset = 0
        for graph, group in zip(graphs, links):
            keys = list(graph["nodes"])
            mapping = {key: i for i, key in enumerate(keys)}
            pad = self.max_nodes - len(keys)
            node_indices.append(
                list(range(node_offset, node_offset + len(keys)))
                + [self.node_count] * pad
            )
            node_masks.append([True] * len(keys) + [False] * pad)
            children = {
                child for r in graph["nodes"].values() for child in r["children"]
            }
            roots.append([key not in children for key in keys] + [False] * pad)
            for i in range(self.max_nodes):
                current = group[i] if i < len(keys) else []
                padding = self.max_degree - len(current)
                sources.append([mapping[s] for s, _, _ in current] + [0] * padding)
                relations.append([r for _, r, _ in current] + [[0.0] * 8] * padding)
                reuse.append(
                    [v if v is not None else 0 for _, _, v in current] + [0.0] * padding
                )
                present.append(
                    [v is not None for _, _, v in current] + [False] * padding
                )
                masks.append([True] * len(current) + [i >= len(keys)] * padding)
            count = len(graph["work"])
            work_indices.append(
                list(range(work_offset, work_offset + count))
                + [self.work_count] * (self.max_work - count)
            )
            work_masks.append([True] * count + [False] * (self.max_work - count))
            node_offset += len(keys)
            work_offset += count
        self.node_indices = tensor(node_indices, torch.long)
        self.node_mask = tensor(node_masks, torch.bool)
        self.root_mask = tensor(roots, torch.bool)
        shape = self.batch_size, self.max_nodes, self.max_degree
        self.sources = tensor(sources, torch.long).reshape(shape)
        self.relations = tensor(relations).reshape(*shape, 8)
        self.reuse = tensor(reuse).reshape(shape)
        self.present = tensor(present, torch.bool).reshape(shape)
        self.edge_mask = tensor(masks, torch.bool).reshape(shape)
        self.work_indices = tensor(work_indices, torch.long)
        self.work_mask = tensor(work_masks, torch.bool)
        self.sizes = tensor([[math.log1p(len(g["work"]))] for g in graphs])

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors)

    @staticmethod
    def field_values(encoder, bag):
        ids, offsets, count = bag
        if not ids.numel():
            return encoder.fields.weight.new_zeros((count, encoder.width))
        return embedding_bag(
            ids, encoder.fields.weight, offsets, mode="sum", include_last_offset=True
        )

    def forward(self, encoder):
        zero = encoder.words.weight.new_zeros((1, encoder.width))
        descriptions = [zero]
        for kind, packed in self.descriptions:
            embedded = PackedSequence(
                encoder.words(packed.data),
                packed.batch_sizes,
                packed.sorted_indices,
                packed.unsorted_indices,
            )
            _, hidden = encoder.descriptions[kind - 1](embedded)
            descriptions.append(hidden[0])
        descriptions = torch.cat(descriptions, dim=0)
        fields = self.field_values(encoder, self.fields).reshape(
            self.node_count, encoder.width * 2
        )
        first = (
            self.field_values(encoder, self.first).reshape(
                self.node_count, encoder.width * 2
            )
            if "implementation" in encoder.feature_groups
            else fields.new_zeros((self.node_count, encoder.width * 2))
        )
        nodes = encoder.node(
            torch.cat(
                (
                    descriptions[self.node_descriptions],
                    fields,
                    first,
                    self.history,
                    self.numeric,
                ),
                dim=-1,
            )
        )
        vectors = torch.cat((nodes, zero), dim=0)[self.node_indices]
        edges = encoder.edge_features(self.relations, self.reuse, self.present)
        for layer in encoder.layers:
            sources = vectors.gather(
                1,
                self.sources.unsqueeze(-1)
                .expand(-1, -1, -1, encoder.width)
                .reshape(
                    self.batch_size, self.max_nodes * self.max_degree, encoder.width
                ),
            ).reshape(self.batch_size, self.max_nodes, self.max_degree, encoder.width)
            vectors = layer.read_batch(vectors, sources, edges, self.edge_mask)
            vectors = vectors * self.node_mask.unsqueeze(-1)
        plan = encoder.readout(
            encoder.final_norm(vectors), self.node_mask, self.root_mask
        )
        if self.work_count:
            fields = self.field_values(encoder, self.work_fields).reshape(
                self.work_count, encoder.width * 2
            )
            work = encoder.work(
                torch.cat(
                    (descriptions[self.work_descriptions], fields, self.work_numeric),
                    dim=-1,
                )
            )
            work = torch.cat((work, zero), dim=0)[self.work_indices]
        else:
            work = zero.expand(self.batch_size, 1, -1)
        work = encoder.remaining_readout(plan, work, self.work_mask)
        return torch.cat((plan, work, self.sizes), dim=-1)


class FrozenBatchCache:
    """Bounded, per-fit tensors; never retains learned outputs across updates."""

    def __init__(self, encoder, *, max_bytes=None, max_entries=1024):
        self.encoder = encoder
        if max_bytes is None:
            max_bytes = 512 * 1024 * 1024
            device = encoder.words.weight.device
            if device.type == "cuda":
                free, _ = torch.cuda.mem_get_info(device)
                # Training and monitoring use different immutable batch
                # layouts. Keep both working sets instead of rebuilding the
                # monitor's tensors after every parameter-update pass.
                max_bytes = min(2 * 1024**3, max(1, free // 4))
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("Frozen input cache budget must be positive")
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("Frozen input cache entry limit must be positive")
        self.max_bytes, self.max_entries = max_bytes, max_entries
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = 0

    def predict(self, model, unique, rows, *, log_time):
        key = tuple(sorted(row["base_key"] for row in unique))
        cached = self.entries.pop(key, None)
        if cached is None:
            self.misses += 1
            order = {row["base_key"]: i for i, row in enumerate(unique)}
            batch = FrozenGraphBatch(self.encoder, [r["features"] for r in unique])
            cached = order, batch
            self.bytes += batch.nbytes
        else:
            self.hits += 1
        self.entries[key] = cached
        while self.entries and (
            self.bytes > self.max_bytes or len(self.entries) > self.max_entries
        ):
            _, (_, discarded) = self.entries.popitem(last=False)
            self.bytes -= discarded.nbytes
        order, batch = cached
        z = batch.forward(self.encoder)
        indices = torch.tensor(
            [order[row["base_key"]] for row in rows], dtype=torch.long, device=z.device
        )
        return model.predictions(z[indices], log_time=log_time)
