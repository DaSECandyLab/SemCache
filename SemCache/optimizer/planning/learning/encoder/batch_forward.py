"""Vectorized padded-graph batch encoding and static input memoization."""

import math
from typing import cast

import torch

from ..batches.graph import node_inputs, remaining_inputs
from .layers import AttentionLayer, neighborhood


def remaining_batch(encoder, graph_work):
    if encoder._inference_static_memo is None:
        return remaining_inputs(encoder, graph_work)
    # Cache only each plan-independent operator token. Attention depends
    # on the current GAT output and must be recomputed for every state.
    cache = encoder._inference_static_memo["work"]
    row_keys = encoder._inference_static_memo["work_rows"]
    templates = encoder._inference_static_memo["work_templates"]
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
        encoded, _ = remaining_inputs(encoder, [list(missing.values())])
        cache.update(zip(missing, encoded[0].unbind(0)))
    unique_ids = []
    seen_ids = {}
    for group in groups:
        for identifier in group:
            if identifier not in seen_ids:
                seen_ids[identifier] = len(unique_ids)
                unique_ids.append(identifier)
    max_len = max(1, max((len(group) for group in groups), default=0))
    if not unique_ids:
        device = encoder.words.weight.device
        result = (
            encoder.words.weight.new_zeros((len(groups), max_len, encoder.width)),
            torch.zeros((len(groups), max_len), dtype=torch.bool, device=device),
        )
    else:
        pad_idx = len(unique_ids)
        zero_row = encoder.words.weight.new_zeros((1, encoder.width))
        bank = torch.cat(
            (torch.stack([cache[key] for key in unique_ids]), zero_row), dim=0
        )
        padded = [
            [seen_ids[key] for key in group] + [pad_idx] * (max_len - len(group))
            for group in groups
        ]
        idx = torch.tensor(padded, dtype=torch.long, device=bank.device)
        result = (bank[idx], idx != pad_idx)
    if len(cache) > 8192:
        cache.clear()
    if len(row_keys) > 65536 or len(templates) > 65536:
        row_keys.clear()
        templates.clear()
        cache.clear()
    return result

def padded_topology(encoder, keys, mapping, links, max_nodes, max_degree):
    assert encoder._inference_static_memo is not None
    cache = encoder._inference_static_memo["topology"]
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

def links(encoder, key, row):
    if "structure" not in encoder.feature_groups:
        return [(key, [0.0] * 8, None)]
    if "cache" not in encoder.feature_groups:
        row = {**row, "l_previous": {}}
    if encoder._inference_static_memo is not None:
        cache = encoder._inference_static_memo["links"]
        signature = (key, tuple(row["children"]), tuple(row["l_previous"].items()))
        if signature not in cache:
            cache[signature] = neighborhood(key, row)
        return cache[signature]
    return neighborhood(key, row)

def edge_features(encoder, relations, reuse, present):
    """§4.2.3 [r_P, b_C, Linear_C(match)], shared by all GAT layers.

    Unknown strength retains b_C but masks the numeric encoding. Absent
    cache relations, including self edges, have a strictly zero r_C.
    """
    exists = relations[..., 1:2] + relations[..., 2:3]
    match = encoder.cache_relation(reuse.unsqueeze(-1))
    match = match * present.unsqueeze(-1) * exists
    return torch.cat((relations, exists, match), dim=-1)

def forward_batch(encoder, feature_batch, *, incremental=False, packed=None):
    """Encode independent plan graphs in one padded tensor batch.

    Training previously called ``forward`` once per state.  This method
    keeps every graph's edges disjoint, so attention is vectorized over
    states without changing the graph or feature semantics.
    """
    if not feature_batch:
        return encoder.words.weight.new_zeros((0, encoder.output_size))
    if packed is None:
        packed = [None] * len(feature_batch)
    if len(packed) != len(feature_batch):
        raise ValueError("Packed feature count does not match feature batch")
    with encoder.shared_descriptions():
        assert encoder._description_memo is not None
        # Descriptions are immutable inputs, but embeddings remain attached
        # to this update's autograd graph. Encode unique sequences once per
        # kind, never reuse activations after an optimizer step.
        if "task" in encoder.feature_groups:
            descriptions = {}
            for graph in feature_batch:
                for row in (*graph["nodes"].values(), *graph["work"].values()):
                    kind, ids = row["description"]
                    if (
                        kind
                        and ids
                        and (kind, tuple(ids)) not in encoder._description_memo
                    ):
                        descriptions.setdefault((kind, tuple(ids)), None)
            for kind in (1, 2):
                keys = [key for key in descriptions if key[0] == kind]
                if not keys:
                    continue
                sequences = [encoder.words(encoder.tensor(ids).long()) for _, ids in keys]
                packed_tasks = torch.nn.utils.rnn.pack_sequence(
                    sequences, enforce_sorted=False
                )
                _, hidden = encoder.descriptions[kind - 1](packed_tasks)
                for key, vector in zip(keys, hidden[0]):
                    encoder._description_memo[key] = vector
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
                links.append(encoder.links(key, graph["nodes"][key]))
            graph_links.append(links)
            graph_work.append(list(graph["work"].values()))

        vectors = torch.nn.utils.rnn.pad_sequence(
            list(
                node_inputs(encoder, input_rows, input_packed).split(
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
            if encoder._inference_static_memo is not None:
                buffers = encoder.padded_topology(
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
        relations = encoder.tensor(relation_rows).reshape(*shape, 8)
        reuse = encoder.tensor(reuse_rows).reshape(shape)
        reuse_present = torch.tensor(
            present_rows, dtype=torch.bool, device=vectors.device
        ).reshape(shape)
        edge_mask = torch.tensor(
            mask_rows, dtype=torch.bool, device=vectors.device
        ).reshape(shape)
        relations = encoder.edge_features(relations, reuse, reuse_present)
        for module in encoder.layers:
            layer = cast(AttentionLayer, module)
            sources = vectors.gather(
                1,
                source_indices.unsqueeze(-1)
                .expand(-1, -1, -1, encoder.width)
                .reshape(batch_size, max_nodes * max_degree, encoder.width),
            ).reshape(batch_size, max_nodes, max_degree, encoder.width)
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
        plan = encoder.readout(encoder.final_norm(vectors), node_mask, root_mask)
        remaining, work_mask = encoder.remaining_batch(graph_work)
        work = encoder.remaining_readout(plan, remaining, work_mask)
        sizes = vectors.new_tensor(
            [math.log1p(len(rows)) for rows in graph_work]
        ).unsqueeze(-1)
        return torch.cat((plan, work, sizes), dim=-1)
