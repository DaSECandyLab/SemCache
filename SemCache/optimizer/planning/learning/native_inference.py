"""CUDA inference over native-owned graph buffers; independent of GAT parameters."""

import torch

from .graph_batch import field_sums


def prepare_native_batch(encoder, batch):
    """Transfer native-owned contiguous buffers once per scoring call."""
    if encoder.training or torch.is_grad_enabled():
        raise RuntimeError("Native graph batches require eval and no gradients")
    floating = ("numeric", "history", "relations", "reuse", "sizes")
    indices = ("node_indices", "source_indices")
    masks = ("reuse_present", "edge_mask", "node_mask", "root_mask")
    device = encoder.words.weight.device
    return {
        **{
            name: torch.as_tensor(
                getattr(batch, name), dtype=encoder.words.weight.dtype, device=device
            )
            for name in floating
        },
        **{
            name: torch.as_tensor(getattr(batch, name), dtype=torch.long, device=device)
            for name in indices
        },
        **{
            name: torch.as_tensor(getattr(batch, name), dtype=torch.bool, device=device)
            for name in masks
        },
    }


def forward_native_batch(encoder, batch, buffers, start, end, node_vectors):
    """Evaluate one original chunk using native topology and unique rows.

    Unique node projections retain the original chunk's first occurrence
    order and batch size. This path is eval-only; training uses forward_batch.
    """
    if encoder.training or torch.is_grad_enabled():
        raise RuntimeError("Native graph batches require eval and no gradients")
    chunk = start // batch.chunk_size
    first, last = batch.chunk_offsets[chunk : chunk + 2]
    with encoder.shared_descriptions():
        assert encoder._description_memo is not None
        descriptions = {}
        for kind, ids in batch.descriptions_by_chunk[chunk]:
            key = (kind, tuple(ids))
            if kind and ids and key not in encoder._description_memo:
                descriptions.setdefault(key, None)
        for kind in (1, 2):
            keys = [key for key in descriptions if key[0] == kind]
            if not keys:
                continue
            sequences = [encoder.words(encoder.tensor(ids).long()) for _, ids in keys]
            tasks = torch.nn.utils.rnn.pack_sequence(sequences, enforce_sorted=False)
            _, hidden = encoder.descriptions[kind - 1](tasks)
            encoder._description_memo.update(zip(keys, hidden[0]))
        if first != last:
            rows = batch.templates[first:last]
            fields = field_sums(
                encoder, [ids for row in rows for ids in row["fields"]]
            ).reshape(len(rows), -1)
            initial = field_sums(
                encoder, [ids for row in rows for ids in row["first_fields"]]
            ).reshape(len(rows), -1)
            descriptions = torch.stack(
                [encoder.task(row["description"]) for row in rows]
            )
            node_vectors[first:last] = encoder.node(
                torch.cat(
                    (
                        descriptions,
                        fields,
                        initial,
                        buffers["history"][first:last],
                        buffers["numeric"][first:last],
                    ),
                    dim=-1,
                )
            )
        vectors = node_vectors[buffers["node_indices"][start:end]]
        node_mask = buffers["node_mask"][start:end]
        # Native buffers use the layer-wide maximum; trim to the same
        # per-chunk padded dimensions as forward_batch before any matmul.
        max_nodes, max_degree = batch.chunk_shapes[chunk]
        vectors = vectors[:, :max_nodes]
        vectors = vectors * node_mask[:, :max_nodes].unsqueeze(-1)
        source_indices = buffers["source_indices"][start:end, :max_nodes]
        edge_mask = buffers["edge_mask"][start:end, :max_nodes]
        source_indices = source_indices[:, :, :max_degree]
        edge_mask = edge_mask[:, :, :max_degree].contiguous()
        relations = encoder.edge_features(
            buffers["relations"][start:end, :max_nodes, :max_degree].contiguous(),
            buffers["reuse"][start:end, :max_nodes, :max_degree].contiguous(),
            buffers["reuse_present"][start:end, :max_nodes, :max_degree].contiguous(),
        )
        node_mask = node_mask[:, :max_nodes].contiguous()
        batch_size = end - start
        for module in encoder.layers:
            layer = module
            sources = vectors.gather(
                1,
                source_indices.unsqueeze(-1)
                .expand(-1, -1, -1, encoder.width)
                .reshape(batch_size, max_nodes * max_degree, encoder.width),
            ).reshape(batch_size, max_nodes, max_degree, encoder.width)
            vectors = layer.read_batch(vectors, sources, relations, edge_mask)
            vectors = vectors * node_mask.unsqueeze(-1)
        plan = encoder.readout(
            encoder.final_norm(vectors),
            node_mask,
            buffers["root_mask"][start:end, :max_nodes].contiguous(),
        )
        work_ids = batch.work_indices[start:end].tolist()
        unique_work = list(dict.fromkeys(work_ids))
        work_positions = {key: index for index, key in enumerate(unique_work)}
        remaining, work_mask = encoder.remaining_batch(
            [batch.work[index] for index in unique_work]
        )
        positions = torch.tensor(
            [work_positions[key] for key in work_ids],
            dtype=torch.long,
            device=remaining.device,
        )
        work = encoder.remaining_readout(
            plan, remaining[positions], work_mask[positions]
        )
        return torch.cat((plan, work, buffers["sizes"][start:end]), dim=-1)
