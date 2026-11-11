"""Vectorize node inputs without changing field sums or learned projections."""

import torch
from torch.nn.functional import embedding_bag

from planning.learning.features.schema import (
    HISTORY_NUMERIC,
    IMPLEMENTATION_SIZE,
    KINDS,
    NODE_NUMERIC,
)


def field_sums(encoder, groups):
    ids, offsets = [], [0]
    for group in groups:
        ids.extend(group)
        offsets.append(len(ids))
    if not ids:
        return encoder.fields.weight.new_zeros((len(groups), encoder.width))
    device = encoder.fields.weight.device
    return embedding_bag(
        torch.tensor(ids, dtype=torch.long, device=device),
        encoder.fields.weight,
        torch.tensor(offsets, dtype=torch.long, device=device),
        mode="sum",
        include_last_offset=True,
    )


def node_inputs(encoder, rows, packed):
    memo = encoder._inference_static_memo
    if memo is None:
        return project_node_inputs(encoder, rows, packed)
    templates = memo["node_templates"]
    cache = memo["node_inputs"]
    identities = memo["node_identities"]
    keys, missing = [], {}
    for row in rows:
        identity = (
            id(row["description"]),
            id(row["fields"]),
            id(row["first_fields"]),
            id(row["numeric"]),
            id(row["l_existing"]),
        )
        previous = identities.get(identity)
        if previous is None:
            if len(row["numeric"]) != NODE_NUMERIC:
                raise ValueError("Numeric feature width mismatch")
            if (
                row["l_existing"] is not None
                and len(row["l_existing"]) != HISTORY_NUMERIC
            ):
                raise ValueError("History feature width mismatch")
            template = (
                row["description"][0],
                tuple(row["description"][1]),
                tuple(tuple(side) for side in row["fields"]),
                tuple(tuple(side) for side in row["first_fields"]),
            )
            template_id = templates.setdefault(template, len(templates))
            key = (
                template_id,
                tuple(row["numeric"]),
                tuple(row["l_existing"]) if row["l_existing"] is not None else None,
            )
            # Retain inputs for this frozen inference call so IDs cannot be
            # recycled while a value key is shared by multiple plan states.
            identities[identity] = (row, key)
        else:
            key = previous[1]
        keys.append(key)
        if key not in cache:
            missing.setdefault(key, row)
    if missing:
        encoded = project_node_inputs(encoder, list(missing.values()), None)
        cache.update(zip(missing, encoded.unbind(0)))
    result = torch.stack([cache[key] for key in keys])
    # Bound host keys and detached vectors even for layers with little reuse.
    if len(cache) > 65536:
        for key in list(cache)[: len(cache) - 65536]:
            del cache[key]
    if len(identities) > 65536:
        identities.clear()
    return result


def project_node_inputs(encoder, rows, packed):
    numeric = (
        encoder.tensor([row["numeric"] for row in rows])
        if packed is None
        else torch.stack(
            [
                value[0] if value is not None else encoder.tensor(row["numeric"])
                for row, value in zip(rows, packed)
            ]
        )
    )
    boundary = len(KINDS) + IMPLEMENTATION_SIZE
    if "implementation" not in encoder.feature_groups:
        numeric[:, len(KINDS) : boundary] = 0
    if "statistics" not in encoder.feature_groups:
        numeric[:, boundary:-3] = 0
    if "cache" not in encoder.feature_groups:
        numeric[:, -3:] = 0
    history = numeric.new_zeros((len(rows), HISTORY_NUMERIC))
    if "cache" in encoder.feature_groups:
        history = (
            encoder.tensor(
                [
                    row["l_existing"]
                    if row["l_existing"] is not None
                    else [0.0] * HISTORY_NUMERIC
                    for row in rows
                ]
            )
            if packed is None
            else torch.stack(
                [
                    (encoder.tensor(row["l_existing"]) if value is None else value[1])
                    if row["l_existing"] is not None
                    else history[i]
                    for i, (row, value) in enumerate(zip(rows, packed))
                ]
            )
        )
        if "history" not in encoder.feature_groups:
            history[:, 3:6] = 0
            history[:, 9:12] = 0
            history[:, 15:18] = 0
    fields = field_sums(
        encoder, [ids for row in rows for ids in row["fields"]]
    ).reshape(len(rows), -1)
    first = (
        field_sums(
            encoder, [ids for row in rows for ids in row["first_fields"]]
        ).reshape(len(rows), -1)
        if "implementation" in encoder.feature_groups
        else fields.new_zeros((len(rows), encoder.width * 2))
    )
    descriptions = torch.stack([encoder.task(row["description"]) for row in rows])
    return encoder.node(
        torch.cat((descriptions, fields, first, history, numeric), dim=-1)
    )


def remaining_inputs(encoder, graph_work):
    from .remaining import pad_remaining

    rows = [row for group in graph_work for row in group]
    if not rows:
        return pad_remaining(
            [encoder.words.weight.new_zeros((0, encoder.width)) for _ in graph_work],
            encoder.words.weight,
            encoder.width,
        )
    descriptions = torch.stack([encoder.task(row["description"]) for row in rows])
    fields = field_sums(
        encoder, [ids for row in rows for ids in row["fields"]]
    ).reshape(len(rows), -1)
    numeric = encoder.tensor([row["numeric"] for row in rows])
    if "statistics" not in encoder.feature_groups:
        numeric[:, len(KINDS) :] = 0
    encoded = encoder.work(torch.cat((descriptions, fields, numeric), dim=-1))
    return pad_remaining(
        list(encoded.split([len(group) for group in graph_work])),
        encoder.words.weight,
        encoder.width,
    )
