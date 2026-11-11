"""Full-population monitoring losses with query/context-balanced reductions."""

import statistics
from collections import defaultdict

import torch

from planning.learning.epoch_batches import predict_batch

from .loss import value_loss_terms


def evaluation_metrics(model, rows, config):
    """Compute query-averaged loss and context-averaged terms in one pass."""
    model.eval()
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["query_key"]][row["context"]].append(row)
    losses, collected = [], defaultdict(list)
    with torch.no_grad():
        # Independent states can share an inference batch; loss aggregation
        # remains per context and then per query, exactly as during selection.
        values = [
            r
            for contexts in groups.values()
            for group in contexts.values()
            for r in group
        ]
        predictions = (
            torch.cat(
                [
                    predict_batch(
                        model, values[i : i + config.batch_size], log_time=True
                    )
                    for i in range(0, len(values), config.batch_size)
                ]
            )
            if values
            else model.encoder.words.weight.new_empty((0, 2))
        )
        start = 0
        for contexts in groups.values():
            context_losses = []
            for values in contexts.values():
                terms = value_loss_terms(
                    predictions[start : start + len(values)],
                    values,
                    config,
                    log_time=True,
                )
                start += len(values)
                context_losses.append(terms["total"].item())
                for name, value in terms.items():
                    collected[name].append(value.item())
            losses.append(statistics.mean(context_losses))
    return (
        {
            "queries": len(groups),
            "states": len(rows),
            "value_loss": statistics.mean(losses) if losses else None,
        },
        {name: statistics.mean(values) for name, values in collected.items()},
    )


def evaluate(model, rows, config):
    return evaluation_metrics(model, rows, config)[0]


def evaluate_terms(model, rows, config):
    """Validation loss split into time / accuracy / censor components."""
    return evaluation_metrics(model, rows, config)[1]
