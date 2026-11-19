"""Full-population monitoring losses with query/context-balanced reductions."""

import statistics
from collections import defaultdict

import torch

from planning.learning.batches.epoch import predict_batch

from .loss import value_loss_terms


def evaluation_metrics(model, rows, config):
    """Compute query-averaged loss and context-averaged terms in one pass."""
    model.eval()
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["query_key"]][row["context"]].append(row)
    losses = []
    collected = {}
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
        detached_totals = []
        context_query = []
        collected_tensors = defaultdict(list)
        for query_index, contexts in enumerate(groups.values()):
            for values in contexts.values():
                terms = value_loss_terms(
                    predictions[start : start + len(values)],
                    values,
                    config,
                    log_time=True,
                )
                start += len(values)
                detached_totals.append(terms["total"].detach())
                context_query.append(query_index)
                for name, value in terms.items():
                    collected_tensors[name].append(value.detach())
        if detached_totals:
            total_host = torch.stack(detached_totals).cpu().tolist()
            by_query = defaultdict(list)
            for query_index, total in zip(context_query, total_host, strict=True):
                by_query[query_index].append(total)
            losses = [
                statistics.mean(by_query[query_index])
                for query_index in range(len(groups))
            ]
            term_names = list(collected_tensors)
            term_host = (
                torch.stack(
                    [torch.stack(collected_tensors[name]) for name in term_names]
                )
                .cpu()
                .tolist()
            )
            collected = dict(zip(term_names, term_host, strict=True))
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
