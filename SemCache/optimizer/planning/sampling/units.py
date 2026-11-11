"""Bounded input-unit construction for independent operator sampling.

This module contains only deterministic population sampling and unit evaluation;
operator evidence propagation remains in :mod:`independent`.
"""

import math
import random

from ..state import digest, field_key
from .rows import CartesianRows


def field_tokens(fields):
    result = set()
    for field in fields:
        result.add(field_key(field))
        if isinstance(field, dict):
            if field.get("name"):
                result.add(field["name"])
        elif isinstance(field, str):
            result.add(field)
    return result


def sampling_operator(coordinator, problem, state, oid):
    op = problem.execution_operator(state, oid)
    action = state._action_by_operator.get(oid)
    if (
        oid in problem.ops
        and action is not None
        and not coordinator.sample_depth_selectivity
    ):
        op["_sample_model"] = action["model_id"]
        op["default_layout"] = action["layout"]
    return op


def bounded(coordinator, rows, salt, *, limit=None):
    """Choose coordinates before observing outcomes; never refill survivors."""
    if limit is None:
        limit = coordinator.operator_sample_rows
    if len(rows) <= limit:
        return tuple(rows)
    rng = random.Random(int(digest([coordinator.seed, salt])[:16], 16))
    return tuple(rows[i] for i in sorted(rng.sample(range(len(rows)), limit)))


def combine(coordinator, groups, salt, *, limit=None):
    if not groups:
        return ()
    if coordinator.direct_sources is not None:
        result = groups[0]
        for group in groups[1:]:
            result = tuple(
                CartesianRows.combine(unit[0][0], unit[1][0])
                for unit in units(coordinator, (result, group), salt, limit=limit)
            )
        return tuple(result)
    result = groups[0]
    for index, group in enumerate(groups[1:]):
        result = bounded(
            coordinator, CartesianRows(result, group), [salt, index], limit=limit
        )
    return bounded(coordinator, result, salt, limit=limit)


def units(coordinator, inputs, salt, *, limit=None):
    if limit is None:
        limit = coordinator.operator_sample_rows
    if not inputs or any(not group for group in inputs):
        return ()
    if len(inputs) == 1:
        if coordinator.direct_sources is not None:
            return tuple(([row],) for row in inputs[0][:limit])
        return tuple(
            ([row],) for row in bounded(coordinator, inputs[0], salt, limit=limit)
        )
    size = math.prod(map(len, inputs))
    if coordinator.direct_sources is not None:
        n, m = map(len, inputs)
        return tuple(
            (
                [inputs[0][i % n if n >= m else (i // m + i % m) % n]],
                [inputs[1][(i // n + i % n) % m if n >= m else i % m]],
            )
            for i in range(min(size, limit))
        )
    rng = random.Random(int(digest([coordinator.seed, salt])[:16], 16))
    indices = range(size) if size <= limit else sorted(rng.sample(range(size), limit))
    return tuple(
        ([inputs[0][i // len(inputs[1])]], [inputs[1][i % len(inputs[1])]])
        for i in indices
    )


def evaluate(coordinator, problem, op, inputs):
    limit = coordinator.sample_limit(problem, op)
    candidates = units(coordinator, inputs, op["operator_id"], limit=limit)
    if len(inputs) == 1:
        selected = (tuple(unit[0][0] for unit in candidates),)
        output = coordinator._selection(problem, op, selected)
        if output is None:
            return (), candidates, (), False
        passed = {row.identity for row in output}
        return (
            tuple(output),
            candidates,
            tuple(unit[0][0].identity in passed for unit in candidates),
            True,
        )
    output, successes = [], []
    for unit in candidates:
        emitted = coordinator._selection(problem, op, unit)
        if emitted is None:
            return (), candidates, (), False
        output.extend(emitted)
        successes.append(bool(emitted))
    return tuple(output), candidates, tuple(successes), True
