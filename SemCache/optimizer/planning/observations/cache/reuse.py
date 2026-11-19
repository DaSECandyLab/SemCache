"""Overlap, conditional survival and self/cross reuse estimators."""

import itertools
import math
import sys
from typing import Any

from .sketches import decode_prefix_hash, require_hash_entries


def present(alpha, count):
    if not 0 <= alpha <= 1:
        raise ValueError("Survival probability must be in [0,1]")
    return 1.0 if alpha == 1 else -math.expm1(count * math.log1p(-alpha))


def repeated_prefix_hits(alpha, count):
    """§3.1.1 binomial repeats, without cancelling a small positive result."""
    if not alpha or count <= 1:
        return 0.0
    if alpha == 1:
        return count - 1.0
    visits = alpha * count
    if visits >= 0.125:
        return max(0.0, visits + math.expm1(count * math.log1p(-alpha)))
    term = visits * ((count - 1) * alpha) / 2
    total, k = term, 2
    while term:
        term *= -((count - k) * alpha) / (k + 1)
        updated = total + term
        if updated == total:
            break
        total, k = updated, k + 1
    return max(0.0, total)


def integrate(values, max_observed=None):
    if max_observed is not None:
        if max_observed == 0:
            return 0.0
        limit = max_observed + 1
        if values and max(values) > limit:
            values = {k: v for k, v in values.items() if k <= limit}
            values[limit] = 0.0
    if not values:
        return 0.0
    ordered = sorted(values)
    total = values[ordered[0]]
    for left, right in itertools.pairwise(ordered):
        gap = right - left
        total += (gap - 1) * values[left] / 2 + (gap + 1) * values[right] / 2
    return total


def _cross_observed_max(source, target):
    lengths = [
        x for x in (source.observed_max_length, target.observed_max_length) if x is not None
    ]
    return min(lengths) if lengths else None


def survival_rate(alpha, depth=None):
    if isinstance(alpha, dict):
        value = (
            alpha["total"]
            if depth is None
            else alpha.get("depths", {}).get(str(depth), alpha["total"])
        )
    else:
        value = alpha
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Survival probability must be finite and in [0,1]")
    if 0 <= value <= 1:
        return value
    # Ratios of independently rounded weighted sums can land on the adjacent
    # float outside an endpoint. Preserve that valid endpoint, not an unknown
    # or an invented zero; genuinely invalid probabilities still fail.
    if math.nextafter(0.0, -math.inf) <= value <= math.nextafter(1.0, math.inf):
        return min(1.0, max(0.0, value))
    raise ValueError("Survival probability must be finite and in [0,1]")


def self_reuse(
    summary, alpha: float | dict[str, Any] = 1.0, *, requests=None
) -> float | None:
    rate = survival_rate(alpha)
    if rate is None:
        return None if summary.count else 0.0
    distinct = rate * summary.count
    denominator = distinct if requests is None else requests
    if denominator == 0:
        return 0.0
    if not distinct:
        return None
    scale = denominator / distinct
    values = {}
    for depth, sketch in summary.sketches.items():
        rate = survival_rate(alpha, depth)
        tau, sample = sketch.sample()
        if rate is None:
            if sample:
                return None
            rate = 0.0
        values[depth] = (
            sum(
                max(
                    0.0,
                    repeated_prefix_hits(scale * rate, c)
                    if scale < 1.0
                    else repeated_prefix_hits(rate, c) + (scale - 1) * rate * c,
                )
                for c in sample.values()
            )
            / tau
        )
    return integrate(values, summary.observed_max_length) / denominator


def _cross_sample(left, right):
    if (
        left.complete is True
        and right.complete is True
        and all(
            type(sketch.capacity) is int and 2 <= sketch.capacity <= sys.maxsize
            for sketch in (left, right)
        )
        and all(
            type(prefix) is int
            and 0 <= prefix < 2**256
            and type(count) is int
            and count > 0
            for sketch in (left, right)
            for prefix, count in sketch.entries.items()
        )
    ):
        # A complete union has tau=1; mutable or invalid entries use merge validation.
        tau, sample = 1.0, {**left.entries, **right.entries}
    else:
        tau, sample = left.merge(right).sample()
    common = set(sample) & set(left.entries) & set(right.entries)
    return tau, common


def joint_presence(source, target, source_count, target_count, common_count):
    """Nested same-row survivals, with independent rows as an explicit approximation.

    Each depth supplies its own rates. Common rows are measured by shared row
    identity, never inferred from hash-bucket overlap.
    """
    if not 0 <= common_count <= min(source_count, target_count):
        raise ValueError("Common frequency must be between zero and both marginals")
    a, b = present(source, source_count), present(target, target_count)
    correction = (
        (1.0 - a)
        * (
            1.0
            if target_count == common_count
            else 1.0 - present(target, target_count - common_count)
        )
        * (present(target, common_count) if common_count else 0.0)
    )
    return min(a, b, max(max(0.0, a + b - 1.0), a * b + correction))


def cross_reuse(
    source,
    target,
    alpha_source: float | dict[str, Any] = 1.0,
    alpha_target: float | dict[str, Any] = 1.0,
    *,
    coupling=None,
    requests=None,
    _terms=None,
    _reduce=None,
) -> float | None:
    if not source.compatible(target):
        raise ValueError("Cross-source summaries must use compatible coordinates")
    rate = survival_rate(alpha_target)
    count = target.count
    if rate is None:
        return None if count else 0.0
    distinct = rate * count
    denominator = distinct if requests is None else requests
    if denominator <= 0:
        return 0.0
    if requests is not None and distinct > 0 and denominator < distinct:
        denominator = distinct
    values = {}
    if coupling and coupling.get("kind") == "same_field":
        for depth, sketch in target.sketches.items():
            tau, sample = sketch.sample()
            alpha = survival_rate(alpha_target, depth)
            if alpha is None and sample:
                return None
            values[depth] = min(
                denominator,
                sum(present(alpha or 0.0, c) for c in sample.values()) / tau,
            )
        return integrate(values, target.observed_max_length) / denominator
    for depth in sorted(source.sketches.keys() & target.sketches.keys()):
        left, right = source.sketches[depth], target.sketches[depth]
        counts, total = None, 0.0
        if _terms is None or coupling:
            tau, common = _cross_sample(left, right)
        else:
            tau, counts = _terms(source, target, depth)
            common = counts
        rate_source, rate_target = (
            survival_rate(alpha_source, depth),
            survival_rate(alpha_target, depth),
        )
        if rate_source is None or rate_target is None:
            if common:
                return None
            values[depth] = 0.0
            continue
        if counts is None:
            overlap = {}
            if coupling and coupling.get("kind") == "same_relation":
                raw_common = coupling["common"]
                require_hash_entries(raw_common)
                overlap = {
                    decode_prefix_hash(h): c
                    for h, c in raw_common.get(str(depth), {}).items()
                }
            total = sum(
                joint_presence(
                    rate_source,
                    rate_target,
                    left.entries[p],
                    right.entries[p],
                    overlap.get(p, 0.0),
                )
                if coupling and coupling.get("kind") == "same_relation"
                else present(rate_source, left.entries[p])
                * present(rate_target, right.entries[p])
                for p in common
            )
        elif _reduce is None:
            total = sum(
                present(rate_source, source_count) * present(rate_target, target_count)
                for source_count, target_count in counts
            )
        values[depth] = min(
            denominator,
            _reduce(counts, rate_source, rate_target, tau)
            if _reduce is not None and counts is not None
            else total / tau,
        )
    return integrate(values, _cross_observed_max(source, target)) / denominator


def join_self(
    left,
    right,
    left_total_length=None,
    alpha_left: float | dict[str, Any] = 1.0,
    alpha_right: float | dict[str, Any] = 1.0,
    *,
    left_requests=None,
    right_requests=None,
) -> float | None:
    left_rate, right_rate = survival_rate(alpha_left), survival_rate(alpha_right)
    if left_rate == 0 or right_rate == 0 or not left.count or not right.count:
        return 0.0
    if left_rate is None or right_rate is None:
        return None
    left_count = left.count * left_rate if left_requests is None else left_requests
    right_count = right.count * right_rate if right_requests is None else right_requests
    if left_count <= 0 or right_count <= 0:
        return 0.0
    eff_right = max(1.0, right_count)
    mean_left = (
        left.total_length if left_total_length is None else left_total_length
    ) / left.count
    left_reuse, right_reuse = (
        self_reuse(left, alpha_left, requests=left_requests),
        self_reuse(right, alpha_right, requests=right_requests),
    )
    if left_reuse is None or right_reuse is None:
        return None
    return (
        left_reuse / eff_right
        + (eff_right - 1.0) / eff_right * mean_left
        + right_reuse
    )
