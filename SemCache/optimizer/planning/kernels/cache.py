"""Lazy owned numeric terms for problem-private, immutable decoded summaries.

Public mutable summaries keep the original validation and reduction path. Only
validated builtin frequencies enter native buffers; exceptional values preserve
the Python arithmetic and exception order.
"""

from planning.observations.cache import integrate, present, survival_rate


def safe_count(value):
    return type(value) is int and 0 < value <= 2**53


class MatchCounts(tuple):
    """Reference-compatible count pairs with one lazy native numeric buffer."""

    def evaluate(self, source, target, tau):
        from . import _search_core

        native = getattr(self, "_native", None)
        if native is None:
            if not all(safe_count(a) and safe_count(b) for a, b in self):
                return (
                    sum(present(source, a) * present(target, b) for a, b in self) / tau
                )
            native = self._native = _search_core.CrossTerms(self, tau)
        return native.evaluate(source, target)


def self_reuse(summary, alpha, *, requests=None):
    """Same per-depth short circuits, integration and denominator as cache.py."""
    from planning.observations.cache import self_reuse as reference

    from . import _search_core

    if requests is not None or type(summary.count) is not int or summary.count < 0:
        return reference(summary, alpha, requests=requests)
    rate = survival_rate(alpha)
    if rate is None:
        return None if summary.count else 0.0
    denominator = rate * summary.count
    if denominator == 0:
        return 0.0
    terms = getattr(summary, "_native_self_terms", None)
    if terms is None:
        terms = summary._native_self_terms = {}
    values = {}
    for depth, sketch in summary.sketches.items():
        rate = survival_rate(alpha, depth)
        cached = terms.get(depth)
        if cached is None:
            tau, sample = sketch.sample()
        else:
            term, present_sample = cached
            sample = None
        if rate is None:
            if bool(sample) if cached is None else present_sample:
                return None
            rate = 0.0
        if cached is None:
            assert sample is not None
            counts = tuple(sample.values())
            if not all(safe_count(count) for count in counts):
                return reference(summary, alpha)
            term = _search_core.SelfTerms(counts, tau)
            terms[depth] = term, bool(sample)
        values[depth] = term.evaluate(rate)
    return integrate(values) / denominator


def join_self(
    left,
    right,
    left_total_length,
    alpha_left,
    alpha_right,
    *,
    left_requests=None,
    right_requests=None,
):
    if left_requests is not None or right_requests is not None:
        from planning.observations.cache import join_self as reference

        return reference(
            left,
            right,
            left_total_length,
            alpha_left,
            alpha_right,
            left_requests=left_requests,
            right_requests=right_requests,
        )
    left_rate, right_rate = survival_rate(alpha_left), survival_rate(alpha_right)
    if left_rate == 0 or right_rate == 0 or not left.count or not right.count:
        return 0.0
    if left_rate is None or right_rate is None:
        return None
    left_count, right_count = left.count * left_rate, right.count * right_rate
    if not left_count or not right_count:
        return 0.0
    mean_left = (
        left.total_length if left_total_length is None else left_total_length
    ) / left.count
    left_reuse, right_reuse = (
        self_reuse(left, alpha_left),
        self_reuse(right, alpha_right),
    )
    if left_reuse is None or right_reuse is None:
        return None
    return (
        left_reuse / right_count
        + max(0, right_count - 1) / right_count * mean_left
        + right_reuse
    )


def reduce_cross(counts, source, target, tau):
    return counts.evaluate(source, target, tau)
