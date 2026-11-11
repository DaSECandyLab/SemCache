"""Shared typed API-reference agreement metrics.

The same comparator is used by calibration and the external result scorer.  It
deliberately scores rows by stable identity first, then scores values only for
rows present on both sides.  This prevents a wrong row from being hidden by a
good text value and gives partial credit to a semantically close field value.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from planning.state import canonical

_TOKENS = re.compile(r"\w+", re.UNICODE)


def _number_score(left, right, *, absolute=1e-6, relative=1e-6):
    if isinstance(left, bool) or isinstance(right, bool):
        return float(left is right)
    try:
        left, right = float(left), float(right)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(left) or not math.isfinite(right):
        return float(left == right)
    return float(abs(left - right) <= max(absolute, relative * abs(right)))


def text_f1(left, right):
    """Case-insensitive Unicode word-multiset F1; punctuation separates tokens."""
    left = " ".join(str(left).casefold().split())
    right = " ".join(str(right).casefold().split())
    if left == right:
        return 1.0
    a, b = Counter(_TOKENS.findall(left)), Counter(_TOKENS.findall(right))
    if not a and not b:
        return 1.0
    overlap = sum((a & b).values())
    return 2.0 * overlap / (sum(a.values()) + sum(b.values())) if overlap else 0.0


def value_score(left, right):
    """Score one typed output value in [0, 1]."""
    if left is None or right is None:
        return float(left is right)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return _number_score(left, right)
    if isinstance(left, bool) or isinstance(right, bool):
        return float(left == right)
    if isinstance(left, dict) and isinstance(right, dict):
        keys = set(left) | set(right)
        return (
            sum(value_score(left.get(k), right.get(k)) for k in keys) / len(keys)
            if keys
            else 1.0
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        left_counts = Counter(canonical(v) for v in left)
        right_counts = Counter(canonical(v) for v in right)
        common = sum((left_counts & right_counts).values())
        denominator = sum(left_counts.values()) + sum(right_counts.values())
        return 2.0 * common / denominator if denominator else 1.0
    return text_f1(left, right)


def compare_atomic(kind, actual, reference):
    """Return protocol metrics for one semantic operator evaluation.

    ``actual`` and ``reference`` map stable output row identities to typed
    output values.  Filter and join are row-set metrics; projection also gets
    a field score for matched rows.  ``value`` is the scalar used by planning.
    """
    actual, reference = dict(actual), dict(reference)
    actual_ids, reference_ids = set(actual), set(reference)
    common = actual_ids & reference_ids
    tp, fp, fn = (
        len(common),
        len(actual_ids - reference_ids),
        len(reference_ids - actual_ids),
    )
    row_f1 = 1.0 if tp == fp == fn == 0 else (2.0 * tp / (2.0 * tp + fp + fn))
    if not common:
        attribute = 1.0 if not actual_ids and not reference_ids else 0.0
    else:
        attribute = sum(value_score(actual[i], reference[i]) for i in common) / len(
            common
        )
    if str(kind).upper() in {"SEMANTIC_FILTER", "SEMANTIC_JOIN"}:
        value = row_f1
        attribute = None
    else:
        value = (row_f1 + attribute) / 2.0
    exact = int(actual == reference)
    return {
        "value": value,
        "row_f1": row_f1,
        "attribute_score": attribute,
        "exact": exact,
    }
