"""Deterministic dominance layers and bounded sampling at the cutoff layer."""

import math
import random


def pareto_fronts(scores):
    """All two-objective dominance layers in O(n log n), larger is better.

    Descending lexicographic order visits every dominator first. A Fenwick
    prefix maximum gives its deepest layer; identical points enter together.
    """
    points = {}
    for i, row in enumerate(scores):
        if len(row) != 2 or not all(math.isfinite(v) for v in row):
            raise ValueError("Ranking heads must return two finite scores")
        points.setdefault(tuple(row), []).append(i)
    ranks = {
        value: i + 1
        for i, value in enumerate(sorted({p[1] for p in points}, reverse=True))
    }
    tree = [0] * (len(ranks) + 1)
    fronts = []
    for point in sorted(points, reverse=True):
        pos, depth = ranks[point[1]], 0
        while pos:
            depth = max(depth, tree[pos])
            pos -= pos & -pos
        if depth == len(fronts):
            fronts.append([])
        fronts[depth].extend(points[point])
        pos = ranks[point[1]]
        while pos < len(tree):
            tree[pos] = max(tree[pos], depth + 1)
            pos += pos & -pos
    return [sorted(front) for front in fronts]


def trim(scores, limit, *, fronts=None, rng=None):
    """Fill Pareto layers, sampling without replacement at the cutoff layer."""
    if limit <= 0:
        return []
    rng = rng if rng is not None else random.Random(0)
    retained = []
    for front in pareto_fronts(scores) if fronts is None else fronts:
        slots = limit - len(retained)
        if len(front) <= slots:
            retained.extend(front)
        else:
            retained.extend(rng.sample(front, slots))
            break
        if len(retained) == limit:
            break
    return retained
