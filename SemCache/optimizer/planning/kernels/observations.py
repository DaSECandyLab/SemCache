"""Typed native observation kernels; Python retains serialization and execution.

There is deliberately no implicit Python fallback: a missing native build must
be fixed instead of silently moving search statistics back onto the interpreter.
"""

from planning.observations.evidence import _EvidenceDict, freeze_evidence

from . import _search_core

Weights = _search_core.ObservationWeights
effective = _search_core.observation_effective
interval = _search_core.observation_interval
field_rates = _search_core.observation_field_rates
summarize = _search_core.observation_summarize
SummaryWeights = _search_core.ObservationSummaryWeights


def summarize_values(rows, weights, serialize):
    return _search_core.observation_summarize_values(
        rows, weights, serialize, _EvidenceDict
    )


def observation(token, references, rows, statistics, complete, boundary):
    # The private bank is published once. Mutable state diagnostics are copied
    # at the compatibility boundary, not when descendants retain this handle.
    return _search_core.NativeObservation(
        token,
        references,
        rows,
        _EvidenceDict(
            {
                name: value if name in _SHARED else freeze_evidence(value)
                for name, value in statistics.items()
            }
        ),
        complete,
        (boundary[0], freeze_evidence(boundary[1])),
    )


_SHARED = frozenset(
    (
        "summary",
        "right_summary",
        "conditional_sketch_plan",
        "adaptive",
        "depth_survival_statistics",
        "depth_selectivity_statistics",
        "prefix_selectivity_statistics",
    )
)


def _mutable(value):
    if isinstance(value, dict):
        return {key: _mutable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_mutable(item) for item in value]
    if type(value) is tuple:
        return tuple(_mutable(item) for item in value)
    return value


def mutable_statistics(statistics):
    """Materialize compatibility fields while retaining published sketch banks."""
    from ..state import copy_statistics

    result = {
        name: value if name in _SHARED else _mutable(value)
        for name, value in statistics.items()
    }
    depths = {
        name: value
        for name, value in statistics.items()
        if name
        in (
            "depth_survival_statistics",
            "depth_selectivity_statistics",
            "prefix_selectivity_statistics",
        )
        and type(value) in (dict, list)
    }
    if depths:
        # Incomplete banks have mutable shells around frozen depth subtables.
        # Copy only mutable parts; expanding frozen islands repeats the grid work.
        result.update(copy_statistics({"depths": depths})["depths"])
    return result
