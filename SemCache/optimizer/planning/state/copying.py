"""Branch-local copies that preserve published immutable evidence."""

import copy
from typing import Any


def _copy_depth_evidence(value, memo) -> Any:
    """Clone plain sampling tables without deepcopy dispatch for every scalar.

    Mutable containers remain independent. The common memo preserves aliases
    with other statistics; extension objects retain normal deepcopy behavior.
    """
    identity = id(value)
    if identity in memo:
        return memo[identity]
    kind = type(value)
    if kind in (str, int, float, bool, type(None)):
        return value
    if kind is dict:
        result = {}
        memo[identity] = result
        for key, item in value.items():
            result[key if type(key) is str else copy.deepcopy(key, memo)] = (
                _copy_depth_evidence(item, memo)
            )
        return result
    if kind is list:
        result = []
        memo[identity] = result
        result.extend(_copy_depth_evidence(item, memo) for item in value)
        return result
    return copy.deepcopy(value, memo)


def _statistics_memo(rows):
    rows = [info for info in rows if isinstance(info, dict)]
    immutable = (
        "summary",
        "right_summary",
        "conditional_sketch_plan",
        "adaptive",
    )
    memo = {
        id(info[name]): info[name]
        for info in rows
        for name in immutable
        if name in info
    }
    for info in rows:
        for name in (
            "depth_survival_statistics",
            "depth_selectivity_statistics",
            "prefix_selectivity_statistics",
            "input_field_statistics",
            "output_field_statistics",
        ):
            if name in info:
                _copy_depth_evidence(info[name], memo)
    return memo


def copy_statistics(values):
    """Copy branch-local estimates while sharing published read-only sketches.

    Builders replace complete payloads; they never modify a published summary
    or sketch plan. Frozen depth evidence shares its own immutable containers;
    plain field/depth tables, intervals, reliability and cache edges are copied
    independently for each search branch.
    """
    return _copy_depth_evidence(values, _statistics_memo(values.values()))


def copy_profile(profile):
    """Preserve published statistics payloads across profile/replay copies.

    Only known statistics banks and history entries contain published summaries;
    identically named fields in catalogs or other metadata remain independently
    copied. Prefix and conditional banks add one history-keyed nesting level.
    """
    banks = [
        profile.get("statistics", {}),
        profile.get("node_statistics", {}),
        profile.get("operator_statistics", {}),
    ]
    for name in ("prefix_statistics", "conditional_inputs"):
        banks.extend(profile.get(name, {}).values())
    rows = [info for bank in banks for info in bank.values()]
    rows.extend(profile.get("history", []))
    return copy.deepcopy(profile, _statistics_memo(rows))
