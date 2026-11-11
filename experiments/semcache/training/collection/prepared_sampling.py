"""Validation and merging for per-query frozen sampling statistics.

The first rollout of a query owns the independent-base sampling budget. Later
rollouts import its immutable operator/source evidence into a ProfileEnvironment
while taking timing, accuracy, model and cache state from the current round.
"""

import copy

from planning.state import query_key

# These fields are observations of this query's independent sampling pass. All
# other profile fields are either runtime configuration or calibration that must
# come from the current global profile.
FROZEN_STATISTICS = (
    "operator_statistics",
    "node_statistics",
    "source_statistics",
    # Correlated terminal domains contain cardinality priors and deliberately
    # have no sampled values. Retain that distinction for time-estimate masks.
    "cardinality_only_sources",
    "adaptive_sketch",
    "base_sketch_plans",
    "prefix_cooccurrences",
)


def merge_prepared_statistics(query, current_profile, prepared):
    """Return a current-round profile with validated frozen query statistics.

    ``prepared`` is intentionally a plain JSON-compatible mapping, making it
    safe to pass across a resume boundary. Query identity is ignored while its
    normalized SQL/semantic structure is checked by :func:`query_key`.
    """
    if not isinstance(prepared, dict):
        raise ValueError("prepared_statistics must be a mapping")
    old_query = prepared.get("query")
    old_profile = prepared.get("profile")
    source = prepared.get("source")
    if not isinstance(old_query, dict) or not isinstance(old_profile, dict):
        raise ValueError("prepared_statistics requires query and profile mappings")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("prepared_statistics requires a nonempty source")
    if query_key(old_query) != query_key(query):
        raise ValueError("Prepared sampling query does not match current QueryIR")
    if old_profile.get("sampling_strategy") != "independent_base":
        raise ValueError(
            "Prepared sampling requires completed independent-base statistics"
        )
    for key in ("operator_statistics", "node_statistics", "source_statistics"):
        if not isinstance(old_profile.get(key), dict):
            raise ValueError(f"Prepared sampling is missing its {key} bank")
    merged = copy.deepcopy(current_profile)
    merged["sampling_strategy"] = "independent_base"
    for key in FROZEN_STATISTICS:
        if key in old_profile:
            merged[key] = copy.deepcopy(old_profile[key])
    # Pair estimates belong to this query's frozen input statistics. A missing
    # bank must not inherit another query's field identities from the globals.
    merged["prefix_cooccurrences"] = copy.deepcopy(
        old_profile.get("prefix_cooccurrences", {})
    )
    # Never carry a prior query's cache ledger or dynamic calibration through a
    # search boundary. The caller publishes the current history below.
    merged.pop("history", None)
    return merged


def prepared_descriptor(prepared):
    """Small artifact-safe descriptor; excludes profile/query payloads."""
    return {
        "reused": True,
        "source": prepared["source"],
        "http_requests": 0,
        "statistics": "independent_base_operator_node_source",
    }
