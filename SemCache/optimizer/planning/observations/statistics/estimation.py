"""Compatibility-checked cache edge estimation over decoded sketches."""

from collections import OrderedDict

from planning.kernels.cache import MatchCounts, join_self, reduce_cross, self_reuse
from planning.observations.cache import (
    PrefixSummary,
    _cross_sample,
    cross_reuse,
    survival_rate,
)
from planning.state import action_key, copy_statistics

from .serialization import FIELD_SERIALIZER, prefix, prefix_length


class _SummaryDecoder:
    """Bound decoding of published read-only payloads within one problem.

    Raw payload references prevent recycled object identities from hitting an
    unrelated summary. Public PrefixSummary.load continues to validate and
    return an independent mutable instance on every call.
    """

    def __init__(self, capacity=512, max_entries=262144):
        self.capacity, self.max_entries = capacity, max_entries
        self.entries = OrderedDict()
        self.entry_count = 0
        self.matches = OrderedDict()
        self.match_entry_count = 0

    def load(self, raw):
        identity = id(raw)
        cached = self.entries.get(identity)
        if cached is not None:
            self.entries.move_to_end(identity)
            return cached[1]
        summary = PrefixSummary.load(raw)
        count = sum(len(s["entries"]) for s in raw["sketches"].values())
        if count <= self.max_entries:
            while self.entries and (
                len(self.entries) >= self.capacity
                or self.entry_count + count > self.max_entries
            ):
                _, (_, _, old_count) = self.entries.popitem(last=False)
                self.entry_count -= old_count
            self.entries[identity] = (raw, summary, count)
            self.entry_count += count
        return summary

    def match_terms(self, source, target, depth):
        """Retain only survival-independent counts from immutable decoded pairs.

        Depths are prepared on demand so an earlier unknown/invalid survival
        never evaluates a later depth. Pair keys strongly retain both summaries;
        their full sketch entry counts bound retained payloads conservatively.
        """
        key = (source, target)
        cached = self.matches.get(key)
        if cached is not None:
            self.matches.move_to_end(key)
            depths = cached[0]
            if depth in depths:
                return depths[depth]
        else:
            depths = None
        left, right = source.sketches[depth], target.sketches[depth]
        tau, common = _cross_sample(left, right)
        terms = tau, MatchCounts((left.entries[p], right.entries[p]) for p in common)
        if depths is None:
            count = sum(
                len(sketch.entries)
                for summary in (source, target)
                for sketch in summary.sketches.values()
            )
            if count > self.max_entries:
                return terms
            while self.matches and (
                len(self.matches) >= self.capacity
                or self.match_entry_count + count > self.max_entries
            ):
                _, (_, old_count) = self.matches.popitem(last=False)
                self.match_entry_count -= old_count
            depths = {}
            self.matches[key] = (depths, count)
            self.match_entry_count += count
        depths[depth] = terms
        return terms


def _compatible_edge(
    source_info,
    target,
    alpha_source,
    alpha_target,
    *,
    join_right=None,
    decoder=None,
    coupling=None,
    requests=None,
):
    """Estimate one source-to-target edge, then normalize a join's first field."""
    if not source_info.get("summary"):
        return None
    source = (
        decoder.load(source_info["summary"])
        if decoder is not None
        else PrefixSummary.load(source_info["summary"])
    )
    if not source.compatible(target):
        return None
    value = cross_reuse(
        source,
        target,
        alpha_source,
        alpha_target,
        coupling=coupling,
        requests=requests,
        _terms=decoder.match_terms if decoder is not None else None,
        _reduce=reduce_cross if decoder is not None else None,
    )
    if value is None:
        return None
    if requests is None and join_right is not None:
        return value / join_right if join_right else 0.0
    return value


def _previous_info(problem, state, index):
    """Prefer the observed branch over a profile from an older execution."""
    key = state._action_keys[index]
    observed = state.statistics.get(key)
    if observed:
        return observed
    if problem.profile.get("sampling_strategy") == "independent_base":
        return problem.profile.get("statistics", {}).get(key, {})
    return {}


def input_statistics(problem, state, action, *, copy_observation=True):
    """Merge semantic evidence before final cardinalities and cache reductions."""
    oid = action["operator_id"]
    key = (
        state._action_keys_by_operator[oid]
        if action is state._action_by_operator.get(oid)
        else action_key(action)
    )
    observed = state.statistics.get(key, {})
    profile = {}
    if problem.profile.get("sampling_strategy") == "independent_base":
        profile = {
            **problem.profile.get("operator_statistics", {}).get(str(oid), {}),
            **problem.profile.get("statistics", {}).get(key, {}),
        }
    if copy_observation:
        result = copy_statistics({key: {**profile, **observed}})[key]
    else:
        # A synchronous refresh keeps evidence already owned by this state.
        # New profile metadata must still be isolated from its source bank.
        result = {
            **copy_statistics(
                {
                    key: {
                        name: value
                        for name, value in profile.items()
                        if name not in observed
                    }
                }
            )[key],
            **observed,
        }
    return result


def estimate(
    problem,
    state,
    action,
    *,
    partial_plans=None,
    copy_observation=True,
    statistics=None,
):
    """Derive cache scalars only after the target request counts are final."""
    result = (
        input_statistics(problem, state, action, copy_observation=copy_observation)
        if statistics is None
        else statistics
    )
    # Native history reductions belong to the snapshot that produced them.
    # Python replay derives cache edges in a new history context; let feature
    # construction reduce those fresh edges instead of retaining the old vector.
    result.pop("l_existing_features", None)
    if (
        len(action.get("inputs", [])) == 1
        and "join" in problem.ops[action["operator_id"]]["kind"].lower()
        and not result.get("pair_request_summary")
    ):
        # A base-side summary alone cannot represent a sparse request multiset.
        # Native feedback supplies explicit conditional per-request summaries.
        result.update(
            l_self=None,
            l_existing={},
            l_previous={},
            pair_input=True,
            cache_sources_complete=False,
        )
        return result
    if result.get("summary") is None:
        result["cache_sources_complete"] = False
        return result
    if (
        result.get("pre") is None
        and hasattr(problem, "ops")
        and "default_layout" in problem.ops.get(action["operator_id"], {})
    ):
        op = problem.ops[action["operator_id"]]
        result["pre"] = prefix(
            op,
            action,
            problem.profile.get("system_prefix", "semantic-global-v1"),
            problem.profile.get("operator_specifications", {}).get(
                op["kind"], op["kind"]
            ),
        )
        result["pre_length"] = prefix_length(result["pre"])
    decoder = getattr(problem, "_summary_decoder", None)
    if decoder is None:
        decoder = problem._summary_decoder = _SummaryDecoder()
    current = decoder.load(result["summary"])
    if current.serializer != FIELD_SERIALIZER:
        raise ValueError(
            "Legacy input summary: regenerate conditional profiles from execution telemetry"
        )
    alpha = result.get("alpha", 1.0)
    # Sparse pairs use frequencies of actual requests, rather than a fictitious
    # dense left/right product. This estimates first-field reuse conservatively.
    right = (
        decoder.load(result["right_summary"])
        if result.get("right_summary") and not result.get("pair_input")
        else None
    )
    if right is not None and not right.compatible(current):
        raise ValueError("Join field summaries must use compatible coordinates")
    result["l_self"] = (
        join_self(
            current,
            right,
            result.get("left_total_length"),
            alpha,
            result.get("right_alpha", 1.0),
            left_requests=result.get("input_side_rows", [None, None])[0],
            right_requests=result.get("input_side_rows", [None, None])[1],
        )
        if right
        else self_reuse(current, alpha, requests=result.get("input_rows"))
    )
    result["l_existing"], result["l_previous"] = {}, {}
    result["cache_sources_complete"] = result.get("pre") is not None
    right_rate = survival_rate(result.get("right_alpha", 1.0)) if right else None
    if right and right_rate is None:
        result["cache_sources_complete"] = False
        return result
    right_count = (
        right.count * right_rate
        if right is not None and right_rate is not None
        else None
    )
    for source in getattr(problem, "cache_history", problem.profile.get("history", [])):
        if source["model_id"] != action["model_id"]:
            continue
        if source.get("pre") is None:
            result["cache_sources_complete"] = False
        elif source["pre"] == result.get("pre"):
            value = _compatible_edge(
                source,
                current,
                1.0,
                alpha,
                join_right=right_count,
                decoder=decoder,
                requests=result.get("input_rows"),
            )
            result["l_existing"][source["id"]] = value
    sources = (
        problem.cache_sources(state, action, partial_plans)
        if partial_plans is not None
        else problem.cache_sources(state, action)
    )
    for index, old in enumerate(state.action_views):
        if old["operator_id"] not in sources or old["model_id"] != action["model_id"]:
            continue
        old_info = _previous_info(problem, state, index)
        if old_info.get("pre") is None:
            result["cache_sources_complete"] = False
        if result.get("pre") is not None and old_info.get("pre") == result["pre"]:
            value = _compatible_edge(
                old_info,
                current,
                old_info.get("alpha", 1.0),
                alpha,
                join_right=right_count,
                decoder=decoder,
                coupling=(
                    {"kind": "same_field"}
                    if result.get("cache_field_origin")
                    and result.get("cache_field_origin")
                    == old_info.get("cache_field_origin")
                    else getattr(problem, "_prefix_coupling", lambda *args: None)(
                        old_info, result
                    )
                ),
                requests=result.get("input_rows"),
            )
            result["l_previous"][str(old["operator_id"])] = value
    return result
