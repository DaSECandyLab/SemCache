"""Extract serialized-field summaries from actual requests and construct cache edges."""

import itertools
import json
import math
import re
from collections import OrderedDict, defaultdict
from functools import lru_cache

from semantic_events import logical_rows

from planning.kernels.cache import MatchCounts, join_self, reduce_cross, self_reuse
from planning.observations.cache import (
    PrefixSummary,
    _cross_sample,
    cross_reuse,
    survival_rate,
)
from planning.state import action_key, canonical, copy_statistics, digest

FIELD_SERIALIZER = "prompt-v4-first-field-value-v1"


def prefix_length(namespace):
    """Characters of system/user content before the first Data Block.

    The namespace also encodes serializer/version and JSON escapes. Those
    identity bytes are not part of the prompt length feature.
    """
    if namespace is None:
        return None
    try:
        parts = json.loads(namespace)
    except (TypeError, ValueError):
        return None
    if (
        not isinstance(parts, list)
        or len(parts) != 3
        or not all(isinstance(p, str) for p in parts)
    ):
        return None
    return len(parts[1]) + len(parts[2].split("<DATA_BLOCK>", 1)[0])


def prefix(op, action, system, operator_spec):
    layout = action["layout"]
    fields = layout.get(
        "field_order",
        layout.get("left_field_order", []) + layout.get("right_field_order", []),
    )
    inputs = (
        op["input_fields"]
        if op["input_fields"]
        else op["left_input_fields"] + op["right_input_fields"]
    )
    inverse = {inputs.index(f): i for i, f in enumerate(fields)}
    task = re.sub(
        r"\{(\d+)\}", lambda m: "{" + str(inverse[int(m[1])]) + "}", op["task"]
    )
    blocks = {"OPERATOR_SPECIFICATION": operator_spec, "TASK_DESCRIPTION": task}
    parts = []
    for block in layout["block_order"]:
        if block == "DATA_BLOCK":
            break
        parts.append(f"<{block}>\n{blocks[block]}\n</{block}>")
    # Global system role/content and v4 serialization remain in the namespace.
    # Do not include the full prompt length: it changes with each data value.
    return canonical(
        [
            "prompt-v4",
            system,
            "\n\n".join(parts) + ("\n\n" if parts else "") + "<DATA_BLOCK>\n" + "{0}: ",
        ]
    )


def request_inputs(op, rows):
    """Recover original input-expression identities from dense display numbering."""
    recovered = []
    expected = len(
        op["input_fields"] or op["left_input_fields"] + op["right_input_fields"]
    )
    for row in rows:
        # 原始绑定顺序与 prompt 展示顺序不同；结构化字段避免任务文本中的标签
        # 被误读成 DATA_BLOCK，也让字段重排后的统计仍对应同一输入表达式。
        values = row.get("input_values")
        if (
            not isinstance(values, list)
            or len(values) != expected
            or not all(isinstance(v, str) for v in values)
        ):
            raise ValueError(
                "Structured telemetry inputs do not match bound fields; regenerate old telemetry"
            )
        recovered.append(dict(enumerate(values)))
    return recovered


def _field_row_indices(op, rows):
    """A join profiles each distinct left/right tuple once, not every pair."""
    if op["input_fields"]:
        return {field: range(len(rows)) for field in op["input_fields"]}
    indices = {}
    for side in ("left", "right"):
        seen, selected = set(), []
        for index, row in enumerate(rows):
            identity = row[f"{side}_index"]
            if identity not in seen:
                seen.add(identity)
                selected.append(index)
        indices.update(dict.fromkeys(op[f"{side}_input_fields"], selected))
    return indices


class FieldSummaryBank:
    """Decode once, tune once, build each original field once per input boundary."""

    def __init__(
        self,
        op,
        rows,
        capacity=None,
        max_length=None,
        adaptive=None,
        *,
        pair_input=False,
    ):
        from planning.kernels.sketches import adaptive_build

        self.op, self.rows = op, rows
        self.records = request_inputs(op, rows)
        self.inputs = (
            op["input_fields"] or op["left_input_fields"] + op["right_input_fields"]
        )
        self.pair_input = pair_input
        indices = (
            {field: range(len(rows)) for field in self.inputs}
            if pair_input
            else _field_row_indices(op, rows)
        )
        fields = {
            f: [self.records[i][j] for i in indices[f]]
            for j, f in enumerate(self.inputs)
        }
        built = (
            adaptive_build(fields, adaptive)
            if capacity is None and max_length is None
            else None
        )
        self.plan = built["plan"] if built else None
        self.summaries, self.raw = {}, {}
        self.left_total = sum(
            len(s) for f in op.get("left_input_fields", []) for s in fields[f]
        )
        for f, strings in fields.items():
            summary = (
                PrefixSummary.load(built["summaries"][f])
                if built
                else PrefixSummary.build(
                    strings,
                    128 if capacity is None else capacity,
                    4096 if max_length is None else max_length,
                )
            )
            summary.serializer = FIELD_SERIALIZER
            self.summaries[f] = summary
            self.raw[f] = summary.json()

    def info(self, action):
        layout = action["layout"]
        first = layout.get("field_order", layout.get("left_field_order"))[0]
        right = (
            layout["right_field_order"][0] if "right_field_order" in layout else None
        )
        prompt_prefix = prefix(
            self.op,
            action,
            self.rows[0]["system_text"],
            self.rows[0]["operator_specification"],
        )
        return {
            "summary": self.raw[first],
            "right_summary": self.raw[right] if right else None,
            "left_total_length": self.left_total if right else None,
            "pre": prompt_prefix,
            "pre_length": prefix_length(prompt_prefix),
            "first_field": first,
            "alpha": 1.0,
            "pair_input": self.pair_input,
            "pair_request_summary": self.pair_input,
            "adaptive": self.plan,
            "basis": "actual conditional input requests; character-domain ideal potential",
        }


def execution_rows(events):
    grouped = defaultdict(list)
    for event in logical_rows(events):
        grouped[event["operator_id"]].append(event)
    return grouped


def from_execution(problem, events, capacity=None, max_length=None):
    """Build profiles for the observed prefix, including alternative layouts only.

    Re-rendering field order is valid without inference: layouts don't change this
    operator's INPUT. Its output/selectivity/accuracy is not copied to alternatives.
    Profiles from full executions are for future runs, never retroactive features
    on the same candidate that generated them (which would leak future observations).
    """
    if any(e["event"] == "runtime_initialized" for e in events):
        raise ValueError(
            "Static input profiles cannot combine runtime configuration generations"
        )
    decision = next(e for e in events if e["event"] == "planning_decision")
    conditional = {}
    prefix_actions = []
    grouped = execution_rows(events)
    if getattr(problem, "native", False):
        states = problem.replay(decision["plan_spec"])
        boundaries = [
            (s.actions[-1], previous.history)
            for previous, s in itertools.pairwise(states)
            if s.actions[-1]["operator_id"] in problem.ops
        ]
    else:
        boundaries = []
        for action in decision["plan_spec"]["actions"]:
            boundaries.append((action, tuple(prefix_actions)))
            prefix_actions.append(canonical(action))
    for action, history_prefix in boundaries:
        op = problem.ops[action["operator_id"]]
        rows = grouped[action["operator_id"]]
        if not rows:
            continue
        values = {}
        bank = FieldSummaryBank(
            op,
            rows,
            capacity,
            max_length,
            problem.profile.get("adaptive_sketch"),
            pair_input=len(action.get("inputs", [])) == 1
            and "join" in op["kind"].lower(),
        )
        tokens = [e.get("output_tokens", -1) for e in rows]
        output_tokens = (
            sum(tokens) / len(tokens)
            if all(t is not None and t >= 0 for t in tokens)
            else None
        )
        emitted = sum(e["emitted"] for e in rows)
        executed = {k: v for k, v in action.items() if k != "inputs"}
        for candidate in problem.domain[action["operator_id"]]:
            info = bank.info(candidate)
            info.update(
                input_rows=len(rows),
                conditional_observation=True,
                output_tokens=output_tokens if candidate == executed else None,
            )
            if candidate == executed:
                info.update(output_rows=emitted, selectivity=emitted / len(rows))
            candidate_key = (
                dict(candidate, inputs=action["inputs"])
                if "inputs" in action
                else candidate
            )
            values[action_key(candidate_key)] = info
        conditional[digest(history_prefix)] = values
    return conditional


class _SummaryDecoder:
    """Bound decoding of published read-only payloads within one problem.

    Raw payload references prevent recycled object identities from hitting an
    unrelated summary. Public PrefixSummary.load continues to validate and
    return an independent mutable instance on every call.
    """

    def __init__(self, capacity=512, max_entries=65536):
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


def record_history(
    history,
    op,
    action,
    rows,
    capacity=None,
    max_length=None,
    *,
    started_at=None,
    counters_before=None,
    update_load=True,
    adaptive=None,
):
    # History retains the first request field, including repeated join rows.
    # Alternative-layout banks are useful for profiles, but are discarded here.
    layout = action["layout"]
    first = layout.get("field_order", layout.get("left_field_order"))[0]
    inputs = op["input_fields"] or op["left_input_fields"] + op["right_input_fields"]
    index = inputs.index(first)
    strings = [values[index] for values in request_inputs(op, rows)]
    if capacity is None and max_length is None:
        from planning.kernels.sketches import adaptive_build

        built = adaptive_build({first: strings}, adaptive)
        summary = PrefixSummary.load(built["summaries"][first])
    else:
        summary = PrefixSummary.build(
            strings,
            128 if capacity is None else capacity,
            4096 if max_length is None else max_length,
        )
    summary.serializer = FIELD_SERIALIZER
    history.record(
        action["model_id"],
        prefix(op, action, rows[0]["system_text"], rows[0]["operator_specification"]),
        digest([op["kind"], op["task"], op["output_schema"]]),
        first,
        summary,
        now=started_at,
        requests=len(rows),
        characters=sum(len(r["system_text"]) + len(r["user_text"]) for r in rows),
        update_load=update_load,
        **(
            {
                "requests_before": counters_before[0],
                "characters_before": counters_before[1],
            }
            if counters_before is not None
            else {}
        ),
    )


def _confirmed_request(event):
    """A model response proves execution even when output validation failed."""
    return (
        event["event"] == "semantic_row"
        and (
            event.get("parse_status") == "validated"
            or (
                isinstance(event.get("response_ns"), int)
                and event["response_ns"] >= 0
                and isinstance(event.get("raw_result"), str)
            )
        )
        and isinstance(event.get("system_text"), str)
        and isinstance(event.get("user_text"), str)
        and isinstance(event.get("operator_specification"), str)
        and isinstance(event.get("input_values"), list)
        and all(isinstance(value, str) for value in event["input_values"])
    )


def ingest_history(events, history, capacity=None, max_length=None, *, adaptive=None):
    """Record confirmed requests, including responses that failed output parsing."""
    if any(e["event"] == "runtime_initialized" for e in events):
        raise ValueError("Static history ingestion does not mix runtime generations")
    decisions = [e for e in events if e["event"] == "planning_decision"]
    if not decisions:
        return
    decision = decisions[0]
    operators = {
        o["operator_id"]: o for o in decision["query_ir"]["semantic_operators"]
    }
    grouped = defaultdict(list)
    for event in events:
        if _confirmed_request(event):
            grouped[event["operator_id"]].append(event)
    actions = {a["operator_id"]: a for a in decision["plan_spec"]["actions"]}
    counters, starts = {}, {}
    # Event order is the real pipeline order, including operators that start
    # between chunks of another operator. Distances start at operator admission.
    for event in events:
        oid = event.get("operator_id")
        if oid not in actions:
            continue
        model = actions[oid]["model_id"]
        current = counters.setdefault(model, list(history.load(model)))
        if event["event"] == "operator_start":
            starts.setdefault(oid, (tuple(current), event.get("observed_unix_seconds")))
        elif _confirmed_request(event):
            starts.setdefault(oid, (tuple(current), event.get("observed_unix_seconds")))
            current[0] += 1
            current[1] += len(event["system_text"]) + len(event["user_text"])
    for model, current in counters.items():
        before = history.load(model)
        history.advance_load(model, current[0] - before[0], current[1] - before[1])
    for oid, rows in grouped.items():
        if rows and oid in actions:
            before, started = starts[oid]
            record_history(
                history,
                operators[oid],
                actions[oid],
                rows,
                capacity,
                max_length,
                started_at=started,
                counters_before=before,
                update_load=False,
                adaptive=adaptive,
            )


def serialize_value(value):
    """Match the prompt JSON-scalar character domain (including escaped tags)."""
    if type(value) is str and len(value) <= 4096:
        return _serialize_text(value)
    # Exact built-in scalars have the same representation as json.dumps. Keep
    # subclasses and compound values on the encoder path, and reject nonfinite
    # floats there so the prompt serializer's error behavior stays unchanged.
    if value is None:
        return "null"
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int or (type(value) is float and math.isfinite(value)):
        return repr(value)
    return _serialize_uncached(value)


@lru_cache(maxsize=1024)
def _serialize_text(value):
    """Bounded immutable text reuse across sampling replay; no compound aliasing."""
    return _serialize_uncached(value)


def _serialize_uncached(value):
    import json

    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        .replace("<", r"\u003c")
        .replace(">", r"\u003e")
    )
