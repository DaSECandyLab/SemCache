"""Field summary construction from actual execution telemetry."""

import itertools
from collections import defaultdict

from semantic_events import logical_rows

from planning.observations.cache import PrefixSummary
from planning.state import action_key, canonical, digest

from .serialization import (
    FIELD_SERIALIZER,
    _field_row_indices,
    prefix,
    prefix_length,
    request_inputs,
)


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
