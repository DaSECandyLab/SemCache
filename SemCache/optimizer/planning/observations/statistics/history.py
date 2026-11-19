"""Record and ingest confirmed request telemetry into cache history."""

from collections import defaultdict

from planning.observations.cache import PrefixSummary
from planning.state import digest

from .serialization import FIELD_SERIALIZER, prefix, request_inputs


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
