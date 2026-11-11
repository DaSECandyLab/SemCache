"""Replay runtime state transitions to audit batch snapshots against actual requests."""

import copy
from typing import Any

from optimizer_server import validate_spec


def effective_actions(events, decision, report):
    """Return request -> admitted action; reject inconsistent runtime histories."""
    spec = decision["plan_spec"]
    initialized = [event for event in events if event["event"] == "runtime_initialized"]
    if report.get("runtime_state") is None:
        if initialized or any(
            event["event"].startswith(("batch_", "runtime_update")) for event in events
        ):
            raise ValueError("Runtime events without a terminal runtime state")
        return {}
    if len(initialized) != 1:
        raise ValueError("Expected one runtime initialization")
    state: dict[str, Any] = {
        action["operator_id"]: {
            "operator_id": action["operator_id"],
            "status": "PENDING",
            "current": dict(copy.deepcopy(action), config_generation=0),
            "next_batch_seq": 0,
            "completed": 0,
            "emitted": 0,
        }
        for action in spec["actions"]
    }
    if initialized[0]["state"] != sorted(
        state.values(), key=lambda s: s["operator_id"]
    ):
        raise ValueError("Runtime initial state differs from PlanSpec")
    chunk_mode = initialized[0]["options"]["version"] == 2
    capacity = initialized[0]["options"].get("batch_size")
    active, requests, partial = {}, {}, set()
    last_request = 0
    for event in events:
        kind = event["event"]
        if (
            kind.startswith(("runtime_", "batch_")) or kind == "reoptimization_trigger"
        ) and (
            event["query_id"] != spec["query_id"]
            or (
                event["after_request_seq"] < last_request
                if chunk_mode
                else event["after_request_seq"] != last_request
            )
        ):
            raise ValueError("Runtime event query identity/request boundary mismatch")
        if kind == "runtime_update":
            changed = copy.deepcopy(state)
            targets = set()
            for change in event["changes"]:
                before, after = change["before"], change["after"]
                operator = before["operator_id"]
                if (
                    operator in targets
                    or before != state[operator]
                    or before["status"] == "DONE"
                    or chunk_mode
                    and operator in active
                ):
                    raise ValueError(
                        "Runtime update targeted stale/duplicate/DONE state"
                    )
                targets.add(operator)
                expected = copy.deepcopy(before)
                expected["current"] = after["current"]
                if (
                    after != expected
                    or after["current"]["config_generation"]
                    != before["current"]["config_generation"] + 1
                    or change["effective_from_batch_seq"] != before["next_batch_seq"]
                ):
                    raise ValueError(
                        "Runtime update changed immutable state or generation"
                    )
                changed[operator] = after
            candidate = copy.deepcopy(spec)
            candidate["actions"] = [
                {
                    k: v
                    for k, v in changed[a["operator_id"]]["current"].items()
                    if k != "config_generation"
                }
                for a in spec["actions"]
            ]
            validate_spec(decision["query_ir"], candidate)
            state = changed
        elif kind == "batch_admitted":
            operator = event["operator_id"]
            entry = state[operator]
            action = {
                k: v for k, v in entry["current"].items() if k != "config_generation"
            }
            if (
                operator in active
                or operator in partial
                or entry["status"] == "DONE"
                or event["batch_seq"] != entry["next_batch_seq"]
                or event["first_input"] != entry["completed"]
                or event["config_generation"] != entry["current"]["config_generation"]
                or event["action"] != action
                or (
                    type(event["capacity"]) is not int or event["capacity"] < 1
                    if chunk_mode
                    else event["capacity"] != capacity
                )
            ):
                raise ValueError("Invalid runtime batch admission/snapshot")
            active[operator] = dict(event, completed=0, emitted=0)
            if chunk_mode:
                active[operator]["seen"] = set()
            entry["status"] = "RUNNING"
            entry["next_batch_seq"] += 1
        elif kind in (
            "semantic_row",
            "semantic_result_reuse",
            "semantic_result_cache_hit",
        ):
            operator = event["operator_id"]
            batch = active[operator]
            entry = state[operator]
            if (
                event["batch_seq"] != batch["batch_seq"]
                or event["config_generation"] != batch["config_generation"]
                or (
                    event["batch_index"] != event["input_index"] - batch["first_input"]
                    or event["batch_index"] in batch["seen"]
                    or not 0 <= event["batch_index"] < batch["capacity"]
                    if chunk_mode
                    else event["batch_index"] != batch["completed"]
                    or event["input_index"] != entry["completed"]
                    or batch["completed"] >= capacity
                )
            ):
                raise ValueError("Request escaped its immutable runtime batch")
            requests[(operator, event["input_index"])] = batch["action"]
            if chunk_mode:
                batch["seen"].add(event["batch_index"])
            batch["completed"] += 1
            batch["emitted"] += event["emitted"]
            entry["completed"] += 1
            entry["emitted"] += event["emitted"]
            if kind == "semantic_row":
                last_request = (
                    max(last_request, event["request_seq"])
                    if chunk_mode
                    else event["request_seq"]
                )
            elif kind == "semantic_result_cache_hit":
                if event["after_request_seq"] < last_request:
                    raise ValueError(
                        "Result cache hit moved the request boundary backwards"
                    )
                last_request = event["after_request_seq"]
            elif not chunk_mode and event["after_request_seq"] != last_request:
                raise ValueError("Result reuse changed the physical request boundary")
        elif kind in ("batch_completed", "runtime_chunk_completed"):
            operator = event["operator_id"]
            batch = active.pop(operator)
            if any(
                event[k] != batch[k]
                for k in ("batch_seq", "config_generation", "completed", "emitted")
            ):
                raise ValueError("Runtime batch completion accounting mismatch")
            if not chunk_mode and batch["completed"] < capacity:
                partial.add(operator)
        elif kind == "runtime_operator_done":
            operator = event["operator_id"]
            if operator in active or state[operator]["status"] == "DONE":
                raise ValueError(
                    "Runtime operator finalized with active batch or twice"
                )
            state[operator]["status"] = "DONE"
            if event["state"] != state[operator]:
                raise ValueError("Runtime DONE state mismatch")
    if active or any(entry["status"] != "DONE" for entry in state.values()):
        raise ValueError("Successful execution left unfinished runtime operators")
    if report["runtime_state"] != sorted(
        state.values(), key=lambda s: s["operator_id"]
    ):
        raise ValueError("Terminal runtime state differs from replayed history")
    groups = {}
    for event in events:
        if event["event"] in (
            "semantic_row",
            "semantic_result_reuse",
            "semantic_result_cache_hit",
        ):
            key = (event["operator_id"], event["config_generation"])
            groups.setdefault(key, []).append(event)
    observations = report["runtime_observations"]
    if [(s["operator_id"], s["config_generation"]) for s in observations] != sorted(
        groups
    ):
        raise ValueError("Runtime observations missing/duplicating generations")
    for summary in observations:
        selected = groups[(summary["operator_id"], summary["config_generation"])]
        emitted = sum(row["emitted"] for row in selected)
        # Generation observations are updated only when a result is consumed.
        # Unconsumed paid requests are audited as semantic_request_discarded
        # in execution_report totals and never become generation observations.
        physical = [row for row in selected if row["event"] == "semantic_row"]
        latency = (
            sum(row["latency_ms"] for row in physical) / len(physical)
            if physical
            else None
        )
        if (
            summary["completed"] != len(selected)
            or summary["emitted"] != emitted
            or summary["selectivity"] != emitted / len(selected)
            or summary["action"]
            != requests[(selected[0]["operator_id"], selected[0]["input_index"])]
            or summary.get("llm_calls", len(physical)) != len(physical)
            or (summary["latency_avg_ms"] is None) != (latency is None)
            or latency is not None
            and abs(summary["latency_avg_ms"] - latency) > 1e-8
        ):
            raise ValueError("Runtime generation observations differ from actual calls")
    return requests
