"""Update workload time coefficients from completed execution measurements.

Physical operator self timings are used directly, never DuckDB's cumulative
cpu_time or whole-query latency. Semantic timings use the runtime's measured
union of operator-active and HTTP intervals, without adding overlapping CPU time.
"""

import copy
import hashlib
import json
import math
from collections import Counter, defaultdict


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _name(value):
    return str(value or "").strip().upper().replace(" ", "_")


def _identifier(value):
    try:
        return str(int(value)) if value is not None else None
    except (ValueError, TypeError):
        return str(value)


def _walk(profile, path=()):
    if "operator_name" in profile or "operator_type" in profile:
        yield path, profile
    for index, child in enumerate(profile.get("children", [])):
        yield from _walk(child, (*path, index))


def _mapping(node, bank):
    physical = _name(node.get("operator_name") or node.get("operator_type"))
    engine_type = _name(node.get("operator_type"))
    if (
        physical in {"SEQ_SCAN", "TABLE_SCAN", "INDEX_SCAN"}
        or engine_type == "TABLE_SCAN"
    ):
        kind = "SCAN"
    elif physical in {
        "HASH_JOIN",
        "PIECEWISE_MERGE_JOIN",
        "NESTED_LOOP_JOIN",
        "MERGE_JOIN",
        "BLOCKWISE_NL_JOIN",
        "IE_JOIN",
    }:
        kind = "COMPARISON_JOIN"
    elif physical in {"UNGROUPED_AGGREGATE", "HASH_GROUP_BY", "PERFECT_HASH_GROUP_BY"}:
        extra = node.get("extra_info", {})
        kind = (
            "DISTINCT"
            if extra.get("Groups") and not extra.get("Aggregates")
            else "AGGREGATE"
        )
    elif physical == "TOP_N":
        kind = "ORDER_BY" if "ORDER_BY" in bank else "LIMIT"
    elif physical == "ORDER_BY":
        kind = "ORDER_BY"
    elif physical in {"STREAMING_LIMIT", "LIMIT"}:
        kind = "LIMIT"
    else:
        kind = physical
    return physical, kind if kind in bank else None


def _input_units(node, kind):
    children = node.get("children", [])
    if kind in {"SCAN", "CTE_SCAN"}:
        scanned = node.get("operator_rows_scanned") if kind == "SCAN" else None
        return (
            scanned
            if _number(scanned) and scanned > 0
            else node.get("operator_cardinality")
        ), "scan rows" if kind == "SCAN" else "CTE consumer rows"
    if kind == "CTE":
        return (
            children[-1].get("operator_cardinality") if children else None
        ), "CTE consumption branch cardinality"
    inputs = [child.get("operator_cardinality") for child in children]
    if not inputs or not all(_number(value) for value in inputs):
        return None, "missing actual child cardinality"
    if kind == "CROSS_PRODUCT":
        return math.prod(inputs), "product of actual child cardinalities"
    if len(inputs) > 1 and ("JOIN" in kind):
        return sum(inputs), "sum of actual child cardinalities"
    if len(inputs) == 1:
        return inputs[0], "actual child cardinality"
    return None, "unsupported multi-input physical operator"


def _accumulate(record, seconds, units):
    if not _number(seconds) or not _number(units) or units <= 0:
        return "invalid or zero measurement input"
    initial_rate = record.get("seconds_per_input")
    if not _number(initial_rate):
        return "missing initial coefficient"
    initialized = "timed_seconds_total" in record or "timed_input_units_total" in record
    if initialized:
        total_seconds = record.get("timed_seconds_total")
        total_units = record.get("timed_input_units_total")
    else:
        total_seconds = record.get("seconds")
        initial_units, repeats = record.get("input_units"), record.get("repeats")
        total_units = (
            initial_units * repeats
            if _number(initial_units) and _number(repeats)
            else None
        )
    if not _number(total_seconds) or not _number(total_units) or total_units <= 0:
        return "invalid initial/cumulative calibration totals"
    old_seconds = record.get("execution_seconds", 0)
    old_units = record.get("execution_input_units", 0)
    old_count = record.get("execution_measurements", 0)
    if (
        not _number(old_seconds)
        or not _number(old_units)
        or type(old_count) is not int
        or old_count < 0
    ):
        return "invalid previous execution totals"
    new_seconds, new_units = total_seconds + seconds, total_units + units
    if not math.isfinite(new_seconds) or not math.isfinite(new_units):
        return "nonfinite cumulative measurement"
    record.setdefault("initial_seconds_per_input", initial_rate)
    # Native selection updates require these fields whenever cumulative totals
    # are present, even if this execution was the first rolling observation.
    record.setdefault("selection_seconds", 0.0)
    record.setdefault("selection_input_units", 0.0)
    record.setdefault("selection_measurements", 0)
    record.update(
        timed_seconds_total=new_seconds,
        timed_input_units_total=new_units,
        execution_seconds=old_seconds + seconds,
        execution_input_units=old_units + units,
        execution_measurements=old_count + 1,
        seconds_per_input=new_seconds / new_units,
    )
    return None


def update_execution_times(calibration, result, *, query_id=None):
    """Return an updated JSON-compatible copy, including ``execution_updates``.

    Only ``status='completed'`` executions contribute. Each audit records
    physical mappings, per-operator input units/time sources and explicit skips.
    The profile hash plus query ID makes repeated application idempotent through
    ``applied_execution_ids``. Existing selection totals and original calibration
    snapshots remain intact; coefficients are weighted by measured input units.
    """
    updated = copy.deepcopy(calibration)
    profile = result.get("profile") or {}
    events = result.get("events") or []
    runtime_ids = sorted(
        {str(e["query_id"]) for e in events if e.get("query_id") is not None}
    )
    effective_id = (
        query_id
        if query_id is not None
        else (runtime_ids[0] if len(runtime_ids) == 1 else None)
    )
    identity = json.dumps(
        {
            "query_id": effective_id,
            "runtime_query_ids": runtime_ids,
            "profile": profile,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    execution_id = hashlib.sha256(identity.encode()).hexdigest()
    if execution_id in updated.get("applied_execution_ids", []):
        return updated
    audit = {
        "query_id": effective_id,
        "execution_id": execution_id,
        "status": result.get("status"),
        "basis": "physical self operator_timing; semantic measured union of operator-active and HTTP intervals",
        "measurements": [],
        "skips": [],
    }
    if result.get("status") != "completed":
        audit["skips"].append({"reason": "execution did not complete successfully"})
        updated.setdefault("execution_updates", []).append(audit)
        return updated
    if len(runtime_ids) > 1:
        audit["skips"].append(
            {
                "reason": "events contain multiple runtime queries; cannot safely attribute operator IDs"
            }
        )
        updated.setdefault("execution_updates", []).append(audit)
        return updated

    metadata = defaultdict(lambda: {"models": set(), "processing": []})
    query_ir = result.get("query_ir") or {}
    for action in (result.get("plan_spec") or {}).get("actions", []):
        if action.get("model_id"):
            metadata[_identifier(action.get("operator_id"))]["models"].add(
                action["model_id"]
            )
    for event in events:
        event_type = event.get("event")
        if event_type in {"planning_decision", "execution_report"}:
            if event.get("query_ir"):
                query_ir = event["query_ir"]
            for action in event.get("plan_spec", {}).get("actions", []):
                if action.get("model_id"):
                    metadata[_identifier(action.get("operator_id"))]["models"].add(
                        action["model_id"]
                    )
        if event_type == "execution_report":
            for op in event.get("operators", []):
                meta = metadata[_identifier(op.get("operator_id"))]
                meta.setdefault("input_rows", op.get("input_rows_or_pairs"))
                meta.setdefault("input_source", "execution_report.input_rows_or_pairs")
                meta["llm_calls"] = op.get("llm_calls", meta.get("llm_calls"))
                meta["processing"].append(
                    (
                        "execution_report.processing_seconds",
                        op.get("processing_seconds"),
                        op.get("processing_time_basis"),
                    )
                )
        oid = _identifier(event.get("operator_id"))
        if oid is None:
            continue
        meta = metadata[oid]
        if event.get("model_id"):
            meta["models"].add(event["model_id"])
        if event.get("operator_kind"):
            meta["telemetry_kind"] = _name(event["operator_kind"])
        if event_type == "operator_summary":
            meta.update(
                input_rows=event.get("input_rows"),
                input_source="operator_summary.input_rows",
                llm_calls=event.get("llm_calls"),
                query_failed=event.get("query_failed", False),
            )
        if event_type == "operator_summary":
            meta["processing"].insert(
                0,
                (
                    "operator_summary.processing_seconds",
                    event.get("processing_seconds"),
                    event.get("processing_time_basis"),
                ),
            )
    for op in query_ir.get("semantic_operators", []):
        metadata[_identifier(op.get("operator_id"))]["ir_kind"] = _name(op.get("kind"))

    nodes = list(_walk(profile))
    logical_counts = Counter(
        _name(op.get("kind", op.get("type"))) for op in query_ir.get("operators", [])
    )
    physical_counts = Counter(
        _mapping(node, updated.get("relational", {}))[1] for _, node in nodes
    )
    semantic_nodes = defaultdict(list)
    for path, node in nodes:
        extra = node.get("extra_info", {})
        physical = _name(node.get("operator_name") or node.get("operator_type"))
        oid = _identifier(extra.get("operator_id", extra.get("Operator ID")))
        is_semantic = physical.startswith("SEMANTIC_") and physical not in {
            "SEMANTIC_UNARY_INPUT",
            "SEMANTIC_UNARY_COMPLETION",
            "SEMANTIC_JOIN_INPUT",
            "SEMANTIC_JOIN_COMPLETION",
        }
        if is_semantic:
            if oid is None:
                audit["skips"].append(
                    {
                        "path": list(path),
                        "physical": physical,
                        "reason": "semantic physical node has no operator_id",
                    }
                )
            else:
                semantic_nodes[oid].append((path, node))
            continue
        physical, kind = _mapping(node, updated.get("relational", {}))
        if kind is None:
            audit["skips"].append(
                {
                    "path": list(path),
                    "physical": physical,
                    "reason": "engine helper or no calibrated logical kind",
                }
            )
            continue
        units, source = _input_units(node, kind)
        seconds = node.get("operator_timing")
        measurement = {
            "path": list(path),
            "category": "relational",
            "physical": physical,
            "kind": kind,
            "mapping": f"{physical}->{kind}",
            "input_units": units,
            "seconds": seconds,
            "time_source": "profile.operator_timing (self)",
            "input_source": source,
            "fused": physical == "TOP_N"
            or logical_counts[kind] > physical_counts[kind],
            "logical_operator_count": logical_counts[kind],
            "physical_operator_count": physical_counts[kind],
        }
        if physical == "TOP_N":
            measurement.update(
                fused_kinds=["ORDER_BY", "LIMIT"],
                fusion_basis="TOP_N fuses sorting and limit; full self time recorded once in primary kind; no separate LIMIT time fabricated",
            )
        failure = _accumulate(updated["relational"][kind], seconds, units)
        if failure:
            audit["skips"].append({**measurement, "reason": failure})
        else:
            audit["measurements"].append(measurement)
    semantic_ids = set(semantic_nodes) | {
        oid
        for oid, meta in metadata.items()
        if oid is not None and meta.get("input_rows") is not None
    }
    for oid in sorted(semantic_ids):
        physicals = semantic_nodes.get(oid, [])
        meta = metadata[oid]
        physical_kind = next(
            (
                _name(
                    node.get("extra_info", {}).get("Operator")
                    or node.get("operator_name")
                )
                for _, node in physicals
            ),
            None,
        )
        kind = meta.get("ir_kind") or physical_kind or meta.get("telemetry_kind")
        kind_source = (
            "QueryIR"
            if meta.get("ir_kind")
            else "physical operator"
            if physical_kind
            else "telemetry label"
        )
        measurement = {
            "category": "semantic",
            "operator_id": oid,
            "kind": kind,
            "kind_source": kind_source,
            "physical_nodes": [
                {"path": list(path), "physical": node.get("operator_name")}
                for path, node in physicals
            ],
            "input_units": meta.get("input_rows"),
            "input_source": meta.get("input_source"),
            "fused": len(physicals) > 1,
        }
        if meta.get("query_failed"):
            audit["skips"].append({**measurement, "reason": "semantic operator failed"})
            continue
        if len(meta["models"]) != 1:
            audit["skips"].append(
                {
                    **measurement,
                    "reason": "missing or conflicting semantic model identity",
                }
            )
            continue
        model = next(iter(meta["models"]))
        measurement["model_id"] = model
        bank = updated.get("semantic", {}).get(model, {})
        if kind == "SEMANTIC_PROJECT":
            kind = "SEMANTIC_PROJECTION"
            measurement["kind"] = kind
        if kind not in bank:
            audit["skips"].append(
                {**measurement, "reason": "no calibrated semantic model/kind"}
            )
            continue
        candidates = [
            (source, seconds, basis)
            for source, seconds, basis in meta["processing"]
            if _number(seconds)
            and basis == "union_of_operator_active_and_http_intervals"
        ]
        if not candidates:
            audit["skips"].append(
                {
                    **measurement,
                    "reason": "missing valid processing_seconds measured as union_of_operator_active_and_http_intervals; legacy HTTP intervals plus physical timing are not combined",
                }
            )
            continue
        source, seconds, basis = candidates[0]
        measurement.update(
            seconds=seconds,
            time_source=source,
            processing_time_basis=basis,
            llm_calls=meta.get("llm_calls"),
            mapping=f"semantic operator {oid}->{model}/{kind}",
        )
        failure = _accumulate(bank[kind], seconds, meta.get("input_rows"))
        if failure:
            audit["skips"].append({**measurement, "reason": failure})
        else:
            audit["measurements"].append(measurement)
    audit["applied_measurements"] = len(audit["measurements"])
    updated.setdefault("execution_updates", []).append(audit)
    updated.setdefault("applied_execution_ids", []).append(execution_id)
    return updated
