"""One cohort measures each model/kind/layout quality and atomic latency."""

import copy
import math
import statistics
from collections import Counter
from pathlib import Path

from planning.backends.duckdb import DuckDBConnection
from planning.backends.execution import SQLBackend, identifier, quote
from planning.observations.accuracy_protocol import compare_atomic
from planning.prompt_layouts import BLOCK_ORDERS
from planning.sampling import SampleRow
from planning.state import digest, task_key
from semantic_events import logical_rows

from training.calibration.inputs import (
    FILTER_TASK,
    JOIN_TASK,
    PROJECTION_TASK,
    calibration_input_sql,
    calibration_table,
)
from training.calibration.requests import calibration_requests


def calibration_operator(kind, task):
    joined = kind == "SEMANTIC_JOIN"
    fields = [f"arg{i}" for i in range(4 if joined else 2)]
    layout = {"block_order": list(BLOCK_ORDERS[0])}
    if joined:
        layout.update(left_field_order=fields[:2], right_field_order=fields[2:])
    else:
        layout["field_order"] = fields
    return {
        "operator_id": 1,
        "kind": kind,
        "task": task,
        "input_fields": [] if joined else fields,
        "left_input_fields": fields[:2] if joined else [],
        "right_input_fields": fields[2:] if joined else [],
        "input_field_schema": [{"name": field, "type": "VARCHAR"} for field in fields],
        "output_schema": [{"name": "flag", "type": "BOOLEAN"}]
        if kind == "SEMANTIC_PROJECTION"
        else [],
        "default_layout": layout,
    }


def stage_operator(original):
    op = copy.deepcopy(original)
    names = op["input_fields"] or op["left_input_fields"] + op["right_input_fields"]
    if len(names) != (4 if op["kind"] == "SEMANTIC_JOIN" else 2) or any(
        field["type"] != "VARCHAR" for field in op.get("input_field_schema", [])
    ):
        raise ValueError(
            "Generated calibration requires the original text/metadata input contract"
        )
    args = ",".join(identifier(name) for name in names)
    task = quote(op["task"])
    if op["kind"] == "SEMANTIC_PROJECTION":
        if op["output_schema"] != [{"name": "flag", "type": "BOOLEAN"}]:
            raise ValueError(
                "Generated calibration currently requires the flag BOOLEAN contract"
            )
        op["sql"] = (
            f"SELECT __sample_id,semantic_project({task},{args},NULL::STRUCT(flag BOOLEAN)) "
            "AS quality_result FROM $INPUT"
        )
    elif op["kind"] == "SEMANTIC_FILTER":
        op["sql"] = (
            f"SELECT __sample_id FROM $INPUT WHERE semantic_filter({task},{args})"
        )
    elif op["kind"] == "SEMANTIC_JOIN":
        left, right = op["left_input_fields"], op["right_input_fields"]
        args = (
            ",".join("l." + identifier(n) for n in left)
            + ","
            + ",".join("r." + identifier(n) for n in right)
        )
        op["pair_columns"] = [left, right]
        op["sql"] = (
            "SELECT l.__sample_id AS __left_sample_id,r.__sample_id AS __right_sample_id "
            f"FROM $LEFT l JOIN $RIGHT r ON semantic_join({task},{args})"
        )
    else:
        raise ValueError("Unsupported calibration kind")
    return op


def _relational_calibration(backend, raw):
    """Measure generic DuckDB relational work once for estimated objectives.

    DuckDB supplies per-query cardinality/selectivity estimates. This small
    mixed training cohort supplies the missing conversion from one logical
    input row (or candidate pair) to elapsed relational stage seconds. The
    result is keyed by operator kind; query-specific node IDs are deliberately
    not used as global identities.
    """
    rows = [
        SampleRow(
            str(index),
            {
                "text": row["text"],
                "aux": row["aux"],
                "metric": float(index),
                "pair_bucket": index % 8,
            },
            (("calibration", str(index)),),
        )
        for index, row in enumerate(raw[: max(16, len(raw))])
    ]
    if not rows:
        raise ValueError("Relational calibration needs nonempty training rows")
    columns = ["text", "aux", "metric", "pair_bucket"]
    backend.types.update({"metric": "DOUBLE", "pair_bucket": "BIGINT"})
    unary = {
        "FILTER": "SELECT * FROM $INPUT WHERE metric >= 0",
        "PROJECTION": "SELECT __sample_id,text,aux,metric,pair_bucket FROM $INPUT",
        "ORDER_BY": "SELECT * FROM $INPUT ORDER BY metric, __sample_id",
        "LIMIT": "SELECT * FROM $INPUT LIMIT 16",
        "AGGREGATE": "SELECT count(*)::BIGINT AS value FROM $INPUT",
    }
    binary = {
        "CROSS_PRODUCT": (
            "SELECT l.__sample_id AS __left_sample_id, "
            "r.__sample_id AS __right_sample_id "
            "FROM $LEFT l CROSS JOIN $RIGHT r"
        )
    }
    result = {}
    operator_id = -10_000
    for kind, sql in (*unary.items(), *binary.items()):
        op = {
            "operator_id": operator_id,
            "kind": kind,
            "sql": sql,
            "input_columns": [columns] if kind in unary else [columns, columns],
        }
        operator_id -= 1
        samples = []
        for _ in range(3):
            backend.reset_observations()
            inputs = (rows,) if kind in unary else (rows, rows)
            backend(op, inputs, None, None)
            observation = backend.observations[-1]
            incoming = math.prod(observation["inputs"])
            if incoming <= 0 or not math.isfinite(observation["seconds"]):
                raise ValueError("Invalid relational calibration observation")
            samples.append(observation["seconds"] / incoming)
        result[kind] = {
            "seconds_per_input": statistics.mean(samples),
            "samples": len(samples),
            "latency_m2": sum(
                (value - statistics.mean(samples)) ** 2 for value in samples
            ),
            "basis": "one-shot generic DuckDB relational stage calibration",
        }
    result["default"] = {
        "seconds_per_input": statistics.mean(
            value["seconds_per_input"] for key, value in result.items()
        ),
        "samples": sum(value["samples"] for value in result.values()),
        "basis": "mean of generic DuckDB relational stage calibration",
    }
    return result


def calibrate_models(
    workloads,
    models,
    teacher,
    databases,
    build,
    cache_dir,
    *,
    history=None,
    adaptive=None,
    parallelism=1,
    sample_rows=None,
    cohort_seed=None,
    progress=None,
    shared_cache_dir=None,
):
    """Calibrate only generic SF/SP/SJ, once on a mixed original-table cohort.

    Query-specific boundaries are observed by Beam Search, never startup probes.
    """
    from planning.settings.configuration import load_config

    from training.calibration.cache import load_probe, store_probe

    options = load_config()["calibration"]
    sample_rows = options["sample_rows"] if sample_rows is None else sample_rows
    cohort_seed = options["cohort_seed"] if cohort_seed is None else cohort_seed
    if type(sample_rows) is not int or sample_rows < 1:
        raise ValueError("Calibration sample_rows must be a positive integer")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_hits = 0
    datasets = sorted({w["dataset"] for w in workloads if w["split"] == "train"})
    if not datasets or len(datasets) > sample_rows:
        raise ValueError("Calibration requires a bounded nonempty training cohort")
    ops = [
        calibration_operator("SEMANTIC_FILTER", FILTER_TASK),
        calibration_operator("SEMANTIC_PROJECTION", PROJECTION_TASK),
        calibration_operator("SEMANTIC_JOIN", JOIN_TASK),
    ]
    profile = {
        "calibration_protocol": 2,
        "request_timing_protocol": 2,
        "accuracy": {},
        "request_profiles": {},
        "relational_profiles": {},
    }
    transport = teacher.transport_models(models)
    with (
        SQLBackend(
            build,
            transport,
            # Open one read-only source connection; calibration relations are
            # temporary. Source rows below come from each dataset's own file.
            database=databases[datasets[0]],
            collect_telemetry=True,
            history_sink=history,
            adaptive_sketch=adaptive,
            field_types={"quality_result": "VARCHAR"},
            parallelism=parallelism,
        ) as backend,
        calibration_requests(build, transport, parallelism) as measure,
    ):
        assert backend.connection is not None
        raw = []
        for index, dataset in enumerate(datasets):
            quota = sample_rows // len(datasets) + (index < sample_rows % len(datasets))
            source = DuckDBConnection(build, databases[dataset])
            try:
                rows = source.execute(
                    calibration_input_sql(
                        dataset,
                        quota,
                        cohort_seed,
                        source_table=calibration_table(dataset, source),
                    ),
                    named=True,
                )
            finally:
                source.close()
            if not rows:
                raise ValueError(f"Calibration has no training rows for {dataset}")
            for row in rows:
                raw.append(
                    {
                        **row,
                        "doc_id": f"{dataset}:{row['doc_id']}",
                        "pair_bucket": (dataset, row["pair_bucket"]),
                    }
                )
        relational_contract = {
            "datasets": datasets,
            "split": "train",
            "inputs": [(row["doc_id"], row["text"], row["aux"]) for row in raw],
            "version": 1,
        }
        relational_path = cache_dir / (
            "relational-" + digest(relational_contract) + ".json"
        )
        cached = load_probe(relational_path, relational_contract, shared_cache_dir)
        if cached is not None:
            cache_hits += 1
            profile["relational_profiles"] = cached["profile"]
        else:
            if progress:
                progress({"phase": "relational", "done": 0, "total": 6})
            relational_profile = _relational_calibration(backend, raw)
            if progress:
                progress({"phase": "relational", "done": 6, "total": 6})
            store_probe(
                relational_path,
                {"contract": relational_contract, "profile": relational_profile},
                shared_cache_dir,
            )
            profile["relational_profiles"] = relational_profile
        live_measurements = {}
        for original in ops:
            key = task_key(original)
            op = stage_operator(original)
            names = (
                op["input_fields"] or op["left_input_fields"] + op["right_input_fields"]
            )
            if op["kind"] == "SEMANTIC_JOIN":
                records = sorted(
                    [
                        SampleRow(
                            f"{left['doc_id']}:{right['doc_id']}",
                            dict(
                                zip(
                                    names,
                                    [
                                        left["text"],
                                        left["aux"],
                                        right["text"],
                                        right["aux"],
                                    ],
                                )
                            ),
                        )
                        for left in raw
                        for right in raw
                        if left["pair_bucket"] == right["pair_bucket"]
                    ],
                    key=lambda row: digest(row.identity),
                )[:sample_rows]
            else:
                records = [
                    SampleRow(str(r["doc_id"]), dict(zip(names, [r["text"], r["aux"]])))
                    for r in raw
                ]
            if not records:
                raise ValueError("Calibration needs nonempty real input rows")
            contract = {
                "datasets": datasets,
                "split": "train",
                "task": key,
                "operator_kind": op["kind"],
                "models": models,
                "teacher": teacher.configuration(),
                "inputs": [(r.identity, r.values) for r in records],
                "layouts": [":".join(order) for order in BLOCK_ORDERS],
                "version": 11,
            }
            path = cache_dir / (digest(contract) + ".json")
            measured = load_probe(path, contract, shared_cache_dir)
            if measured is not None:
                cache_hits += 1
            else:
                outcomes, requests, squares = {}, {}, {}
                layouts = BLOCK_ORDERS
                measured_units = 0
                total_units = len(layouts) * len(transport)
                for layout_name, model, result, events in measure(op, records):
                    measured_units += 1
                    if history is not None:
                        from planning.observations.statistics import ingest_history

                        ingest_history(events, history, adaptive=adaptive)
                    outcomes.setdefault(layout_name, {})[model] = {
                        r.identity: r.values for r in result
                    }
                    rows = [e for e in events if e["event"] == "semantic_row"]
                    logical = list(logical_rows(events))
                    if (
                        len(logical) != len(records)
                        or not rows
                        or any(e.get("parse_status") != "validated" for e in rows)
                    ):
                        raise ValueError("Incomplete atomic calibration telemetry")
                    requests.setdefault(layout_name, {})[model] = [
                        e["latency_ms"] / 1000 for e in rows
                    ]
                    frequencies = Counter(
                        e.get("source_request_seq", e.get("request_seq"))
                        for e in logical
                    )
                    squares.setdefault(layout_name, {})[model] = sum(
                        count * count for count in frequencies.values()
                    )
                    if progress:
                        reference = outcomes[layout_name].get(teacher.model_id)
                        cells = live_measurements.setdefault(op["kind"], {}).setdefault(
                            layout_name, {"models": {}}
                        )["models"]
                        for observed_model, actual in outcomes[layout_name].items():
                            values = requests[layout_name][observed_model]
                            judged = (
                                reference is not None
                                and observed_model != teacher.model_id
                            )
                            comparison = (
                                compare_atomic(op["kind"], actual, reference or {})
                                if judged
                                else None
                            )
                            cells[observed_model] = {
                                "seconds_per_request": statistics.mean(values),
                                "samples": len(values),
                                "value": comparison["value"] if comparison else None,
                                "total": len(records) if judged else None,
                            }
                        progress(
                            {
                                "phase": "semantic",
                                "kind": op["kind"],
                                "layout": layout_name,
                                "model": model,
                                "teacher_model": teacher.model_id,
                                "done": measured_units,
                                "total": total_units,
                                "kind_index": ops.index(original) + 1,
                                "kinds": len(ops),
                                "rows": len(records),
                                "measurements": live_measurements,
                            }
                        )
                measured = {"contract": contract, "layouts": {}}
                for layout_name in (":".join(order) for order in layouts):
                    measured["layouts"][layout_name] = {"models": {}}
                    for model in models:
                        actual, reference = (
                            outcomes[layout_name][model],
                            outcomes[layout_name][teacher.model_id],
                        )
                        comparison = compare_atomic(op["kind"], actual, reference)
                        values = requests[layout_name][model]
                        measured["layouts"][layout_name]["models"][model] = {
                            "value": comparison["value"],
                            "score_sum": comparison["value"] * len(records),
                            "row_f1": comparison["row_f1"],
                            "attribute_score": comparison["attribute_score"],
                            "exact_correct": comparison["exact"] * len(records),
                            "correct": comparison["exact"] * len(records),
                            "total": len(records),
                            "weighted_correct": comparison["value"] * len(records),
                            "weighted_total": len(records),
                            "weighted_sum_squares": squares[layout_name][model],
                            "effective_samples": len(records) ** 2
                            / squares[layout_name][model],
                            "judge_model": teacher.model_id,
                            "basis": "typed API reference agreement against teacher; not ground truth",
                            "seconds_per_request": statistics.mean(values),
                            "latency_cv": statistics.stdev(values)
                            / statistics.mean(values)
                            if len(values) > 1
                            else 0.0,
                            "samples": len(values),
                            "latency_m2": sum(
                                (value - statistics.mean(values)) ** 2
                                for value in values
                            ),
                            "selectivity": len(actual) / len(records),
                        }
                store_probe(path, measured, shared_cache_dir)
            for model in models:
                layout_evidence = {
                    layout_name: measured["layouts"][layout_name]["models"][model]
                    for layout_name in measured["layouts"]
                }
                aggregate = dict(next(iter(layout_evidence.values())))
                for count in (
                    "correct",
                    "total",
                    "weighted_correct",
                    "weighted_total",
                    "weighted_sum_squares",
                ):
                    aggregate[count] = sum(e[count] for e in layout_evidence.values())
                aggregate["score_sum"] = sum(
                    e["score_sum"] for e in layout_evidence.values()
                )
                aggregate["row_f1"] = sum(
                    e["row_f1"] for e in layout_evidence.values()
                ) / len(layout_evidence)
                attributes = [
                    e["attribute_score"]
                    for e in layout_evidence.values()
                    if e["attribute_score"] is not None
                ]
                aggregate["attribute_score"] = (
                    sum(attributes) / len(attributes) if attributes else None
                )
                aggregate["value"] = (
                    aggregate["score_sum"] / aggregate["weighted_total"]
                )
                aggregate["effective_samples"] = (
                    aggregate["weighted_total"] ** 2 / aggregate["weighted_sum_squares"]
                )
                aggregate["basis"] = (
                    "layout-specific typed API reference agreement; aggregate over allowed layouts"
                )
                aggregate["layouts"] = {
                    name: {
                        k: v
                        for k, v in evidence.items()
                        if k
                        not in (
                            "seconds_per_request",
                            "latency_cv",
                            "latency_m2",
                            "samples",
                            "selectivity",
                        )
                    }
                    for name, evidence in layout_evidence.items()
                }
                profile["accuracy"].setdefault(model, {})[op["kind"]] = aggregate
                latency_samples = sum(e["samples"] for e in layout_evidence.values())
                mean_latency = (
                    sum(
                        e["samples"] * e["seconds_per_request"]
                        for e in layout_evidence.values()
                    )
                    / latency_samples
                )
                latency_m2 = sum(
                    e["latency_m2"]
                    + e["samples"] * (e["seconds_per_request"] - mean_latency) ** 2
                    for e in layout_evidence.values()
                )
                profile["request_profiles"].setdefault(model, {})[op["kind"]] = {
                    "seconds_per_request": mean_latency,
                    "latency_m2": latency_m2,
                    "samples": latency_samples,
                    "latency_cv": math.sqrt(latency_m2 / (latency_samples - 1))
                    / mean_latency
                    if latency_samples > 1 and mean_latency > 0
                    else 0.0,
                    "basis": "counted initial and online atomic request latency",
                }
    if progress:
        progress(
            {
                "phase": "complete",
                "cache_hits": cache_hits,
                "groups": 4,
                "teacher_model": teacher.model_id,
                "measurements": live_measurements,
                "cache_dir": str(shared_cache_dir)
                if shared_cache_dir is not None
                else None,
            }
        )
    return profile
