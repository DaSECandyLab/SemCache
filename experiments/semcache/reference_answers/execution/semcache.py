"""Execute Teacher references with the fixed-model request-count Beam policy.

The planner samples independent base inputs for operator selectivity, retains
multiple legal structures, and uses the native GOTDF renderer for every request.
"""

import copy
import csv
import json
import threading
from collections import Counter
from pathlib import Path

from experiments.system_comparison.accuracy.artifacts import result_digest, typed_result
from planning.backends.execution.evaluator import DuckDBEvaluator
from planning.backends.execution.sql import quote
from planning.backends.sampling import NativeSamplingFactory
from planning.search.beam import ParetoBeamSearch, SearchConfig
from planning.search.heuristics import GOTDF_BLOCK_ORDER
from planning.settings.configuration import load_config, response_cache_settings
from planning.settings.modes import profile_for_mode, sampling_for_mode
from planning.state import SearchProblem
from training.artifacts import save

from reference_answers.sql.aggregate import capture_plan


class HeuristicTeacherPolicy:
    """The request-only Beam contract, with the teacher fixed as its model."""

    def __init__(
        self, teacher, output=None, *, build=None, database=None, settings=None
    ):
        self.teacher = teacher
        self.output = Path(output) if output is not None else None
        frozen = settings.get("runtime_configuration") if settings else None
        self.settings = copy.deepcopy(
            frozen or (settings if settings and "search" in settings else load_config())
        )
        self.reference_settings = settings or {}
        if "planning_contract" in self.reference_settings:
            self.settings = {
                **self.settings,
                **self.reference_settings["planning_contract"],
            }
        self.build, self.database = build, database
        self.query = None
        self.plan_spec = None
        self.structural_actions = None

    def optimize(self, query):
        self.query = copy.deepcopy(query)
        if self.output is not None:
            save(self.output / "query_ir.json", self.query)
        for op in query["semantic_operators"]:
            if op["models"] != [self.teacher.model_id]:
                raise ValueError(
                    "Native reference registry must contain only the Teacher"
                )
            if op["kind"] not in self.teacher.supported_kinds:
                raise ValueError(f"Teacher does not support {op['kind']}")
        profile = {
            "models": {
                self.teacher.model_id: {
                    "supported_kinds": list(self.teacher.supported_kinds)
                }
            }
        }
        self.teacher.apply_profile(profile)
        profile = profile_for_mode(
            profile, query, "heuristic_requests", fixed_model_id=self.teacher.model_id
        )
        problem = SearchProblem(query, profile)
        search = ParetoBeamSearch(
            None,
            SearchConfig.configured(
                self.settings,
                "heuristic_requests",
                fixed_model_id=self.teacher.model_id,
            ),
        )
        if self.build is None or self.database is None:
            raise ValueError(
                "Teacher request-count planning requires its native build and source database"
            )
        config = sampling_for_mode(
            self.settings["sampling"],
            "heuristic_requests",
            fixed_model_id=self.teacher.model_id,
        )
        if "cpu_parallelism" in self.reference_settings:
            config["cpu_parallelism"] = self.reference_settings["cpu_parallelism"]
        cache = response_cache_settings(self.settings)
        cache["directory"] = (
            self.reference_settings.get(
                "response_cache_directory",
                str(self.output / ".cache" / "teacher-sampling")
                if self.output
                else None,
            )
            if self.reference_settings.get("cache_enabled", True)
            else None
        )
        setup = "SET semantic_response_cache=" + quote(json.dumps(cache)) + ";"
        if not self.reference_settings.get("cache_enabled", True):
            setup += " SET semantic_request_reuse=false;"
        factory = NativeSamplingFactory(
            self.build,
            self.teacher.transport_models({}),
            database=self.database,
            setup=setup,
            config=config,
            teacher=self.teacher,
            catalog=profile["models"],
        )
        try:
            with factory(problem) as sampler:
                problem.sampler = sampler
                result = search.run(problem)
                if self.output is not None:
                    save(
                        self.output / "search.json",
                        {k: v for k, v in result.items() if k != "states"},
                    )
                    save(
                        self.output / "sampling.json",
                        {
                            "calls": sampler.calls,
                            "plan_generation": "heuristic_requests",
                            "operator_sample_rows": config["operator_sample_rows"],
                            "relational_sample_rows": config["relational_sample_rows"],
                        },
                    )
                self.plan_spec = search.choose(problem, result, "fastest")
        finally:
            factory.close()
        self.structural_actions = copy.deepcopy(self.plan_spec["actions"])
        if self.output is not None:
            save(self.output / "plan_spec.json", self.plan_spec)
            save(
                self.output / "heuristic-plan.json",
                {
                    "protocol": "semcache-calibrated-teacher-gotdf-v3",
                    "policy": "heuristic_requests",
                    "teacher": self.teacher.model_id,
                    "layout": "GOTDF",
                    "block_order": list(GOTDF_BLOCK_ORDER),
                    "field_order": "original_binding_order",
                    "sampling": "independent_base_operator_selectivity",
                    "objective_units": "seconds",
                    "structural_actions": self.structural_actions,
                    "plan_spec": self.plan_spec,
                },
            )
        return self.plan_spec


def _final_result(artifact, query):
    folder = Path(artifact)
    children = {child for node in query["operators"] for child in node["children"]}
    roots = [node for node in query["operators"] if node["node_id"] not in children]
    if len(roots) != 1:
        raise ValueError("Native reference requires one final output schema")
    null_value = json.loads((folder / "result-contract.json").read_text())["null_value"]
    with (folder / "result.csv").open(newline="") as stream:
        reader = csv.reader(stream)
        names = next(reader)
        columns = [
            [name, column["type"]]
            for name, column in zip(names, roots[0]["output_schema"], strict=True)
        ]
        rows = [
            [None if value == null_value else value for value in row] for row in reader
        ]
    return typed_result({"columns": columns, "rows": rows})


def _aggregate_result(artifact, query):
    with (Path(artifact) / "aggregate-input.jsonl").open() as stream:
        columns = json.loads(next(stream))["columns"]
        rows = [row for line in stream if line.strip() for row in json.loads(line)]
    logical = {node["node_id"]: node for node in query["operators"]}
    children = {child for node in logical.values() for child in node["children"]}
    roots = [node for key, node in logical.items() if key not in children]
    if len(roots) != 1:
        raise ValueError("Native aggregate capture requires one output root")
    node = roots[0]
    while node["type"] != "AGGREGATE":
        if len(node["children"]) != 1:
            raise ValueError("Native aggregate capture requires one final aggregate")
        node = logical[node["children"][0]]
    if len(node["children"]) != 1:
        raise ValueError("Native aggregate capture requires one input")
    aggregate_input = logical[node["children"][0]]
    # Native QueryIR exports bound column identities and ordered types, but no
    # output names. Names come from the actual capture header; types
    # and arity must match this exact aggregate's bound input.
    if [column[1] for column in columns] != [
        field["type"] for field in aggregate_input["output_schema"]
    ]:
        raise ValueError("Native aggregate snapshot schema differs from bound input")
    return typed_result({"columns": columns, "rows": rows})


class _TransportCounter:
    def __init__(self):
        self.counts = Counter()
        self.tokens = Counter()
        self.peak = 0

    def observe(self, event):
        kind = event["event"]
        self.counts[kind] += 1
        if kind == "submit":
            self.peak = max(self.peak, event.get("model_concurrency", 0))
        if kind == "semantic_row":
            for key in ("input_tokens", "output_tokens"):
                self.tokens[key] += max(0, event.get(key) or 0)

    def snapshot(self):
        return {
            "http_requests": self.counts["submit"],
            "http_items": self.counts["submit"],
            "completed_requests": self.counts["complete"],
            "semantic_items": sum(
                self.counts[kind]
                for kind in (
                    "semantic_row",
                    "semantic_result_reuse",
                    "semantic_result_cache_hit",
                )
            ),
            "input_tokens": self.tokens["input_tokens"],
            "output_tokens": self.tokens["output_tokens"],
            "peak_concurrency": self.peak,
            "active": max(
                0,
                self.counts["submit"] - self.counts["complete"] - self.counts["error"],
            ),
            "request_granularity": "one_row_or_pair",
            "effective_batch_size": 1,
            "http_requests_are_lower_bound": True,
            "accounting_basis": "native scheduler telemetry; HTTP retries are internal",
        }


def _transport(events):
    counter = _TransportCounter()
    for event in events:
        counter.observe(event)
    return counter.snapshot()


def _monitor_transport(path, stop, progress):
    """Read each complete telemetry line once, retaining an unfinished last line."""
    counter, offset, pending = _TransportCounter(), 0, b""

    def publish():
        nonlocal offset, pending
        if path.exists():
            with path.open("rb") as stream:
                stream.seek(offset)
                appended = stream.read()
                offset = stream.tell()
            lines = (pending + appended).split(b"\n")
            pending = lines.pop()
            for line in lines:
                if line.strip():
                    counter.observe(json.loads(line))
        progress(counter.snapshot())

    while not stop.wait(1):
        publish()
    publish()


def execute(workload, database, build, teacher, output, settings, progress):
    """Run native SQL once and save its result and same-execution aggregate input.

    ``parallelism`` bounds in-flight Teacher requests; ``cpu_parallelism`` sets
    DuckDB threads independently. The production runtime sends one row or pair
    per request; sampling-only batch settings are deliberately unused.
    """
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sql = workload["sql"]
    aggregate = capture_plan(sql)
    policy = HeuristicTeacherPolicy(
        teacher, output, build=build, database=database, settings=settings
    )
    with DuckDBEvaluator(
        build,
        teacher.transport_models({}),
        sql,
        setup=""
        if settings.get("cache_enabled", True)
        else "SET semantic_request_reuse=false;",
        database=database,
        output=output / "native",
        timeout=settings["timeout_seconds"],
        parallelism=settings.get("cpu_parallelism", settings.get("native_threads", 1)),
        max_concurrent_requests_per_model=settings["parallelism"],
        enforce_model_operator_order=settings.get(
            "enforce_model_operator_order", False
        ),
    ) as evaluator:
        # Reference generation owns its cache. Never consume or populate the
        # measured training run's shared persistent answer cache.
        evaluator.response_cache = {
            **evaluator.response_cache,
            "directory": settings.get(
                "response_cache_directory", str(output / ".cache" / "teacher")
            )
            if settings.get("cache_enabled", True)
            else None,
        }
        stop = threading.Event()
        monitor = None
        if progress is not None:
            monitor = threading.Thread(
                target=_monitor_transport,
                args=(output / "native/run-1/telemetry.jsonl", stop, progress),
                daemon=True,
            )
            monitor.start()
        try:
            result = evaluator.run(
                policy_override=policy,
                capture_aggregate_input=aggregate is not None,
                include_rows=False,
            )
        finally:
            stop.set()
            if monitor is not None:
                monitor.join()
    if policy.query is None:
        raise ValueError("Training reference SQL contains no native semantic operators")
    final = _final_result(result["artifact"], policy.query)
    save(output / "result.json", final)
    captured = None
    if aggregate is not None:
        snapshot = _aggregate_result(result["artifact"], policy.query)
        path = output / "pre-aggregate-result.json"
        save(path, snapshot)
        captured = {
            "path": str(path),
            "row_count": len(snapshot["rows"]),
            "columns": snapshot["columns"],
            "result_sha256": result_digest(snapshot),
            "input_sql": aggregate["input_sql"],
            "protocol": "final_aggregate_input_v1",
            "native_capture_path": str(
                Path(result["artifact"]) / "aggregate-input.jsonl"
            ),
        }
    transport = _transport(result["events"])
    diagnostics = {
        "status": result["status"],
        "transport": transport,
        "seconds": result["process_seconds"],
        "native_query_seconds": result["seconds"],
        "native_artifact": result["artifact"],
        "requested_batch_size": settings.get("batch_size", 1),
        "effective_batch_size": 1,
    }
    save(output / "execution.json", diagnostics)
    if progress is not None:
        progress(transport)
    return {
        **final,
        "accuracy_basis": "pre_aggregate_rows"
        if captured is not None
        else "final_output",
        **({"pre_aggregate": captured} if captured is not None else {}),
        "transport": transport,
        "seconds": diagnostics["seconds"],
        "artifact": str(output),
    }
