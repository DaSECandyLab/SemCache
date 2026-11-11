"""Full SQL execution and timeout feedback for candidate plans."""

import copy
import csv
import io
import json
import math
import subprocess
import tempfile
import threading
import time
import urllib.request
import uuid
from pathlib import Path

from optimizer_server import OptimizerServer
from query_runner import quote, verify

from planning.execution.timeouts import execution_progress
from planning.search.protocol import validate_spec
from planning.settings.configuration import response_cache_settings
from planning.settings.parallelism import positive_parallelism


def read_telemetry(path, *, retries=20, delay=0.05):
    """Read append-only telemetry after the native writer's final flush.

    A process can return just before the extension's last JSONL line becomes
    visible to this reader. Retry only a malformed final line; corruption in
    any earlier line remains fatal.
    """
    path = Path(path)
    if not path.exists():
        return []
    for attempt in range(retries + 1):
        lines = path.read_bytes().splitlines()
        events = []
        try:
            for index, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    if index != len(lines) - 1 or attempt == retries:
                        raise
                    time.sleep(delay)
                    break
            else:
                return events
        except json.JSONDecodeError:
            raise
    return events


class ExplicitPolicy:
    def __init__(self, actions=None, source_query=None):
        self.actions = actions
        self.source_query = source_query
        self.query = None

    def optimize(self, query):
        self.query = query
        actions = self.actions
        if isinstance(actions, dict):
            result = copy.deepcopy(actions)
            if (
                "predicate_order" in result or "tree_choices" in result
            ) and self.source_query is not None:
                # EXPLAIN inserts a root and shifts query-local IDs. Replay the
                # complete decision only after checking the same logical tree.
                old = sorted(
                    (
                        n
                        for n in self.source_query["operators"]
                        if n["type"] != "EXPLAIN"
                    ),
                    key=lambda n: n["node_id"],
                )
                new = sorted(
                    (n for n in query["operators"] if n["type"] != "EXPLAIN"),
                    key=lambda n: n["node_id"],
                )
                if len(old) != len(new):
                    raise ValueError(
                        "Cannot replay a predicate plan on a different logical tree"
                    )
                ids = {a["node_id"]: b["node_id"] for a, b in zip(old, new)}
                for a, b in zip(old, new):
                    keys = (
                        "type",
                        "expressions",
                        "required_columns",
                        "output_schema",
                        "semantic_operator_id",
                        "parameters",
                    )
                    if (
                        any(a.get(k) != b.get(k) for k in keys)
                        or [ids[c] for c in a["children"]] != b["children"]
                    ):
                        raise ValueError(
                            "Cannot replay a predicate plan after a logical or binding change"
                        )
                contracts = (
                    "operator_id",
                    "kind",
                    "task",
                    "input_field_schema",
                    "output_schema",
                )
                if [
                    tuple(o.get(k) for k in contracts)
                    for o in self.source_query["semantic_operators"]
                ] != [
                    tuple(o.get(k) for k in contracts)
                    for o in query["semantic_operators"]
                ]:
                    raise ValueError(
                        "Cannot replay a predicate plan after a semantic task change"
                    )
                if "predicate_order" in result:
                    result["predicate_order"] = [
                        {
                            "region_id": ids[r["region_id"]],
                            "nodes": [ids[n] for n in r["nodes"]],
                        }
                        for r in result["predicate_order"]
                    ]
                if "tree_choices" in result:
                    result["tree_choices"] = [
                        {
                            "region_id": ids[r["region_id"]],
                            "root": ids[r["root"]],
                            "actions": [
                                dict(
                                    node_id=ids[a["node_id"]],
                                    inputs=[ids[n] for n in a["inputs"]],
                                    **(
                                        {"pair_for": ids[a["pair_for"]]}
                                        if "pair_for" in a
                                        else {}
                                    ),
                                )
                                for a in r["actions"]
                            ],
                        }
                        for r in result["tree_choices"]
                    ]
            result["query_id"] = query["query_id"]
            validate_spec(query, result)
            return result
        if actions is None:
            actions = [
                {
                    "operator_id": i,
                    "model_id": next(
                        o for o in query["semantic_operators"] if o["operator_id"] == i
                    )["models"][0],
                    "layout": copy.deepcopy(
                        next(
                            o
                            for o in query["semantic_operators"]
                            if o["operator_id"] == i
                        )["default_layout"]
                    ),
                }
                for group in query["order_groups"]
                for i in group
            ]
        result = {
            "version": 1,
            "kind": "PlanSpec",
            "query_id": query["query_id"],
            "actions": copy.deepcopy(actions),
        }
        validate_spec(query, result)
        return result


class DuckDBEvaluator:
    """Own connection per full plan; samples and original inputs are not overwritten.

    Workload SQL/setup are trusted local developer inputs. Complete feedback uses
    native DataChunk execution; SQLBackend runs bounded sampling stages only.
    """

    def __init__(
        self,
        build,
        models,
        query,
        setup="",
        output=None,
        database=None,
        reset_cache=False,
        history_sink=None,
        timeout: float | None = 600.0,
        parallelism=1,
        max_concurrent_requests_per_model=1,
        enforce_model_operator_order=False,
    ):
        self.parallelism = positive_parallelism(parallelism)
        self.response_cache = response_cache_settings()
        self.max_concurrent_requests_per_model = positive_parallelism(
            max_concurrent_requests_per_model
        )
        if type(enforce_model_operator_order) is not bool:
            raise ValueError("enforce_model_operator_order must be a boolean")
        self.enforce_model_operator_order = enforce_model_operator_order
        self.build = Path(build).resolve()
        self.models, self.query, self.setup = models, query, setup
        self.database = database
        self.reset_cache = reset_cache
        self.history_sink = history_sink
        self.timeout = timeout
        self.temporary = tempfile.TemporaryDirectory(prefix="sem-planning-")
        self.output = Path(output) if output else Path(self.temporary.name) / "runs"
        self.output.mkdir(parents=True, exist_ok=False)
        self.counter = 0

    def run(
        self,
        actions=None,
        *,
        explain=False,
        setup=None,
        query=None,
        policy_override=None,
        in_memory=False,
        source_query=None,
        expected_progress=None,
        allow_timeout=False,
        execution_timeout=None,
        capture_aggregate_input=False,
        include_rows=True,
        runtime_options=None,
    ):
        self.counter += 1
        if self.reset_cache and not explain:
            for endpoint in sorted(
                {m["base_url"].rstrip("/") for m in self.models.values()}
            ):
                # Explicit experiment-only SGLang reset, never an implicit retry.
                with urllib.request.urlopen(
                    urllib.request.Request(endpoint + "/flush_cache", method="POST"),
                    timeout=30,
                ) as response:
                    if response.status != 200:
                        raise ValueError("Serving cache reset failed")
        folder = self.output / f"run-{self.counter}"
        folder.mkdir()
        telemetry = folder / "telemetry.jsonl"
        profile = folder / "profile.json"
        policy = policy_override or ExplicitPolicy(actions, source_query)
        socket = Path(self.temporary.name) / "policy.sock"
        if runtime_options is not None:
            from planning.execution.adaptation_service import RuntimeOptimizerServer

            server_class = RuntimeOptimizerServer
        else:
            server_class = OptimizerServer
        with server_class(socket, policy) as server:
            thread = threading.Thread(
                target=lambda: server.serve_forever(poll_interval=0.02), daemon=True
            )
            thread.start()
            clock = []
            server.record = lambda query, response: (
                clock.append(time.perf_counter()) if not clock else None
            )
            null_value = "__SEMCACHE_NULL_" + uuid.uuid4().hex
            folder.joinpath("result-contract.json").write_text(
                json.dumps({"null_value": null_value})
            )
            sql = "\n".join(
                [
                    f"LOAD {quote(self.build / 'extension/semantic/semantic.duckdb_extension')};",
                    f"SET threads={self.parallelism};",
                    f"SET semantic_parallelism={self.parallelism};",
                    "SET semantic_max_concurrent_requests_per_model="
                    f"{self.max_concurrent_requests_per_model};",
                    "SET semantic_enforce_model_operator_order="
                    f"{str(self.enforce_model_operator_order).lower()};",
                    f"SET semantic_models={quote(json.dumps(self.models))};",
                    f"SET semantic_response_cache={quote(json.dumps(self.response_cache))};",
                    f"SET semantic_optimizer_socket={quote(socket)};",
                    "SET semantic_optimizer_timeout_ms=120000;",
                    f"SET semantic_runtime={quote(json.dumps(runtime_options) if runtime_options and not explain else '')};",
                    "SET semantic_capture_aggregate_input="
                    + quote(
                        str((folder / "aggregate-input.jsonl").resolve())
                        if capture_aggregate_input and not explain
                        else ""
                    )
                    + ";",
                    f"SET semantic_telemetry={quote(telemetry.resolve())};",
                    self.setup if setup is None else setup,
                    "PRAGMA enable_profiling='json';",
                    f"SET profiling_output={quote(profile.resolve())};",
                    ("EXPLAIN " if explain else "") + (query or self.query),
                ]
            )
            folder.joinpath("query.sql").write_text(sql)
            cli = [
                str(self.build / "duckdb"),
                "-no-init",
                "-unsigned",
                "-batch",
                "-bail",
                "-csv",
                "-header",
                "-nullvalue",
                null_value,
            ]
            if self.database and not in_memory:
                cli.extend(["-readonly", str(self.database)])
            start = time.perf_counter()
            limit = self.timeout if execution_timeout is None else execution_timeout
            if limit is not None and (not math.isfinite(limit) or limit <= 0):
                raise ValueError("Execution timeout must be positive")
            timed_out = False
            try:
                with subprocess.Popen(
                    cli,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                ) as process:
                    first = True
                    while True:
                        try:
                            stdout, stderr = process.communicate(
                                input=sql if first else None, timeout=0.1
                            )
                            result = subprocess.CompletedProcess(
                                cli, process.returncode, stdout, stderr
                            )
                            break
                        except subprocess.TimeoutExpired:
                            first = False
                            now = time.perf_counter()
                            if (
                                clock and limit is not None and now - clock[0] >= limit
                            ) or (
                                not clock
                                and self.timeout is not None
                                and now - start >= self.timeout
                            ):
                                process.kill()
                                stdout, stderr = process.communicate()
                                result = subprocess.CompletedProcess(
                                    cli, process.returncode, stdout, stderr
                                )
                                timed_out = True
                                break
            finally:
                server.shutdown()
                thread.join()
        socket.unlink()
        elapsed = time.perf_counter() - start
        folder.joinpath("result.csv").write_text(result.stdout)
        folder.joinpath("stderr.log").write_text(result.stderr)
        if timed_out:
            events = read_telemetry(telemetry, retries=0)
            if not clock or explain:
                raise TimeoutError(
                    "Setup/planning exceeded its budget; no execution censor label is valid"
                )
            bound = limit
            progress = execution_progress(
                events, expected_progress, time.perf_counter() - clock[0]
            )
            partial = {
                "status": "timeout",
                "seconds": None,
                "time_lower_bound": bound,
                "process_seconds": elapsed,
                "events": events,
                "rows": None,
                "artifact": str(folder),
                "progress": progress,
                "clock_basis": "validated PlanSpec delivery; setup excluded",
            }
            folder.joinpath("timeout.json").write_text(
                json.dumps({k: v for k, v in partial.items() if k != "events"})
            )
            if self.history_sink is not None:
                from planning.observations.statistics import ingest_history

                if runtime_options:
                    from planning.execution.runtime import update_history

                    update_history(events, policy.query, self.history_sink)
                else:
                    ingest_history(events, self.history_sink)
            if not allow_timeout:
                raise TimeoutError(
                    f"Plan execution exceeded {limit} seconds; partial telemetry at {folder}"
                )
            return partial
        if result.returncode:
            raise ValueError(result.stderr)
        if explain:
            if policy.query is None:
                raise ValueError("No semantic QueryIR discovered by EXPLAIN")
            return policy.query
        events = read_telemetry(telemetry)
        if policy.query is not None:
            verify(events)
            if self.history_sink is not None:
                from planning.observations.statistics import ingest_history

                if runtime_options:
                    from planning.execution.runtime import update_history

                    update_history(events, policy.query, self.history_sink)
                else:
                    ingest_history(events, self.history_sink)
        rows = None
        if include_rows:
            reader = csv.reader(io.StringIO(result.stdout))
            columns = next(reader, [])
            rows = []
            for row in reader:
                # DuckDB CSV renders one empty string as a blank record.
                # DictReader skips it, losing a valid projection result row.
                if not row and len(columns) == 1:
                    row = [""]
                rows.append(
                    {
                        key: None if value == null_value else value
                        for key, value in zip(columns, row, strict=True)
                    }
                )
        profiling = json.loads(profile.read_text())
        return {
            "status": "completed",
            "query_ir": policy.query,
            "rows": rows,
            "csv": result.stdout,
            "seconds": profiling["latency"],
            "process_seconds": elapsed,
            "events": events,
            "profile": profiling,
            "artifact": str(folder),
        }

    def close(self):
        self.temporary.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
