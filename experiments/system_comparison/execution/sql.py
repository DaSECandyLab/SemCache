"""Native SQL experiments with a retained prepared Sema plan when possible."""

import csv
import decimal
import io
import json
import os
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path

from experiments.core.io import digest, save
from experiments.core.paths import ROOT
from experiments.core.resources import BATCH_SIZE, CONCURRENCY
from experiments.system_comparison.reporting.results import normalize
from experiments.validation.runtime.sql_ports import native_sql

SESSIONS = {}


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def sql_setup(job):
    if job.get("input_loading") == "database_memory":
        from experiments.system_comparison.data.memory import preload_sql

        return preload_sql(job["input_database"], job["inputs"])
    statements = []
    for table, item in job["inputs"].items():
        columns = ",".join(
            literal(name) + ":" + literal(kind)
            for name, kind in [
                *item["columns"],
                ["__source_rowid", "BIGINT"],
                ["__validation_rowid", "BIGINT"],
            ]
        )
        statements.append(
            f'CREATE TEMP TABLE "{table}" AS SELECT * EXCLUDE({"__validation_rowid" if table == "user_reviews" else "__source_rowid,__validation_rowid"}) FROM read_csv({literal(item["csv"])},header=true,columns={{{columns}}},nullstr={literal(item["csv_null"])},allow_quoted_nulls=false);'
        )
    return "\n".join(statements)


def parse_result(path, schema):
    class Pairs(list):
        pass

    text = Path(path).read_text()
    parsed = (
        json.loads(
            text,
            object_pairs_hook=Pairs,
            parse_float=decimal.Decimal,
            parse_int=int,
        )
        if text.strip()
        else []
    )

    def nested(value):
        if isinstance(value, Pairs):
            return {key: nested(item) for key, item in value}
        if isinstance(value, list):
            return [nested(item) for item in value]
        return value

    rows = []
    column_mapping = None
    for row in parsed:
        native_names = [name for name, _ in row]
        expected_names = [name for name, _ in schema]
        if native_names != expected_names:
            # The common binder uses a subquery, whose DuckDB contract makes
            # duplicate names unique. Preserve the native duplicate positions,
            # and permit only that exact, explicit identifier normalization.
            used = set()
            unique_names = []
            for name in native_names:
                candidate, suffix = name, 1
                while candidate.casefold() in used:
                    candidate = f"{name}_{suffix}"
                    suffix += 1
                unique_names.append(candidate)
                used.add(candidate.casefold())
            if unique_names != expected_names:
                raise ValueError("Native JSON output columns differ from canonical SQL")
            column_mapping = [
                {"position": index, "native_name": name, "canonical_name": expected}
                for index, (name, expected) in enumerate(
                    zip(native_names, expected_names, strict=True)
                )
            ]
        rows.append([nested(value) for _, value in row])
    if column_mapping is not None:
        save(
            Path(path).with_suffix(".schema-mapping.json"),
            {
                "mapping": column_mapping,
                "rule": "DuckDB duplicate-column names normalized by position; no column/value dropped",
            },
        )
    return normalize({"columns": schema, "rows": rows})


class Session:
    """Keep the native compiled statement in its original process; no SQL reload."""

    def __init__(self, executable, folder, env, cpu_set=None, *, unsigned=False):
        self.folder = folder
        self.stdout = (folder / "stdout.log").open("w")
        self.stderr = (folder / "stderr.log").open("w")
        self.lines = queue.Queue()
        command = [str(executable), "-init", "/dev/null", "-batch", "-bail", ":memory:"]
        if unsigned:
            command.insert(1, "-unsigned")
        if cpu_set:
            # Pin the whole native process (and the threads it spawns) to its own
            # core range so parallel variants do not share CPU time.
            command = ["taskset", "-c", str(cpu_set)] + command
        self.child = subprocess.Popen(
            command,
            cwd=folder,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        def read():
            for line in self.child.stdout:
                self.stdout.write(line)
                self.stdout.flush()
                self.lines.put(line.rstrip("\r\n"))
            self.lines.put(None)

        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()
        self.serial = 0

    def execute(self, text, timeout, *, heartbeat=None, start=None):
        start = time.monotonic() if start is None else start
        self.serial += 1
        marker = f"__PAPER_DONE_{self.child.pid}_{self.serial}__"
        self.child.stdin.write(text + "\n.print " + marker + "\n")
        self.child.stdin.flush()
        timed_out = False
        while True:
            elapsed = time.monotonic() - start
            if heartbeat:
                heartbeat(elapsed, self.child.pid)
            if timeout is not None and elapsed >= timeout:
                timed_out = True
                self.close(kill=True)
                break
            try:
                line = self.lines.get(
                    timeout=2
                    if timeout is None
                    else max(0.001, min(2, timeout - elapsed))
                )
            except queue.Empty:
                continue
            if line == marker:
                return {
                    "status": "completed",
                    "seconds": time.monotonic() - start,
                    "returncode": 0,
                    "timed_out": False,
                }
            if line is None:
                self.child.wait()
                break
        return {
            "status": "timeout" if timed_out else "failed",
            "seconds": time.monotonic() - start,
            "returncode": self.child.returncode,
            "timed_out": timed_out,
        }

    def close(self, *, kill=False):
        if self.child.poll() is None:
            if kill:
                os.killpg(self.child.pid, signal.SIGKILL)
            else:
                try:
                    self.child.stdin.write(".quit\n")
                    self.child.stdin.flush()
                except BrokenPipeError:
                    pass
            try:
                self.child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.child.pid, signal.SIGKILL)
                self.child.wait()
        self.reader.join(timeout=5)
        self.stdout.close()
        self.stderr.close()


def explain_fingerprint(path):
    from hashlib import sha256

    # EXPLAIN EXECUTE wraps the same prepared tree and contains no actual timing.
    return sha256(Path(path).read_bytes()).hexdigest()


def run_sql(system, job, folder, *, heartbeat=None, cpu_set=None, port=None):
    if system == "sema":
        executable = ROOT / "systems/sema/vendor/sema-paper/Sema"
    else:
        # Parallel PLOP variants need their own gateway port; each port has a
        # separately patched binary because the native URL is compiled in.
        name = f"duckdb-local-{port}" if port else "duckdb-local"
        executable = ROOT / f"systems/plop/{name}"
        if not executable.exists():
            executable = ROOT / "systems/plop/duckdb-local"
    setup = sql_setup(job)
    env = os.environ.copy()
    if system == "sema":
        setup = "\n".join(
            [
                f"SET llm_url={literal(job['endpoint'] + '/chat/completions')};",
                f"SET llm_model={literal(job['model'])};",
                "SET llm_api_key='EMPTY';",
                f"SET semantic_batch_size={job.get('batch_size', BATCH_SIZE)};",
                "SET enable_semantic_filter_multiplexer="
                + ("true;" if job["dataset"] == "appstore" else "false;"),
                "SET semantic_filter_accuracy_threshold=0.9;",
                "SET semantic_filter_latency_first=true;",
                f"SET semantic_filter_batch_size={job.get('batch_size', BATCH_SIZE)};",
                f"SET threads={CONCURRENCY};",
                setup,
            ]
        )
    else:
        env.update(
            SEMANTIC_OPTIMIZER="true",
            DUCKDB_SEMANTIC_MODE="costmodel",
            API_KEY="EMPTY",
            NUM_LLM_WORKERS="4",
        )
        setup = f"SET threads={CONCURRENCY};\n" + setup
    translated = native_sql(
        job["sql"],
        system,
        rowid_column="__source_rowid" if job["dataset"] == "appstore" else None,
    )
    (folder / "query.sql").write_text(translated + ";\n")
    explain, raw = folder / "prepared-explain.csv", folder / "native-result.json"
    query_program = f"\nPREPARE __paper_plan AS {translated.rstrip(';')};\n.mode csv\n.once {explain}\nEXPLAIN EXECUTE __paper_plan;\nPRAGMA enable_profiling='json';\nPRAGMA profiling_output={literal(folder / 'profile.json')};\n.mode json\n.once {raw}\nEXECUTE __paper_plan;\n"
    program = setup + query_program
    (folder / "program.sql").write_text(program)
    session = Session(executable, folder, env, cpu_set=cpu_set)
    try:
        timeout = job.get("timeout", 1800)
        # Table creation / input loading runs untimed: the reported query time
        # starts only after the tables exist, so it can be compared with the
        # native systems that keep their inputs preloaded.
        load_started = time.monotonic()
        load = session.execute(setup, timeout, heartbeat=heartbeat)
        load_seconds = time.monotonic() - load_started
        if load["status"] != "completed":
            stats = dict(load)
        else:
            remaining = (
                None if timeout is None else max(1.0, float(timeout) - load_seconds)
            )
            stats = session.execute(
                query_program, remaining, heartbeat=heartbeat, start=time.monotonic()
            )
        stats["load_seconds"] = load_seconds
        if stats["status"] == "completed":
            result = parse_result(raw, job["output_schema"])
            save(folder / "result.json", result)
            plan = {
                "system": system,
                "capture_point": "retained_native_prepared_statement",
                "query_sha256": job["sql_sha256"],
                "binary_sha256": digest(executable),
                "executable": str(executable),
                "cpu_set": cpu_set,
                "load_seconds": round(load_seconds, 6),
                "explain_sha256": explain_fingerprint(explain),
                "explain": list(csv.reader(io.StringIO(explain.read_text()))),
                "prepared_name": "__paper_plan",
                "native_pid": session.child.pid,
                "model": job["model"],
                "input_bindings": {
                    name: {"rows": entry["rows"], "sha256": entry["sha256"]}
                    for name, entry in job["inputs"].items()
                },
                "output_schema": job["output_schema"],
                "adaptive_native_strategy": system == "sema"
                and job["dataset"] == "appstore",
                "replay_supported": True,
                "replay_limitations": "Same-process prepared plan only; reference verifies unchanged EXPLAIN and fresh teacher requests. Native AQE strategy is retained; internal decisions may depend on new teacher responses."
                if system == "sema"
                else "Same-process prepared plan only; an audited cache isolation bugfix supplies explicit clear between completed runs. Native DP and within-query response reuse are retained.",
            }
            save(folder / "plan.json", plan)
            SESSIONS[str(folder)] = session
            return {
                **stats,
                "result_rows": len(result["rows"]),
                "execution_mode": "native_sql",
                "optimizer_enabled": True,
                "optimizer_settings": "paper AQE recommendation on Appstore, native scalar pipeline elsewhere"
                if system == "sema"
                else "native DP costmodel",
                "plan_replay_supported": plan["replay_supported"],
                "plan": str(folder / "plan.json"),
            }
        session.close(kill=True)
        return {
            **stats,
            "reason": (folder / "stderr.log").read_text()[-6000:],
            "execution_mode": "native_sql",
        }
    except BaseException:
        session.close(kill=True)
        raise


def replay_sema(source, job, folder, *, heartbeat=None):
    """Execute the same retained compiled statement, with a teacher-only route."""
    session = SESSIONS.get(str(source))
    if session is None or session.child.poll() is not None:
        return {
            "status": "reference_unavailable",
            "reason": "The original prepared-plan process is not alive; re-running SQL would reoptimize",
        }
    plan = json.loads((source / "plan.json").read_text())
    executable = (
        Path(plan["executable"])
        if plan.get("executable")
        else ROOT
        / (
            "systems/sema/vendor/sema-paper/Sema"
            if plan["system"] == "sema"
            else "systems/plop/duckdb-local"
        )
    )
    if digest(executable) != plan["binary_sha256"]:
        raise ValueError("Native binary changed after plan capture")
    if plan["system"] == "plop":
        cleared = folder / "cleared-response-cache.json"
        stats = session.execute(
            f".mode json\n.once {cleared}\nSELECT semantic_clear_response_cache() AS cleared;",
            None,
            heartbeat=heartbeat,
        )
        if stats["status"] != "completed" or parse_result(
            cleared, [["cleared", "BOOLEAN"]]
        )["rows"] != [[True]]:
            return {
                "status": "reference_unavailable",
                "reason": "Native response cache could not be cleared before teacher replay",
            }
    explain, raw = folder / "prepared-explain.csv", folder / "native-result.json"
    text = f".mode csv\n.once {explain}\nEXPLAIN EXECUTE __paper_plan;\nPRAGMA profiling_output={literal(folder / 'profile.json')};\n.mode json\n.once {raw}\nEXECUTE __paper_plan;\n"
    (folder / "replay.sql").write_text(text)
    stats = session.execute(text, None, heartbeat=heartbeat)
    if stats["status"] == "completed":
        if explain_fingerprint(explain) != plan["explain_sha256"]:
            return {
                **stats,
                "status": "failed",
                "reason": "Prepared physical EXPLAIN changed during teacher replay",
            }
        result = parse_result(raw, job["output_schema"])
        save(folder / "result.json", result)
        save(
            folder / "plan.json",
            {
                **plan,
                "model": job["teacher_model"],
                "replay": "same_process_prepared",
                "structure_verified": True,
            },
        )
    return {
        **stats,
        "execution_mode": "same_process_prepared",
        "plan_structure_verified": stats["status"] == "completed",
    }


def close_sessions():
    for session in SESSIONS.values():
        session.close()
    SESSIONS.clear()
