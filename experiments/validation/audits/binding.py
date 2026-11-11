"""Bind the selected SQL files in native binaries, without model inference."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import duckdb

from experiments.core.sql import literal
from experiments.validation.audits.fixtures import ROOT, TABLES, create_fixture


def main(output, manifest_path=None):
    output.mkdir(parents=True, exist_ok=True)
    setups = {}
    for dataset, tables in TABLES.items():
        statements = []
        folder = output / "inputs" / dataset
        folder.mkdir(parents=True, exist_ok=True)
        with duckdb.connect() as con:
            schemas = create_fixture(con, dataset)
            if dataset == "appstore":
                con.execute("ALTER TABLE user_reviews ADD COLUMN __source_rowid BIGINT")
                con.execute("UPDATE user_reviews SET __source_rowid=rowid")
                schemas["user_reviews"].append(("__source_rowid", "BIGINT"))
            for table in tables:
                path = folder / (table + ".csv")
                con.execute(
                    f"COPY \"{table}\" TO ? (HEADER TRUE, NULL '__AUDIT_NULL__')",
                    [str(path)],
                )
                schema = ",".join(
                    literal(n) + ":" + literal(t) for n, t in schemas[table]
                )
                statements.append(
                    f"CREATE TABLE \"{table}\" AS SELECT * FROM read_csv({literal(str(path))},header=true,columns={{{schema}}},nullstr='__AUDIT_NULL__',allow_quoted_nulls=false);"
                )
        setups[dataset] = "\n".join(statements)
    manifest_path = (
        manifest_path or ROOT / "benchmarks/queries/translated/manifest.json"
    ).resolve()
    manifest = json.loads(manifest_path.read_text())
    records = []
    binaries = {
        "semcache": ROOT / "SemCache/build/release/duckdb",
        "sema": ROOT / "systems/sema/vendor/sema-paper/Sema",
        "plop": ROOT / "systems/plop/duckdb-local",
    }
    sys.path.insert(0, str(ROOT / "SemCache/optimizer"))
    from optimizer_server import OptimizerServer

    from experiments.validation.audits.fixed_plan import FixedPlan

    temporary = tempfile.TemporaryDirectory(prefix="query-bind-")
    socket = Path(temporary.name) / "optimizer.sock"
    server = OptimizerServer(socket, FixedPlan(), output / "semcache-plans.jsonl")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for system, exe in binaries.items():
        setup = "SET threads=1;\n"
        args = [str(exe), "-init", "/dev/null", "-batch", "-bail", "-json", ":memory:"]
        if system == "semcache":
            args.insert(1, "-unsigned")
            setup += f"LOAD {literal(str(ROOT / 'SemCache/build/release/extension/semantic/semantic.duckdb_extension'))};\n"
            setup += "SET semantic_optimizer_socket=" + literal(str(socket)) + ";\n"
            setup += (
                "SET semantic_models="
                + literal(
                    json.dumps(
                        {
                            "validation": {
                                "model": "fixture",
                                "base_url": "http://127.0.0.1:1",
                                "structured_output": True,
                            }
                        }
                    )
                )
                + ";\n"
            )
        env = {
            **os.environ,
            "SEMANTIC_OPTIMIZER": "true",
            "DUCKDB_SEMANTIC_MODE": "none",
            "NUM_LLM_WORKERS": "1",
            "API_KEY": "offline-audit",
        }
        if system == "sema":
            setup += "SET enable_semantic_filter_multiplexer=false;\n"
        for query in manifest["records"]:
            if query["system"] != system:
                continue
            dataset, query_name = query["id"].split("/")
            translated = (ROOT / "benchmarks/queries" / query["query_file"]).read_text()
            folder = output / system / dataset / query_name
            folder.mkdir(parents=True, exist_ok=True)
            program = (
                setup + setups[dataset] + "\nEXPLAIN " + translated.rstrip(";") + ";\n"
            )
            (folder / "program.sql").write_text(program)
            try:
                result = subprocess.run(
                    args,
                    check=False,
                    input=program,
                    text=True,
                    capture_output=True,
                    env=env,
                    cwd=folder,
                    timeout=15,
                )
                (folder / "stdout.log").write_text(result.stdout)
                (folder / "stderr.log").write_text(result.stderr)
                records.append(
                    {
                        "id": query["id"],
                        "system": system,
                        "bound": result.returncode == 0,
                        "query_sha256": hashlib.sha256(translated.encode()).hexdigest(),
                        "binary_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
                        "error": result.stderr[-2000:] if result.returncode else None,
                    }
                )
            except subprocess.TimeoutExpired:
                records.append(
                    {
                        "id": query["id"],
                        "system": system,
                        "bound": False,
                        "error": "EXPLAIN timeout",
                    }
                )
        count = sum(r["system"] == system for r in manifest["records"])
        print(
            system,
            "binding",
            sum(r["bound"] for r in records if r["system"] == system),
            "/",
            count,
            flush=True,
        )
    server.shutdown()
    thread.join()
    temporary.cleanup()
    summary = {
        "model_requests": 0,
        "binding_scope": "actual native EXPLAIN; fixed legal SemCache plan, PLOP sampling disabled, model inference not executed",
        "query_manifest": str(manifest_path.relative_to(ROOT)),
        "query_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "bound": sum(r["bound"] for r in records),
        "total": len(records),
        "records": records,
    }
    (output / "audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    expected = sum(r["system"] in binaries for r in manifest["records"])
    if len(records) != expected or not all(r["bound"] for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    a = p.parse_args()
    main(a.output.resolve(), a.manifest)
