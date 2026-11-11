"""Inspect native SQL optimization on all 19 queries without model inference."""

import argparse
import os
import subprocess
from pathlib import Path

from experiments.core.io import digest, save
from experiments.core.paths import ROOT
from experiments.system_comparison.backends.sema import settings
from experiments.system_comparison.execution.sql import literal


def audit_sql(output):
    import duckdb

    from experiments.validation.audits.fixtures import TABLES, create_fixture

    setups = {}
    for dataset in TABLES:
        folder = output / "inputs" / dataset
        folder.mkdir(parents=True, exist_ok=True)
        statements = []
        with duckdb.connect(config={"threads": 1}) as con:
            schemas = create_fixture(con, dataset)
            for table, columns in schemas.items():
                csv = folder / (table + ".csv")
                con.execute(
                    'COPY "' + table + "\" TO ? (HEADER TRUE, NULL '__AUDIT_NULL__')",
                    [str(csv)],
                )
                schema = ",".join(literal(n) + ":" + literal(t) for n, t in columns)
                statements.append(
                    f"CREATE TABLE \"{table}\" AS SELECT * FROM read_csv({literal(csv)},header=true,columns={{{schema}}},nullstr='__AUDIT_NULL__');"
                )
        setups[dataset] = "\n".join(statements)
    records = []
    for system, binary in {
        "sema": ROOT / "systems/sema/vendor/sema-paper/Sema",
        "plop": ROOT / "systems/plop/duckdb-local",
    }.items():
        binary_hash = digest(binary)
        for path in sorted(
            (ROOT / "benchmarks/queries/translated" / system).glob("*/*.sql")
        ):
            query = path.read_text()
            query_id = path.parent.name + "/" + path.stem
            folder = output / system / query_id
            folder.mkdir(parents=True, exist_ok=True)
            configuration, selected = settings(query) if system == "sema" else ([], {})
            program = (
                "\n".join(configuration)
                + "\nSET threads=16;\n"
                + setups[path.parent.name]
                + "\n.print __MECHANISM_QUERY_BEGIN__\nEXPLAIN "
                + query
            )
            (folder / "program.sql").write_text(program)
            env = {
                **os.environ,
                "SEMANTIC_OPTIMIZER": "true",
                "DUCKDB_SEMANTIC_MODE": "costmodel",
                "DUCKDB_SEMANTIC_DEBUG": "1",
                "NUM_LLM_WORKERS": "16",
                "API_KEY": "offline-audit",
            }
            result = subprocess.run(
                [str(binary), "-init", "/dev/null", "-batch", "-bail", ":memory:"],
                check=False,
                input=program,
                text=True,
                capture_output=True,
                env=env,
                cwd=folder,
                timeout=30,
            )
            (folder / "stdout.log").write_text(result.stdout)
            (folder / "stderr.log").write_text(result.stderr)
            query_stdout = result.stdout.partition("__MECHANISM_QUERY_BEGIN__")[2]
            physical = query_stdout.split("Physical Plan")[-1]
            multiplexer = physical.count("SEM_FILTER_MULTIPLEXER")
            dp = "RunDPPlacement" in query_stdout
            passed = result.returncode == 0 and (
                multiplexer > 0 if selected.get("aqe_eligible") else True
            )
            if system == "plop":
                passed = passed and dp
            record = {
                "system": system,
                "id": query_id,
                "passed": passed,
                "exit_code": result.returncode,
                "query_sha256": digest(path),
                "binary_sha256": binary_hash,
                "multiplexer_nodes": multiplexer,
                "dp_placement_observed": dp,
                **selected,
            }
            records.append(record)
            print(system, query_id, "PASS" if passed else "FAIL", flush=True)
    save(
        output / "sql-plans.json",
        {
            "scope": "native EXPLAIN on controlled fixtures; no model inference",
            "passed": sum(r["passed"] for r in records),
            "total": len(records),
            "records": records,
        },
    )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    records = audit_sql(args.output)
    if not all(r["passed"] for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
