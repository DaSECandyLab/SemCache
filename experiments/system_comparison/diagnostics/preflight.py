"""Real-API integration checks on two rows, excluded from the full benchmark."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

from experiments.core.io import save
from experiments.core.models import model_snapshot
from experiments.core.resources import job_resources
from experiments.system_comparison.scheduling.campaign import (
    native_gateway_port,
    worker_command,
)
from experiments.system_comparison.transport.gateway import (
    PoolGateway,
    flush_model_caches,
)


def main():
    import duckdb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--systems", nargs="+", default=["plop", "sema", "abacus", "nirvana", "lotus"]
    )
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--query-id", default="estate/Q9")
    parser.add_argument("--rows", type=int, default=2)
    args = parser.parse_args()
    run = args.run.resolve()
    pre = run / "preflight"
    pre.mkdir(exist_ok=True)
    if args.query_id != "estate/Q9" or args.rows != 2:
        pre = pre / (args.query_id.replace("/", "-") + f"-rows{args.rows}")
        pre.mkdir(exist_ok=True)
    frozen = json.loads((run / "inputs/manifest.json").read_text())
    if args.query_id not in frozen["jobs"] or args.rows < 1:
        raise ValueError("Invalid preflight query or row count")
    base = frozen["jobs"][args.query_id]
    dataset = base["dataset"]
    db = pre / "inputs.duckdb"
    with duckdb.connect(str(db)) as con:
        path = str(run / "inputs/inputs.duckdb").replace("'", "''")
        con.execute("ATTACH '" + path + "' AS frozen (READ_ONLY)")
        if dataset == "estate":
            con.execute(
                f"CREATE TABLE IF NOT EXISTS nirvana_estate AS SELECT * FROM frozen.nirvana_estate WHERE Title ILIKE '%4 bedroom%' LIMIT {args.rows}"
            )
        elif dataset == "movies":
            con.execute(
                f"CREATE TABLE IF NOT EXISTS nirvana_movies AS SELECT * FROM frozen.nirvana_movies LIMIT {args.rows}"
            )
        elif dataset == "appstore":
            source = (
                Path("/home/lyp/project/benchmarks/queries/translated/semcache")
                / (args.query_id + ".sql")
            ).read_text()
            category = re.search(r"Category\s*=\s*'([^']+)'", source, re.IGNORECASE)[1]
            con.execute(
                f"CREATE TABLE IF NOT EXISTS user_reviews AS SELECT * FROM frozen.user_reviews WHERE App IN (SELECT App FROM frozen.playstore WHERE Category=?) LIMIT {args.rows}",
                [category],
            )
            con.execute(
                "CREATE TABLE IF NOT EXISTS playstore AS SELECT * FROM frozen.playstore WHERE App IN (SELECT App FROM user_reviews)"
            )
        else:
            raise ValueError(
                "TPC-H graph shapes are checked offline; choose a bounded Estate/Movies/Appstore API probe"
            )
    inputs = {}
    with duckdb.connect(str(db), read_only=True) as con:
        for name, entry in base["inputs"].items():
            table = entry.copy()
            csv = pre / (name + ".csv")
            con.execute(
                'COPY (SELECT *,rowid AS __source_rowid,rowid AS __validation_rowid FROM "'
                + name
                + '") TO ? (HEADER TRUE,NULL ?)',
                [str(csv), table["csv_null"]],
            )
            table.update(
                csv=str(csv),
                rows=con.execute('SELECT COUNT(*) FROM "' + name + '"').fetchone()[0],
            )
            inputs[name] = table
    models = model_snapshot()
    results = []
    save(pre / "models.json", models)
    with PoolGateway(
        models, port=native_gateway_port(), request_timeout=120
    ) as gateway:
        for system in args.systems:
            for policy in ["MaxQuality", "MinTime"] if system == "abacus" else [None]:
                name = system + ("-" + policy if policy else "")
                if args.attempt > 1:
                    name += "-attempt-" + str(args.attempt)
                save(
                    run / "status.json",
                    {
                        "phase": "preflight",
                        "reason": "Real-API two-row integration check: "
                        + name
                        + "; not included in the 228 full-input tasks",
                        "updated_unix": time.time(),
                    },
                )
                folder = pre / name
                if (folder / "job.json").exists():
                    raise ValueError(
                        "Preflight attempt already exists; preserve its logs and select a fresh name"
                    )
                folder.mkdir(exist_ok=True)
                allowed = (
                    [models[0]["alias"]]
                    if system in {"plop", "sema"}
                    else [m["alias"] for m in models]
                )
                save(
                    folder / "cache-reset.json",
                    {
                        "models": allowed,
                        "services": flush_model_caches(
                            [m for m in models if m["alias"] in allowed],
                            idle_timeout=20,
                        ),
                    },
                )
                endpoint = gateway.register(
                    name,
                    folder,
                    allowed=allowed,
                    aliases={
                        "gpt-5-mini": allowed[0],
                        "gpt-5-mini-2025-08-07": allowed[0],
                    }
                    if system == "plop"
                    else {},
                    default=system == "plop",
                )
                query = frozen["query_files"][system + "/" + args.query_id]
                job = {
                    **base,
                    "system": system,
                    "output": str(folder),
                    "query_file": query["path"],
                    "query_sha256": query["sha256"],
                    "input_database": str(db),
                    "inputs": inputs,
                    "model": allowed[0],
                    "models": [
                        {**m, "upstream_endpoint": m["endpoint"], "endpoint": endpoint}
                        for m in models
                    ],
                    "endpoint": endpoint,
                    "policy": policy,
                    "teacher_model": "semantic-large",
                    "timeout": 240,
                    "request_timeout": 120,
                    "max_tokens": 1024,
                    **job_resources(),
                    "native_mechanisms": True,
                    "native_mechanism_profile": "paper-mechanisms-v1",
                    "optimization_cost_unit": "parameter_weighted_tokens_per_1000; not USD",
                    "model_costs": {
                        m["alias"]: {"Input": weight, "Output": weight, "Cache": weight}
                        for m, weight in zip(models, (1.7, 2.0, 4.0, 12.0), strict=True)
                    },
                    "abacus_optimizer": {
                        "sample_budget": 8,
                        "k": 2,
                        "j": 2,
                        "judge_model": "semantic-large",
                        "seed": 42,
                    },
                    "optimizer": {
                        "do_logical_optimization": True,
                        "do_physical_optimization": True,
                        "available_models": [m["alias"] for m in models],
                        "max_rounds": 2,
                        "sample_size": 2,
                    },
                    "embedding": {
                        "backend": "sentence_transformers",
                        "model": str(
                            Path("/home/lyp/project/sglang/models/all-MiniLM-L6-v2")
                        ),
                        "device": "cpu",
                    },
                    "preflight_only": True,
                }
                save(folder / "job.json", job)
                command, env = worker_command(system, folder / "job.json")
                print("PREFLIGHT", name, flush=True)
                with (
                    (folder / "stdout.log").open("w") as out,
                    (folder / "stderr.log").open("w") as err,
                ):
                    p = subprocess.Popen(
                        command,
                        cwd=folder,
                        env=env,
                        stdout=out,
                        stderr=err,
                        start_new_session=True,
                    )
                    try:
                        p.wait(timeout=240)
                    except subprocess.TimeoutExpired:
                        from experiments.core.process import stop_stage

                        stop_stage(p, folder)
                gateway.close_run(name)
                while gateway.pending(name):
                    time.sleep(0.2)
                result = (
                    json.loads((folder / "summary.json").read_text())
                    if (folder / "summary.json").exists()
                    else {
                        "status": "failed",
                        "reason": (folder / "stderr.log").read_text()[-3000:],
                    }
                )
                result.update(gateway.summary(folder))
                result.update(system=system, policy=policy)
                if result["status"] == "completed" and result["model_requests"] == 0:
                    result.update(
                        status="failed",
                        reason="Preflight had nonempty semantic input but no model requests were observed",
                    )
                save(folder / "preflight-summary.json", result)
                results.append(result)
                print(
                    "RESULT",
                    name,
                    result.get("status"),
                    result.get("reason", ""),
                    result["model_requests"],
                    flush=True,
                )
    save(pre / "latest-preflight.json", results)
    passed = all(result["status"] == "completed" for result in results)
    save(
        run / "status.json",
        {
            "phase": "ready" if passed else "preflight_failed",
            "reason": "Preflight finished; inspect per-system integration evidence",
            "updated_unix": time.time(),
        },
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
