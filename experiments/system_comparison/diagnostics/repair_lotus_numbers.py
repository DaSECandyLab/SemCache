"""Finish failed LOTUS numeric serialization from saved complete aggregate inputs.

No query or model request is replayed. Original snapshots, errors, process exit
codes and wall clocks remain available as evidence.
"""

import argparse
import json
from pathlib import Path
import time

import pandas as pd

from experiments.core.io import digest, save
from experiments.system_comparison.backends.lotus.cascade import save_plan
from experiments.system_comparison.backends.lotus.compat import normalize_numeric_columns
from experiments.system_comparison.reporting.results import normalize


def repair(folder):
    summary_path = folder / "summary.json"
    summary = json.loads(summary_path.read_text())
    job = json.loads((folder / "job.json").read_text())
    if summary.get("postprocessing_recovered"):
        return {"query": job["id"], "already_repaired": True}
    if not (
        job["system"] == "lotus" and job.get("skip_final_aggregate")
        and summary.get("parent_finalized") and summary["status"] == "failed"
        and summary.get("reason") == "ValueError: could not convert string to float: 'null'"
        and summary.get("model_errors", 0) == 0
        and summary.get("response_delivery_errors", 0) == 0
    ):
        raise ValueError("This repair requires completed LOTUS inference and the exact numeric-null serialization failure")
    snapshot = folder / "pre-aggregate-result.json"
    raw = json.loads(snapshot.read_text())
    schema = job["pre_aggregate_schema"]
    names = [c[0] for c in schema]
    if [c[0] for c in raw["columns"]] != names:
        raise ValueError("Saved aggregate input fields differ from the frozen contract")
    started = time.monotonic()
    frame = pd.DataFrame(raw["rows"], columns=names, dtype=object)
    fixed, conversion = normalize_numeric_columns(frame, schema)
    payload = normalize({"columns": schema, "rows": list(fixed.itertuples(index=False, name=None))})
    timing_path = folder / "timing.json"
    timing = json.loads(timing_path.read_text())
    loading = json.loads((folder / "input-loading.json").read_text())
    evidence = {
        "query": job["id"], "original_summary": summary,
        "original_snapshot_sha256": digest(snapshot),
        "conversion": conversion, "rows": len(payload["rows"]),
        "model_requests_added": 0, "query_reexecuted": False,
        "repair_source_sha256": digest(Path(__file__)),
        "numeric_adapter_sha256": digest(Path(__file__).parents[1] / "backends/lotus/compat.py"),
        "recovered_unix": time.time(),
    }
    raw_path = folder / "pre-aggregate-raw-result.json"
    if raw_path.exists():
        raise ValueError("Raw recovery artifact already exists; inspect before retrying")
    save(folder / "postprocessing-repair.json", evidence)
    snapshot.rename(raw_path)
    save(snapshot, payload)
    save(folder / "result.json", payload)
    save(folder / "numeric-output-conversion.json", conversion)
    save_plan(job)
    recovery_seconds = time.monotonic() - started
    timing.update(load_seconds=loading["load_seconds"])
    timing.setdefault("details", {}).update(
        postprocessing_recovery=True, recovery_seconds=recovery_seconds,
        original_inference_times_preserved=True,
    )
    save(timing_path, timing)
    summary.pop("reason", None)
    summary.update(
        status="completed", postprocessing_recovered=True,
        seconds=timing["query_seconds"], load_seconds=loading["load_seconds"],
        result_rows=len(payload["rows"]), timing=timing,
        pre_aggregate={"path": str(snapshot), "sha256": digest(snapshot)},
        postprocessing_repair=str(folder / "postprocessing-repair.json"),
    )
    save(summary_path, summary)
    return {"query": job["id"], "rows": len(payload["rows"]), "conversion": conversion, "HTTP_added": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--queries", nargs="+", required=True)
    parser.add_argument("--variant", default="lotus-two-stage")
    args = parser.parse_args()
    for query in args.queries:
        folder = (args.run / "tasks" / args.variant / query / "attempt-1").resolve()
        if not folder.is_relative_to((args.run / "tasks" / args.variant).resolve()):
            raise ValueError("Query escaped LOTUS task folder")
        print(json.dumps(repair(folder), ensure_ascii=False))


if __name__ == "__main__":
    main()
