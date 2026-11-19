"""Bounded native/adapted PLOP string-prompt diagnostics, with no teacher calls.

Prepare with ``python -m experiments.system_comparison.diagnostics.plop_protocol``.
Only --execute contacts local models; preparation reads saved benchmark artifacts.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from experiments.core.io import digest, save

from .plop_prepare import (
    DEFAULT_REFERENCES as DEFAULT_REFERENCES,
)
from .plop_prepare import (
    LIMIT as LIMIT,
)
from .plop_prepare import (
    QUERIES as QUERIES,
)
from .plop_prepare import (
    SCOPE as SCOPE,
)
from .plop_prepare import (
    VARIANTS as VARIANTS,
)
from .plop_prepare import (
    http_json as http_json,
)
from .plop_prepare import (
    metrics as metrics,
)
from .plop_prepare import (
    prepare as prepare,
)
from .plop_prepare import (
    progress as progress,
)
from .plop_prepare import (
    read as read,
)
from .plop_prepare import (
    utc as utc,
)


def report(manifest, records, status):
    groups = defaultdict(list)
    for row in records:
        groups[(row["query"], row["alias"], row["condition"])].append(row)
    summaries = []
    lines = [
        "# PLOP protocol diagnostic",
        "",
        f"Status: {status}",
        "",
        SCOPE,
        "",
        f"Generation calls planned: {manifest['generation_calls_planned']}; completed: {len(records)}.",
        "",
        "Selection uses source input alignment and stable hashes, never reference answer classes.",
        "",
        "| Query | Model | Condition | Calls | Errors | NULL | Empty | Median words | Literal equal / comparable |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (query, alias, condition), rows in sorted(groups.items()):
        counts = Counter()
        word_counts = []
        for row in rows:
            counts["calls"] += 1
            counts["errors"] += bool(row.get("error"))
            m = row.get("metrics", {})
            counts["sql_null_token"] += m.get("sql_null_token") is True
            counts["empty_string"] += m.get("empty_string") is True
            counts["literal_comparable_successful"] += (
                m.get("frozen_literal_equal") is not None
            )
            counts["literal_equal"] += m.get("frozen_literal_equal") is True
            if m.get("word_count") is not None:
                word_counts.append(m["word_count"])
        median = statistics.median(word_counts) if word_counts else None
        summaries.append(
            {
                "query": query,
                "alias": alias,
                "condition": condition,
                **counts,
                "median_word_count": median,
            }
        )
        lines.append(
            f"| {query} | {alias} | {condition} | {counts['calls']} | {counts['errors']} | "
            f"{counts['sql_null_token']} | {counts['empty_string']} | {median} | "
            f"{counts['literal_equal']} / {counts['literal_comparable_successful']} |"
        )
    lines += [
        "",
        "## Per-sample literal comparisons",
        "",
        "`null` means no comparable same-stage frozen text or no successful string response. "
        "Q15 inner-stage output cannot be compared to final category text.",
        "",
        "| Sample | Model | Condition | Literal equal | Words | One-sentence surface heuristic |",
        "|---|---|---|---|---:|---|",
    ]
    for row in sorted(
        records, key=lambda r: (r["sample_id"], r["alias"], r["condition"])
    ):
        m = row.get("metrics", {})

        def render(key):
            return json.dumps(m.get(key))

        lines.append(
            f"| {row['sample_id']} | {row['alias']} | {row['condition']} | "
            f"{render('frozen_literal_equal')} | {render('word_count')} | "
            f"{render('single_sentence_surface_heuristic')} |"
        )
    lines += [
        "",
        "The sentence metric counts terminal punctuation and words only; it does not judge grammar, "
        "completeness, factuality, extraction recall, category correctness, or semantic quality. "
        "No quality improvement conclusion follows from these descriptive counts.",
        "",
    ]
    return {
        "status": status,
        "scope": SCOPE,
        "completed_generation_calls": len(records),
        "planned_generation_calls": manifest["generation_calls_planned"],
        "groups": summaries,
    }, "\n".join(lines)


def execute(manifest, output, workers=4):
    """Validate each selected local model, then run exactly one paired call per input."""
    validations = []
    for model in manifest["models"]:
        try:
            response = http_json(model["endpoint"], "/models")
            ids = [entry["id"] for entry in response.get("data", [])]
            validations.append(
                {
                    "alias": model["alias"],
                    "expected_id": model["id"],
                    "response": response,
                    "matches": model["id"] in ids,
                    "checked_utc": utc(),
                }
            )
        except Exception as error:
            validations.append(
                {
                    "alias": model["alias"],
                    "matches": False,
                    "error": f"{type(error).__name__}: {error}",
                    "checked_utc": utc(),
                }
            )
    save(output / "model-verification.json", validations)
    if not all(row["matches"] for row in validations):
        raise ValueError(
            "GET /models identity verification failed; no generation calls made"
        )
    jobs = [
        (sample, model, condition)
        for sample in manifest["samples"]
        for model in manifest["models"]
        for condition in ("native", "adapted")
    ]
    if len(jobs) > LIMIT:
        raise ValueError("Generation call budget exceeded")
    state = {"started": 0, "completed": 0, "errors": 0}
    lock = threading.Lock()
    started = time.monotonic()
    records = []

    def call(job):
        sample, model, condition = job
        with lock:
            state["started"] += 1
        sent = {
            "model": model["id"],
            **manifest["settings"],
            "messages": sample["messages"][condition]["messages"],
        }
        row = {
            "sample_id": sample["sample_id"],
            "query": sample["query"],
            "stage": sample["stage"],
            "input_sha256": sample["input_sha256"],
            "original_prompt": sample["original_prompt"],
            "alias": model["alias"],
            "condition": condition,
            "endpoint": model["endpoint"],
            "payload": sent,
            "protocol_evidence": sample["messages"][condition]["protocol_evidence"],
            "frozen_reference": sample["frozen_reference"],
            "started_utc": utc(),
        }
        tick = time.monotonic()
        try:
            row["response"] = http_json(model["endpoint"], "/chat/completions", sent)
            if row["response"].get("model") not in (None, model["id"]):
                raise ValueError(
                    "Generation response model differs from verified requested model"
                )
            choice = row["response"]["choices"][0]
            row["text"] = choice["message"].get("content")
            row["finish_reason"] = choice.get("finish_reason")
            row["metrics"] = metrics(row["text"], sample)
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            if isinstance(error, urllib.error.HTTPError):
                row["error_response"] = error.read().decode(errors="replace")
            row["metrics"] = metrics(None, sample)
        row["seconds"] = time.monotonic() - tick
        row["completed_utc"] = utc()
        return row

    progress(output, state, len(jobs), started, "running")
    last_progress = time.monotonic()
    with (
        ThreadPoolExecutor(max_workers=workers) as pool,
        (output / "probes.jsonl").open("x") as stream,
    ):
        pending = {pool.submit(call, job) for job in jobs}
        while pending:
            finished, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in finished:
                row = future.result()
                records.append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                with lock:
                    state["completed"] += 1
                    state["errors"] += bool(row.get("error"))
            if time.monotonic() - last_progress >= 10:
                with lock:
                    progress(output, state, len(jobs), started, "running")
                last_progress = time.monotonic()
                print(
                    f"Generation calls {state['completed']}/{len(jobs)}; errors={state['errors']}",
                    flush=True,
                )
    status = "completed_with_errors" if state["errors"] else "completed"
    progress(output, state, len(jobs), started, status)
    summary, markdown = report(manifest, records, status)
    summary["elapsed_seconds"] = time.monotonic() - started
    integrity = [
        {
            **artifact,
            "sha256_after": digest(artifact["path"]),
            "unchanged": digest(artifact["path"]) == artifact["sha256"],
        }
        for artifact in manifest["protected_artifacts"]
    ]
    save(output / "protected-artifacts-verification.json", integrity)
    summary["protected_artifacts_unchanged"] = all(
        item["unchanged"] for item in integrity
    )
    markdown += (
        "\nProtected reference and formal report artifacts unchanged: "
        + str(summary["protected_artifacts_unchanged"])
        + "\n"
    )
    save(output / "summary.json", summary)
    (output / "report.md").write_text(markdown)
    print(
        f"Generation calls {state['completed']}/{len(jobs)}; errors={state['errors']}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--models", nargs="+", default=["semantic-e2b", "semantic-large"]
    )
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--workers", type=int, choices=range(4, 9), default=4)
    args = parser.parse_args()
    if "diagnostics" not in args.output.resolve().parts:
        parser.error("--output must be a new diagnostics subdirectory")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(
            "--output must be empty; existing diagnostic artifacts are never overwritten"
        )
    aliases = tuple(
        alias for value in args.models for alias in value.split(",") if alias
    )
    manifest = prepare(args.run, args.references, args.samples, aliases)
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output / "manifest.json", manifest)
    summary, markdown = report(manifest, [], "prepared; no API requests made")
    save(args.output / "summary.json", summary)
    (args.output / "report.md").write_text(markdown)
    progress(
        args.output,
        {"started": 0, "completed": 0, "errors": 0},
        manifest["generation_calls_planned"],
        time.monotonic(),
        "prepared; no API requests made",
    )
    print(
        f"Prepared {len(manifest['samples'])} stable inputs; "
        f"{manifest['generation_calls_planned']} generation calls planned (hard cap {LIMIT}).",
        flush=True,
    )
    if args.execute:
        try:
            execute(manifest, args.output, args.workers)
        except Exception as error:
            summary["status"] = "failed_before_or_during_execution"
            summary["error"] = f"{type(error).__name__}: {error}"
            save(args.output / "summary.json", summary)
            (args.output / "report.md").write_text(
                markdown + f"\nExecution failure: {summary['error']}\n"
            )
            raise


if __name__ == "__main__":
    main()
