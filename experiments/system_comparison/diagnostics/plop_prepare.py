"""Bounded native/adapted PLOP string-prompt diagnostics, with no teacher calls.

Prepare with ``python -m experiments.system_comparison.diagnostics.plop_protocol``.
Only --execute contacts local models; preparation reads saved benchmark artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import ipaddress
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from experiments.core.io import digest
from experiments.system_comparison.transport.plop_protocol import (
    PLOP_TASK_OUTPUT_PROTOCOL,
    STRING_SUFFIX,
    make_plop_messages,
)

QUERIES = ("estate/Q5", "estate/Q7", "estate/Q8", "movies/Q12", "movies_sema/Q15")
VARIANTS = ("plop-small", "plop-e2b", "plop-e4b", "plop-large")
DEFAULT_REFERENCES = (
    Path(__file__).resolve().parents[3]
    / "experiments/results/reference_answers/systems/answers-20261030-1235/sema"
)
LIMIT = 64
SCOPE = (
    "Stable, bounded diagnostic sample of saved semantic_string operator inputs; "
    "not a representative accuracy sample or a full-query rerun. No teacher calls, "
    "new judgments, cache resets, or benchmark artifact changes. Q15 probes only "
    "the inner conflict-extraction stage; no final category accuracy or row F1 is measured. "
    "Literal equality to frozen text measures spelling/format agreement, not semantic quality."
)


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def prompt_hash(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()


def rows_by_id(result):
    names = [column[0] for column in result["columns"]]
    return {row[0]: dict(zip(names, row)) for row in result["rows"]}


def saved_strings(path):
    """Yield completed string requests, retaining the saved request and response."""
    pending = {}
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            if row.get("event") == "submit":
                prompt = row.get("requested", {}).get("input")
                if isinstance(prompt, str) and prompt.endswith(STRING_SUFFIX):
                    pending[row["request_id"]] = (number, row)
            elif row.get("event") == "completed":
                entry = pending.pop(row["request_id"], None)
                if entry:
                    yield entry[0], entry[1], row


def task_input(query, prompt):
    """Parse only the known native operator templates, without changing their text."""
    body = prompt.removesuffix(STRING_SUFFIX)
    body = body.split("\nOutput protocol: If the requested value is missing,", 1)[0]
    if query.startswith("estate/"):
        marker = "\nInputs: Details: "
        if marker in body:
            column = "Features" if query.endswith("Q7") else "Amenities"
            return "extraction", "Details", body.split(marker, 1)[1], column
    if query == "movies/Q12":
        marker = "\nInputs: Plot: "
        if marker in body:
            column = (
                "Genre"
                if body.startswith("According to the movie plot,")
                else "Character"
            )
            return "extraction", "Plot", body.split(marker, 1)[1], column
    if query == "movies_sema/Q15" and body.startswith(
        "Extract all conflicts from movie "
    ):
        suffix = " as one sentence."
        if body.endswith(suffix):
            overview = body[len("Extract all conflicts from movie ") : -len(suffix)]
            return "inner_conflict_extraction", "overview", overview, "conflict"
    return None


def movie_overviews():
    source = (
        Path(__file__).resolve().parents[3]
        / "benchmarks/datasets/movies/data/sema/csv/movie.csv"
    )
    indexed = defaultdict(list)
    with source.open(newline="") as stream:
        for row in csv.DictReader(stream):
            indexed[row["overview"]].append(int(row["movie_id"]))
    return source, indexed


def prepare(run, references, samples=16, aliases=("semantic-e2b", "semantic-large")):
    """Freeze stable inputs and align only identical input fields and record IDs."""
    if not 5 <= samples <= 16:
        raise ValueError("--samples must be between 5 and 16 to cover all five queries")
    if samples * len(aliases) * 2 > LIMIT:
        raise ValueError(f"Requested design exceeds {LIMIT} generation calls")
    if references.resolve() != DEFAULT_REFERENCES.resolve():
        raise ValueError(
            f"References are frozen for this diagnostic: {DEFAULT_REFERENCES}"
        )
    registry = {model["alias"]: model for model in read(run / "models.json")}
    if len(set(aliases)) != len(aliases) or not aliases:
        raise ValueError("Model aliases must be nonempty and unique")
    selected_models = []
    for alias in aliases:
        if alias not in registry:
            raise ValueError(f"Unknown model alias {alias!r}")
        model = registry[alias]
        endpoint = urllib.parse.urlparse(model["endpoint"])
        try:
            is_local = ipaddress.ip_address(endpoint.hostname or "").is_loopback
        except ValueError:
            is_local = endpoint.hostname == "localhost"
        if (
            endpoint.scheme != "http"
            or not is_local
            or endpoint.username
            or endpoint.password
        ):
            raise ValueError(
                "Diagnostics permit only credential-free local HTTP endpoints"
            )
        selected_models.append({key: model[key] for key in ("alias", "id", "endpoint")})
    overview_source, overview_ids = movie_overviews()
    pools, sources = {}, []
    for query in QUERIES:
        reference_path = references / query.replace("/", "-") / "answer.json"
        reference_rows = rows_by_id(read(reference_path))
        sources.append(
            {"path": str(reference_path.resolve()), "sha256": digest(reference_path)}
        )
        candidates = {}
        for variant in VARIANTS:
            directory = run / "tasks" / variant / query / "attempt-1"
            gateway = directory / "gateway.jsonl"
            if not gateway.exists():
                continue
            actual_path = directory / "result.json"
            actual_rows = rows_by_id(read(actual_path)) if actual_path.exists() else {}
            sources.append({"path": str(gateway.resolve()), "sha256": digest(gateway)})
            if actual_path.exists():
                sources.append(
                    {"path": str(actual_path.resolve()), "sha256": digest(actual_path)}
                )
            for line, request, response in saved_strings(gateway):
                prompt = request["requested"]["input"]
                parsed = task_input(query, prompt)
                if parsed is None:
                    continue  # Q15 outer-stage inputs depend on earlier model outputs.
                stage, input_column, input_text, output_column = parsed
                if query == "movies_sema/Q15":
                    ids = [
                        identifier
                        for identifier in overview_ids.get(input_text, [])
                        if identifier in actual_rows and identifier in reference_rows
                    ]
                else:
                    ids = [
                        identifier
                        for identifier, row in actual_rows.items()
                        if row.get(input_column) == input_text
                        and identifier in reference_rows
                        and reference_rows[identifier].get(input_column) == input_text
                    ]
                key = prompt_hash(prompt)
                item = candidates.setdefault(
                    key,
                    {
                        "sample_id": f"{query.replace('/', '-')}-{key[:12]}",
                        "query": query,
                        "stage": stage,
                        "input_sha256": key,
                        "original_prompt": prompt,
                        "input_column": input_column,
                        "output_column": output_column,
                        "record_ids": [],
                        "saved_calls": [],
                        "frozen_reference": {
                            "path": str(reference_path.resolve()),
                            "same_stage": stage == "extraction",
                            "literal_text": None,
                            "alignment": "unavailable",
                        },
                    },
                )
                item["record_ids"] = sorted(set(item["record_ids"]) | set(ids), key=str)
                item["saved_calls"].append(
                    {
                        "variant": variant,
                        "gateway": str(gateway.resolve()),
                        "line": line,
                        "request_id": request["request_id"],
                        "actual_model": request.get("actual_model"),
                        "saved_response": response.get("response"),
                    }
                )
                if ids:
                    frozen = item["frozen_reference"]
                    frozen["alignment"] = "same_id_and_identical_input_field"
                    if stage == "extraction":
                        texts = [
                            reference_rows[identifier].get(output_column)
                            for identifier in item["record_ids"]
                        ]
                        # Ambiguous repeated contexts receive no literal comparison.
                        if all(text == texts[0] for text in texts):
                            frozen["literal_text"] = texts[0]
                            frozen["value_is_sql_null"] = texts[0] is None
                            frozen["literal_comparable"] = isinstance(texts[0], str)
                        else:
                            frozen["literal_comparable"] = False
                    else:
                        frozen["literal_comparable"] = False
                        frozen["note"] = (
                            "Frozen answer contains final categories, not inner conflict text."
                        )
        aligned = [item for item in candidates.values() if item["record_ids"]]
        pools[query] = sorted(
            aligned or candidates.values(),
            key=lambda item: (len(item["record_ids"]) != 1, item["input_sha256"]),
        )
        if not pools[query]:
            raise ValueError(f"No completed saved semantic_string inputs for {query}")
    chosen = []
    # Round robin query quotas, then a stable input hash within each query.
    offsets = Counter()
    while len(chosen) < samples:
        before = len(chosen)
        for query in QUERIES:
            if offsets[query] < len(pools[query]) and len(chosen) < samples:
                chosen.append(pools[query][offsets[query]])
                offsets[query] += 1
        if len(chosen) == before:
            raise ValueError(
                "Insufficient distinct saved inputs for the requested sample count"
            )
    for item in chosen:
        item["messages"] = {}
        for condition, protocol in (
            ("native", None),
            ("adapted", PLOP_TASK_OUTPUT_PROTOCOL),
        ):
            messages, evidence = make_plop_messages(
                {"input": item["original_prompt"]}, protocol=protocol
            )
            item["messages"][condition] = {
                "messages": messages,
                "protocol_evidence": evidence,
            }
    return {
        "created_utc": utc(),
        "scope": SCOPE,
        "run": str(run.resolve()),
        "references": str(references.resolve()),
        "protocol": PLOP_TASK_OUTPUT_PROTOCOL,
        "selection": "Round robin queries; prefer same-ID identical-input availability, then SHA256 of original input. No reference answer values or model correctness used in selection.",
        "models": selected_models,
        "samples": chosen,
        "generation_call_limit": LIMIT,
        "generation_calls_planned": len(chosen) * len(aliases) * 2,
        "settings": {
            "temperature": 0,
            "max_tokens": 1024,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "protected_artifacts": [
            {"path": str(path.resolve()), "sha256": digest(path)}
            for path in [
                references / "contract.json",
                *sorted(references.glob("*/answer.json")),
                *sorted(
                    (Path(__file__).resolve().parents[3] / "experiments/results").glob(
                        "system-comparison-20261028-0033.*"
                    )
                ),
            ]
        ],
        "sources": sources
        + [
            {
                "path": str((run / "models.json").resolve()),
                "sha256": digest(run / "models.json"),
            },
            {"path": str(overview_source.resolve()), "sha256": digest(overview_source)},
        ],
    }


def metrics(text, sample):
    available = isinstance(text, str)
    frozen = sample["frozen_reference"]
    comparable = bool(frozen.get("literal_comparable"))
    stripped = text.strip() if available else ""
    words = len(re.findall(r"\S+", text)) if available else None
    terminal = (
        len(re.findall(r"[.!?]+(?:[\"')\]]*)?(?=\s|$)", stripped))
        if available
        else None
    )
    return {
        "content_is_string": available,
        "sql_null_token": text == "__SQL_NULL__" if available else None,
        "empty_string": text == "" if available else None,
        "whitespace_only": bool(text) and not stripped if available else None,
        "word_count": words,
        "terminal_sentence_markers": terminal,
        "single_sentence_surface_heuristic": (
            available
            and words >= 3
            and terminal == 1
            and bool(re.search(r"[.!?][\"')\]]*$", stripped))
        )
        if sample["stage"] == "inner_conflict_extraction"
        else None,
        "frozen_value_is_sql_null": frozen.get("value_is_sql_null"),
        "frozen_sql_null_consistent": (text == "__SQL_NULL__")
        if available and frozen.get("value_is_sql_null") is True
        else None,
        "frozen_literal_comparable": comparable,
        "frozen_literal_equal": text == frozen.get("literal_text")
        if comparable and available
        else None,
    }


def http_json(endpoint, path, body=None):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        endpoint.rstrip("/") + path,
        None if body is None else json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with opener.open(request, timeout=120) as response:
        return json.load(response)


def progress(output, state, planned, started, status):
    text = (
        f"# PLOP protocol diagnostic\n\nStatus: {status}\n\n"
        f"Updated: {utc()}\n\nGeneration calls: {state['completed']}/{planned} completed; "
        f"{state['started']} started; {state['errors']} errors.\n\n"
        f"Elapsed: {time.monotonic() - started:.1f}s.\n\n{SCOPE}\n"
    )
    temporary = output / "calls.md.tmp"
    temporary.write_text(text)
    temporary.replace(output / "calls.md")
