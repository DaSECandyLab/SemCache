"""Read-only log audit and bounded prompt ablations; never rerun benchmark queries."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments.core.io import digest, save

SUFFIX = "\n Return a single yes or no, do not contain any other words."
STRING_SUFFIX = "\n Return a single word or phrase, do not contain any other words."
TARGETS = {
    "tpch/Q16": "You are validating whether this customer profile",
    "tpch/Q17": "You are identifying high-value B2B customer profiles.",
}
BATCH_SYSTEM = (
    "Evaluate each supplied semantic request independently. Follow each prompt's "
    "YES/NO answer requirements. Return ONLY a JSON array of objects with integer "
    "id and string answer, exactly once per input id."
)
FACTUAL_SYSTEM = "You are a factual reasoning assistant. Evaluate the supplied predicate. Return only YES or NO."


def read(path):
    return json.loads(path.read_text())


def boolean(text, strict=False):
    if not isinstance(text, str):
        return None
    text = text.lower() if strict else text.strip().lower()
    return {"yes": True, "true": True, "no": False, "false": False}.get(text)


def unjson(text):
    return json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip()))


def local_calls(path):
    submissions = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["event"] == "submit":
                submissions[row["request_id"]] = row
            elif row["event"] == "completed":
                request = submissions.pop(row["request_id"], None)
                if request:
                    choice = row["response"]["choices"][0]
                    yield (
                        request,
                        choice["message"].get("content"),
                        choice.get("finish_reason"),
                    )


def teacher_answers(path):
    answers = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row.get("output"), str):
                continue
            prompt = row["input"][-1]["content"]
            try:
                items, output = json.loads(prompt), unjson(row["output"])
                if not isinstance(items, list) or not isinstance(output, list):
                    continue
                mapped = {x["id"]: x["answer"] for x in output}
                for item in items:
                    answer = boolean(mapped.get(item["id"]))
                    if answer is not None:
                        answers[item["prompt"]] = answer
            except (ValueError, TypeError, KeyError):
                answer = boolean(row["output"])
                if answer is not None:
                    answers[prompt] = answer
    return answers


def audit(run):
    records = []
    for path in sorted((run / "tasks").glob("plop-*/*/*/attempt-1/gateway.jsonl")):
        groups = defaultdict(Counter)
        job = read(path.parent / "job.json")
        allowed = set(job["allowed"])
        expected_models = {m["id"] for m in job["models"] if m["alias"] in allowed}
        for request, text, finish in local_calls(path):
            prompt = request["requested"].get("input", "")
            if not isinstance(prompt, str):
                continue
            kind = "filter" if prompt.endswith(SUFFIX) else "projection"
            group = groups[kind]
            group["requests"] += 1
            group["truncated"] += finish == "length"
            if kind == "filter":
                decision = boolean(text, strict=True)
                group[
                    "YES"
                    if decision is True
                    else "NO"
                    if decision is False
                    else "parse_fallback"
                ] += 1
                group["whitespace_only_parse_failure"] += (
                    decision is None and boolean(text) is not None
                )
            else:
                group["single_word_response"] += (
                    isinstance(text, str) and len(text.split()) == 1
                )
            group["wrong_route"] += (
                request["actual_model"] not in expected_models
                or request["forwarded"]["model"] not in expected_models
                or request["routed_alias"] not in allowed
            )
        records.append(
            {
                "variant": path.parts[-5],
                "query": "/".join(path.parts[-4:-2]),
                "gateway_sha256": digest(path),
                "counts": dict(groups),
            }
        )
    return records


def prepare(run, references):
    samples = {}
    for query, prefix in TARGETS.items():
        answer = read(references / query.replace("/", "-") / "answer.json")
        teacher = teacher_answers(Path(answer["artifact"]) / "requests.jsonl")
        models = {}
        for variant in ("plop-e2b", "plop-large"):
            models[variant] = {
                r["requested"]["input"]: (r, text)
                for r, text, _ in local_calls(
                    run / "tasks" / variant / query / "attempt-1/gateway.jsonl"
                )
                if r["requested"].get("input", "").startswith(prefix)
            }
        shared = sorted(
            teacher.keys() & models["plop-e2b"].keys() & models["plop-large"].keys()
        )
        shared = [p for p in shared if p.startswith(prefix)]
        selected = []
        for label in (True, False):
            selected += [p for p in shared if teacher[p] is label][:4]
        if len(selected) < 8:
            selected += [p for p in shared if p not in selected][: 8 - len(selected)]
        if not selected:
            raise ValueError(f"No identical saved prompts shared with teacher: {query}")
        samples[query] = [
            {
                "prompt": p,
                "teacher": teacher[p],
                "saved_answers": {
                    variant: text
                    for variant, data in models.items()
                    for prompt, (_, text) in data.items()
                    if prompt == p
                },
            }
            for p in selected
        ]
    return samples


def request(model, prompts, mode):
    if mode == "native":
        messages = [{"role": "user", "content": prompts[0]}]
    elif mode == "without_duplicate_suffix":
        messages = [
            {
                "role": "user",
                "content": prompts[0].removesuffix(SUFFIX).removesuffix(STRING_SUFFIX),
            }
        ]
    elif mode == "factual_system":
        messages = [
            {"role": "system", "content": FACTUAL_SYSTEM},
            {"role": "user", "content": prompts[0]},
        ]
    else:
        messages = [
            {"role": "system", "content": BATCH_SYSTEM},
            {
                "role": "user",
                "content": json.dumps(
                    [{"id": i, "prompt": p} for i, p in enumerate(prompts)]
                ),
            },
        ]
    body = {
        "model": model["id"],
        "messages": messages,
        "temperature": 0,
        "max_tokens": 1024,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    started = time.monotonic()
    with opener.open(
        urllib.request.Request(
            model["endpoint"].rstrip("/") + "/chat/completions",
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        ),
        timeout=120,
    ) as response:
        result = json.load(response)
    text = result["choices"][0]["message"]["content"]
    if mode.startswith("batch"):
        output = unjson(text)
        if not isinstance(output, list) or sorted(x["id"] for x in output) != list(
            range(len(prompts))
        ):
            raise ValueError("Invalid diagnostic batch output")
        decisions = {x["id"]: boolean(x["answer"]) for x in output}
        labels = [decisions[i] for i in range(len(prompts))]
    else:
        labels = [boolean(text)]
    return {
        "seconds": time.monotonic() - started,
        "payload": body,
        "response": result,
        "labels": labels,
    }


def probe(run, samples, output):
    models = {x["alias"]: x for x in read(run / "models.json")}
    jobs = []
    for query, rows in samples.items():
        for alias in ("semantic-e2b", "semantic-large"):
            for mode in (
                "native",
                "without_duplicate_suffix",
                "factual_system",
                "batch1",
                "batch4",
            ):
                size = 4 if mode == "batch4" else 1
                for start in range(0, len(rows), size):
                    jobs.append((query, alias, mode, start, rows[start : start + size]))
    save(
        output / "probe-manifest.json",
        {
            "scope": "bounded operator prompts; not query quality estimates",
            "samples": samples,
            "http_requests": len(jobs),
            "cache": "No cache reset; diagnostic calls do not change benchmark artifacts",
        },
    )
    results = []

    def execute(job):
        query, alias, mode, start, rows = job
        record = {
            "query": query,
            "alias": alias,
            "mode": mode,
            "sample_start": start,
            "teacher_labels": [r["teacher"] for r in rows],
        }
        try:
            record.update(request(models[alias], [r["prompt"] for r in rows], mode))
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        return record

    with (
        ThreadPoolExecutor(max_workers=4) as pool,
        (output / "probes.jsonl").open("w") as stream,
    ):
        futures = [pool.submit(execute, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            record = future.result()
            results.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            if index % 20 == 0 or index == len(jobs):
                print(f"Probe HTTP requests {index}/{len(jobs)}", flush=True)
    groups = defaultdict(Counter)
    for row in results:
        group = groups[(row["query"], row["alias"], row["mode"])]
        if row.get("error"):
            group["http_or_format_errors"] += 1
            continue
        for actual, expected in zip(row["labels"], row["teacher_labels"]):
            group["samples"] += 1
            group["YES"] += actual is True
            group["NO"] += actual is False
            group["invalid"] += actual is None
            group["teacher_agreement"] += actual is expected
    save(
        output / "probe-summary.json",
        [
            {"query": q, "alias": a, "mode": m, **counts}
            for (q, a, m), counts in sorted(groups.items())
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output / "native-log-audit.json", audit(args.run))
    samples = prepare(args.run, args.references)
    save(args.output / "samples.json", samples)
    print(
        "Frozen shared-prompt samples:",
        {k: len(v) for k, v in samples.items()},
        flush=True,
    )
    if args.probe:
        probe(args.run, samples, args.output)


if __name__ == "__main__":
    main()
