"""Fit operator mean prefix-match lengths from saved features and public usage."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from planning.execution.cache.features import operator_context, semantic_context
from planning.execution.cache.prefix import prefix_context
from training.artifacts import read

# Versions 16 and 18 share the same 105 numeric values and cache encoding;
# their differences concern the value network's remaining-operator encoder.
CACHE_FEATURE_VERSIONS = {16, 18}


class SkippedObservation(ValueError):
    """An unexecuted timeout has no observed cache-token label."""


def episode_telemetry(leaf, source):
    """Merged episodes own their copied execution, not the old absolute path."""
    source = Path(source)
    if "sql" not in source.parts:
        raise ValueError("Cache sidecar requires a SQL execution artifact")
    local = (
        leaf.joinpath(*source.parts[source.parts.index("sql") :]) / "telemetry.jsonl"
    )
    return local if local.is_file() else source / "telemetry.jsonl"


def collection_status(run):
    path = run / "pipeline-status.json"
    return read(path if path.exists() else run / "status.json")


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def decode(value, valid=True):
    if not valid or not number(value) or value > 700:
        return None
    return math.expm1(value)


def reuse_features(node):
    """Recover the original numeric features from the frozen model input.

    Retain all existing reductions so a later study need not rerun queries.
    Empty previous-source sets represent known zero reuse. Missing values stay
    unknown and cannot become a misleading zero training label.
    """
    numeric, history = node["numeric"], node.get("l_existing")
    previous = {
        str(source): decode(value)
        for source, value in node.get("l_previous", {}).items()
    }
    previous_valid = all(v is not None for v in previous.values())
    values = list(previous.values()) if previous_valid else []
    existing = isinstance(history, list) and len(history) == 18
    return {
        "l_self": decode(numeric[-3], numeric[-1] == 1),
        "l_previous": max(values, default=0.0) if previous_valid else None,
        "l_existing": decode(history[1], history[13] == 1) if existing else None,
        "l_previous_by_source": previous,
        "l_previous_sum": math.fsum(values) if previous_valid else None,
        "l_existing_mean": decode(history[0], history[12] == 1) if existing else None,
        "l_existing_sources": round(decode(history[2], history[14] == 1))
        if existing and history[14] == 1
        else None,
        "l_self_reliability": numeric[-2],
        "encoded_cache_features": {
            "self": numeric[-3:],
            "previous": node.get("l_previous", {}),
            "existing": history,
        },
        **operator_context(node),
    }


def extract_episode(leaf, run):
    payload = read(leaf / "experience.json")
    if payload.get("feature_version") not in CACHE_FEATURE_VERSIONS:
        raise ValueError("Cache feature layout changed; update the explicit decoder")
    (record,) = payload["records"]
    query = read(leaf / "query_ir.json")
    features = record["trajectory"][-1]["features"]["nodes"]
    actions = {a["operator_id"]: a for a in record["plan"]["actions"]}
    sources = record["metadata"]["timing_artifacts"]
    if len(sources) != 1:
        raise ValueError("Cache sidecar requires one physical execution per episode")
    telemetry = episode_telemetry(leaf, sources[0])
    if not telemetry.is_file() and record["metadata"].get("execution_skipped"):
        raise SkippedObservation("execution_skipped_without_request_telemetry")
    # Only LF separates JSONL records. Unicode line-separator characters can
    # occur inside quoted input data and must not split one response in two.
    with telemetry.open() as stream:
        identity, rows, requests = extract_observations(
            stream,
            leaf,
            run,
            record,
            query,
            features,
            actions,
            payload["feature_version"],
        )
    profile_path = leaf / "profile.json"
    profile = read(profile_path) if profile_path.exists() else {}
    for row in rows:
        op = next(
            op
            for op in query["semantic_operators"]
            if op["operator_id"] == row["operator_id"]
        )
        row.update(semantic_context(op, actions[row["operator_id"]], profile))
        row.update(
            prefix_context(
                features[str(op["node_id"])],
                op,
                actions[row["operator_id"]],
                profile,
                row,
            )
        )
    return identity, rows, requests


def extract_observations(
    stream, leaf, run, record, query, features, actions, feature_version
):
    counters, requests = {}, []
    for line in stream:
        if not line.strip():
            continue
        event = json.loads(line)
        # Cache-answer hits/reuses have no public usage of a new HTTP response.
        if (
            event.get("event") != "semantic_row"
            or event.get("parse_status") != "validated"
        ):
            continue
        oid = event["operator_id"]
        action = actions[oid]
        if (
            event.get("model_id") != action["model_id"]
            or event.get("config_generation", 0) != 0
        ):
            raise ValueError(
                "Static training observation differs from frozen model/generation"
            )
        for name, order in action["layout"].items():
            if event.get(name) != order:
                raise ValueError(
                    "Static training response differs from frozen prompt layout"
                )
        bank = counters.setdefault(oid, Counter())
        bank["responses"] += 1
        tokens, prompt = event.get("cached_input_tokens"), event.get("input_tokens")
        usage_valid = number(tokens) and number(prompt) and tokens <= prompt
        if usage_valid:
            bank["usage_responses"] += 1
            bank["cached_tokens"] += tokens
            bank["prompt_tokens"] += prompt
        if number(event.get("latency_ms")):
            bank["latency_ms"] += event["latency_ms"]
        requests.append(
            {
                "operator_id": oid,
                "request_seq": event["request_seq"],
                "cached_tokens": tokens if usage_valid else None,
                "prompt_tokens": prompt if number(prompt) else None,
                "latency_ms": event.get("latency_ms"),
                "usage_known": usage_valid,
                "parse_status": event.get("parse_status"),
            }
        )
    relative = str(leaf.relative_to(run))
    episode_id = hashlib.sha256(relative.encode()).hexdigest()
    rows = []
    for op in query["semantic_operators"]:
        oid = op["operator_id"]
        node = features.get(str(op["node_id"]))
        if node is None or len(node["numeric"]) != 105:
            raise ValueError("Frozen final state has no semantic node features")
        stats = counters.get(oid, Counter())
        count, covered = stats["responses"], stats["usage_responses"]
        known = covered == count and count > 0
        f = reuse_features(node)
        row = {
            "sample_id": f"{episode_id}:{oid}",
            "episode": relative,
            "source_feature_version": feature_version,
            "round": int(leaf.parents[2].name.removeprefix("round-")),
            "query": record["metadata"]["workload"],
            "dataset": record["dataset"],
            "search": int(leaf.name.removeprefix("search-")),
            "operator_id": oid,
            "operator_kind": op["kind"],
            "model_id": actions[oid]["model_id"],
            "layout": actions[oid]["layout"],
            "query_status": record["status"],
            **f,
            "http_response_observations": count,
            "usage_known_responses": covered,
            "usage_coverage": covered / count if count else None,
            "cached_tokens_sum": stats["cached_tokens"] if known else None,
            "mean_cached_tokens": stats["cached_tokens"] / count if known else None,
            "prompt_tokens_sum": stats["prompt_tokens"] if known else None,
            "training_eligible": known
            and record["status"] == "completed"
            and all(
                f[name] is not None for name in ("l_self", "l_previous", "l_existing")
            ),
            "feature_scope": "frozen pre-execution operator mean prefix lengths in characters",
            "label_scope": "mean public cached tokens over consumed HTTP responses",
            "chunk_features_available": False,
        }
        rows.append(row)
    ids = {r["operator_id"]: r["sample_id"] for r in rows}
    for request in requests:
        request["sample_id"] = ids[request["operator_id"]]
    return episode_id, rows, requests
