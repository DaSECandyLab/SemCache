"""Only serving-reported cache counters; no inference from prompt similarity."""


def cache_metrics(events):
    rows = [e for e in events if e.get("event") == "semantic_row"]
    known = [
        e
        for e in rows
        if e.get("cached_input_tokens") is not None and e.get("input_tokens", -1) >= 0
    ]
    cached = sum(e["cached_input_tokens"] for e in known)
    inputs = sum(e["input_tokens"] for e in known)
    return {
        "requests": len(rows),
        "reported_requests": len(known),
        "cached_input_tokens": cached if known else None,
        "input_tokens_on_reported_requests": inputs if known else None,
        "hit_fraction": cached / inputs if inputs else None,
        "coverage": len(known) / len(rows) if rows else None,
        "source": "usage.prompt_tokens_details.cached_tokens",
        "ttft_ms": None,
        "prefill_latency_ms": None,
    }
