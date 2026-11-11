"""Read physical JSONL records without splitting Unicode inside original text."""

import json


def sealed_gateway_events(gateway, token):
    """Only inspect a closed idle route, synchronized with request admission/writes."""
    with gateway.lock:
        route = gateway.routes[token]
        if not route["closed"] or token in gateway.active.values():
            raise ValueError("Evidence requires a closed, idle experiment route")
        path = route["folder"] / "gateway.jsonl"
        if not path.exists():
            return []
        with path.open() as stream:
            return [json.loads(line) for line in stream if line.strip()]
