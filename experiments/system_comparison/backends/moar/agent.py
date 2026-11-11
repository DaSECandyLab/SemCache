"""Configure MOAR's rewrite agent independently of its execution models."""

import json
import os
from pathlib import Path


def agent_model(path):
    config = json.loads(Path(path).read_text())
    base = config["base_url"].rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    overrides = {}
    if config.get("thinking"):
        overrides["thinking"] = {"type": config["thinking"]}
    if "reasoning_effort" in config:
        overrides["reasoning_effort"] = config["reasoning_effort"]
    headers = dict(config.get("headers", {}))
    if "opencode.ai" in base:
        # Its edge rejects Python-urllib's default user agent (HTTP 403/1010).
        headers.setdefault("User-Agent", "opencode/1.0")
    key = config.get("api_key") or os.environ.get(config.get("api_key_env", ""))
    if not key:
        raise ValueError("MOAR agent credential is missing")
    return {
        "alias": "semantic-rewriter", "id": config["model"], "endpoint": base,
        "context_length": config.get("context_length", 131072), "api_key": key,
        "headers": headers, "request_overrides": overrides, "disable_proxy": True,
        "allow_model_alias": True,
    }
