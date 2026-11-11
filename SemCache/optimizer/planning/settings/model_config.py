"""Shared candidate-model discovery for training and online inference."""

import json
import re
import urllib.request

from planning.settings.configuration import read_json


def discover_models(endpoints, *, max_output_tokens=512):
    """Freeze the actual served identities, rather than port nicknames."""
    if type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 262144:
        raise ValueError("max_output_tokens must be an integer in [1, 262144]")
    models, catalog = {}, {}
    for endpoint in endpoints:
        with urllib.request.urlopen(
            endpoint.rstrip("/") + "/v1/models", timeout=10
        ) as response:
            entries = json.load(response)["data"]
        if len(entries) != 1:
            raise ValueError(f"Expected exactly one served model at {endpoint}")
        entry = entries[0]
        name = entry["id"]
        if name in models:
            raise ValueError("Endpoints must serve distinct model identities")
        size = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?:\b|-)", name)
        if not size:
            raise ValueError(f"Configure parameter count for teacher selection: {name}")
        models[name] = {
            "base_url": endpoint,
            "model": name,
            "structured_output": True,
            "enable_thinking": False,
            "max_tokens": max_output_tokens,
        }
        catalog[name] = {
            "parameters": float(size[1]) * 1e9,
            "context_length": entry.get("max_model_len"),
            "gpu_weight_bytes": None,
            "gpu_kv_reserve_bytes": None,
        }
    return models, catalog


def discover_configured_models(path):
    """Read a provider-neutral candidate pool and resolve served model identities."""
    config = read_json(path)
    config = config.get("models", config)
    endpoints = config.get("endpoints")
    if (
        not isinstance(endpoints, list)
        or not endpoints
        or any(not isinstance(url, str) or not url for url in endpoints)
    ):
        raise ValueError("models.endpoints must be a nonempty list of API base URLs")
    return discover_models(endpoints, max_output_tokens=config["max_output_tokens"])


def unavailable_models(models):
    """Check connectivity and frozen identities without inference or cache changes."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    missing = []
    for name, config in models.items():
        try:
            with opener.open(
                config["base_url"].rstrip("/") + "/v1/models", timeout=5
            ) as response:
                identities = {entry["id"] for entry in json.load(response)["data"]}
            if config["model"] not in identities:
                missing.append(name)
        except (OSError, ValueError, KeyError, TypeError):
            missing.append(name)
    return missing
