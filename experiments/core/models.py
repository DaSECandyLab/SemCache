"""Read actual model identities and context limits from the local APIs."""

import json


def model_snapshot(*, strongest_only=False):
    import concurrent.futures
    import urllib.request

    aliases = ("semantic-e2b", "semantic-e4b", "semantic-large")
    ports = (30001, 30002, 30003)
    pairs = list(zip(aliases, ports, strict=True))
    if strongest_only:
        pairs = pairs[-1:]

    def probe(pair):
        alias, port = pair
        endpoint = f"http://127.0.0.1:{port}/v1"
        with urllib.request.urlopen(endpoint + "/models", timeout=10) as response:
            response = json.load(response)
        if len(response["data"]) != 1:
            raise ValueError("Each formal endpoint must identify one model")
        identity = response["data"][0]
        context_length = identity.get("max_model_len") or identity.get("context_length")
        if not isinstance(context_length, int) or context_length <= 0:
            raise ValueError(
                f"Model endpoint {endpoint} did not report a positive context limit"
            )
        return {
            "alias": alias,
            "id": identity["id"],
            "endpoint": endpoint,
            "model_response": response,
            "context_length": context_length,
            "token_dollar_cost": 0,
            "cost_interpretation": "local API, zero metered dollar charge; no resource price assigned",
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(pairs)) as executor:
        models = list(executor.map(probe, pairs))
    if len({model["id"] for model in models}) != len(models):
        raise ValueError(
            "Formal endpoints must expose distinct actual model IDs"
        )
    return models
