"""One complete upstream observation; HTTP retries retain the same request."""

import errno
import io
import json
import time
import urllib.error
import urllib.request


def send_completion(owner, model, forwarded, folder, route, record, trace, validate):
    forwarded = json.loads(json.dumps(forwarded))
    payload = json.dumps(forwarded, ensure_ascii=False).encode()
    trace["calls"] += 1
    request = urllib.request.Request(
        model["endpoint"].rstrip("/") + "/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + model.get("api_key", "EMPTY"),
            "Connection": "close",
            **model.get("headers", {}),
        },
    )
    # Bound only simultaneous upstream HTTP requests. Native
    # workers and prompt batching remain unchanged; queued
    # requests wait for admission in arrival order.
    upstream_semaphore = route.get("upstream_semaphore") or owner.upstream_semaphore
    upstream_admitted = upstream_semaphore is not None
    if upstream_admitted:
        upstream_semaphore.acquire()
    key = str(folder)
    with owner.lock:
        owner.inflight[key] = owner.inflight.get(key, 0) + 1
        owner.peak_inflight[key] = max(
            owner.peak_inflight.get(key, 0), owner.inflight[key]
        )
    record["upstream_started_unix"] = time.time()
    try:
        attempt = 0
        transport_retries = 0
        output_retries = 0
        rate_limit_retries = 0
        network_retries = 0
        compatibility_type = None
        while True:
            try:
                open_request = (
                    urllib.request.build_opener(urllib.request.ProxyHandler({})).open
                    if model.get("disable_proxy")
                    else urllib.request.urlopen
                )
                trace["attempts"] += 1
                with open_request(request, timeout=owner.request_timeout) as response:
                    result = json.load(response)
                trace["http_responses"] = trace.get("http_responses", 0) + 1
                # Keep the usage of each physical response, including responses
                # subsequently retried because they were incomplete or invalid.
                # A logical gateway request can cause more than one HTTP call.
                owner.record(
                    folder,
                    {
                        **record,
                        "event": "upstream_response",
                        "http_response_index": trace["http_responses"],
                        "upstream_attempt": trace["attempts"],
                        "usage": result.get("usage", {}),
                    },
                )
                reason = (result.get("choices") or [{}])[0].get("finish_reason")
                if (
                    reason == "length"
                    and output_retries < model.get("incomplete_output_retries", 0)
                    and forwarded["max_tokens"]
                    < model.get("max_retry_output_tokens", 16384)
                ):
                    previous_budget = forwarded["max_tokens"]
                    forwarded = {
                        **forwarded,
                        "max_tokens": min(
                            previous_budget * 2,
                            model.get("max_retry_output_tokens", 16384),
                        ),
                    }
                    owner.record(
                        folder,
                        {
                            **record,
                            "event": "retry",
                            "retry_index": attempt + 1,
                            "error": "Incomplete Teacher response: finish_reason=length",
                            "previous_max_tokens": previous_budget,
                            "next_max_tokens": forwarded["max_tokens"],
                            "response": result,
                            "usage": result.get("usage", {}),
                            "forwarded": forwarded,
                        },
                    )
                    request.data = json.dumps(forwarded, ensure_ascii=False).encode()
                    output_retries += 1
                    attempt += 1
                    continue
                break
            except (urllib.error.URLError, OSError) as error:
                reason = (
                    error.reason if isinstance(error, urllib.error.URLError) else error
                )
                network_budget = model.get("network_recovery_retries", 0)
                if (
                    network_budget
                    and isinstance(reason, OSError)
                    and reason.errno
                    in {
                        errno.ENETDOWN,
                        errno.ENETUNREACH,
                        errno.EHOSTUNREACH,
                    }
                ):
                    if network_retries >= network_budget:
                        raise
                    delay = min(15.0, 2.0**network_retries)
                    owner.record(
                        folder,
                        {
                            **record,
                            "event": "retry",
                            "retry_index": attempt + 1,
                            "retry_kind": "network_unreachable",
                            "error": str(error),
                            "retry_after_seconds": delay,
                        },
                    )
                    time.sleep(delay)
                    network_retries += 1
                    attempt += 1
                    continue
                if (
                    isinstance(error, urllib.error.HTTPError)
                    and error.code == 400
                    and route.get("retry_unavailable_format")
                    and forwarded.get("response_format", {}).get("type")
                    == "json_schema"
                    and forwarded["response_format"]
                    .get("json_schema", {})
                    .get("schema", {})
                    .get("type")
                    in {"string", "boolean"}
                ):
                    body = error.read()
                    detail = body.decode(errors="replace")
                    if (
                        "this response_format type is unavailable now"
                        not in detail.casefold()
                    ):
                        raise urllib.error.HTTPError(
                            error.url,
                            error.code,
                            error.msg,
                            error.headers,
                            io.BytesIO(body),
                        ) from error
                    if model.get("api_key"):
                        detail = detail.replace(model["api_key"], "[REDACTED]")
                    # Retry this one rejected transport option. The original
                    # request/schema still controls downstream validation/cache.
                    compatibility_type = forwarded["response_format"]["json_schema"][
                        "schema"
                    ]["type"]
                    forwarded = {
                        k: v for k, v in forwarded.items() if k != "response_format"
                    }
                    owner.record(
                        folder,
                        {
                            **record,
                            "event": "retry",
                            "retry_index": attempt + 1,
                            "error": "HTTP 400: " + detail,
                            "compatibility": "sema-response-format-unavailable-v1",
                            "forwarded": forwarded,
                            "messages_unchanged": True,
                        },
                    )
                    request.data = json.dumps(forwarded, ensure_ascii=False).encode()
                    attempt += 1
                    continue
                if (
                    isinstance(error, urllib.error.HTTPError)
                    and error.code == 429
                    and rate_limit_retries < model.get("rate_limit_retries", 0)
                ):
                    try:
                        delay = float(error.headers.get("Retry-After", ""))
                    except ValueError:
                        delay = min(60.0, 2.0**rate_limit_retries)
                    delay = max(0.0, min(60.0, delay))
                    detail = error.read().decode(errors="replace")
                    if model.get("api_key"):
                        detail = detail.replace(model["api_key"], "[REDACTED]")
                    owner.record(
                        folder,
                        {
                            **record,
                            "event": "retry",
                            "retry_index": attempt + 1,
                            "error": "HTTP 429: " + detail,
                            "retry_after_seconds": delay,
                        },
                    )
                    rate_limit_retries += 1
                    attempt += 1
                    time.sleep(delay)
                    continue
                if isinstance(error, urllib.error.HTTPError) and error.code < 500:
                    raise
                if transport_retries >= owner.upstream_retries:
                    raise
                owner.record(
                    folder,
                    {
                        **record,
                        "event": "retry",
                        "retry_index": attempt + 1,
                        "error": str(error),
                    },
                )
                time.sleep(min(2.0, 0.25 * (2**attempt)))
                attempt += 1
                transport_retries += 1
        record["upstream_attempts"] = attempt + 1
    finally:
        record["upstream_finished_unix"] = time.time()
        with owner.lock:
            owner.inflight[key] -= 1
        if upstream_admitted:
            upstream_semaphore.release()
    if result.get("model") not in {None, model["id"]} and not model.get(
        "allow_model_alias", False
    ):
        raise ValueError(f"Endpoint returned unexpected model {result.get('model')!r}")
    try:
        normalized = validate(result)
        if compatibility_type is not None:
            try:
                value = json.loads(normalized["choices"][0]["message"]["content"])
            except (ValueError, TypeError) as error:
                raise ValueError(
                    f"Expected JSON {compatibility_type} in Sema compatibility response"
                ) from error
            if (
                compatibility_type == "boolean"
                and type(value) is not bool
                or compatibility_type == "string"
                and not isinstance(value, str)
            ):
                raise ValueError(
                    f"Expected JSON {compatibility_type} in Sema compatibility response"
                )
    except ValueError:
        owner.record(
            folder,
            {
                **record,
                "event": "rejected_response",
                "response": result,
                "usage": result.get("usage", {}),
            },
        )
        raise
    usage = result.get("usage", {})
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        trace["usage"][key] = trace["usage"].get(key, 0) + (usage.get(key, 0) or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    if (
        trace.get("cached_usage_complete", True)
        and isinstance(cached, int)
        and not isinstance(cached, bool)
        and cached >= 0
    ):
        combined = trace["usage"].setdefault("prompt_tokens_details", {})
        combined["cached_tokens"] = combined.get("cached_tokens", 0) + cached
    else:
        # Do not report a partial sum as the complete upstream cache usage.
        trace["cached_usage_complete"] = False
        trace["usage"].pop("prompt_tokens_details", None)
    if route.get("response_cache") is not None:
        owner.record(
            folder,
            {
                **record,
                "event": "teacher_observation",
                "observation": trace["calls"],
                "response": result,
                "usage": usage,
            },
        )
    return result
