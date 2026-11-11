"""Strict four-model routing with uncached, complete request evidence."""

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from experiments.system_comparison.transport.plop_protocol import (
    PLOP_PROMPT_PROTOCOLS,
    make_plop_messages,
)


def normalize_alias(alias):
    return alias.removeprefix("openai/")


def compact_keep_boolean_schema(response_format):
    """Recognize only the exact strict one-boolean DocETL output contract."""
    if not isinstance(response_format, dict) or response_format.get("type") != "json_schema":
        return False
    schema = (response_format.get("json_schema") or {}).get("schema")
    return (isinstance(schema, dict)
            and set(schema) == {"type", "properties", "required", "additionalProperties"}
            and schema.get("type") == "object"
            and schema.get("properties") == {"keep": {"type": "boolean"}}
            and schema.get("required") == ["keep"]
            and schema.get("additionalProperties") is False)


def valid_keep_boolean_response(response):
    try:
        content = response["choices"][0]["message"]["content"]
        value = json.loads(content)
    except (ValueError, TypeError, KeyError, IndexError):
        return False
    return isinstance(value, dict) and set(value) == {"keep"} and type(value["keep"]) is bool


def kv_cache_observation(usage, cache):
    """Normalize per-request model KV/prompt-cache telemetry.

    SGLang/OpenAI-compatible servers normally expose prefix-cache usage as
    ``usage.prompt_tokens_details.cached_tokens``.  The local response cache is
    stronger: a hit reuses the complete response, so all prompt tokens are
    considered cached when the original prompt-token count is available.
    Missing provider telemetry is kept as ``None`` rather than being mistaken
    for a measured zero.
    """
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("prompt_tokens_details")
    details = details if isinstance(details, dict) else {}
    prompt = usage.get("prompt_tokens")
    prompt = int(prompt) if isinstance(prompt, (int, float)) and prompt >= 0 else None
    complete = bool((cache or {}).get("hit", False))
    if complete:
        return {
            "prompt_tokens_observed": prompt,
            "kv_cache_hit_tokens": prompt,
            "kv_cache_observation": "prompt_cache_complete_hit" if prompt is not None else "prompt_cache_complete_hit_prompt_tokens_unavailable",
            "prompt_cache_complete_hit": True,
        }
    cached = details.get("cached_tokens")
    if not isinstance(cached, (int, float)) or cached < 0:
        cached = None
    else:
        cached = int(cached)
    return {
        "prompt_tokens_observed": prompt,
        "kv_cache_hit_tokens": cached,
        "kv_cache_observation": "model_usage.prompt_tokens_details.cached_tokens" if cached is not None else "model_usage_cached_tokens_unavailable",
        "prompt_cache_complete_hit": False,
    }


def responses_payload(serial, requested, content, usage):
    """Build an OpenAI Responses object for clients that do not speak chat/completions.

    The message must stay at ``output[1]``. The real API puts reasoning items first,
    and Morrila (PLOP) reads ``output[1].content[0].text`` literally; with only one
    output item every PLOP semantic filter parses to false and every projection to an
    empty string.
    """
    response_usage = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "output_tokens_details": {"reasoning_tokens": 0},
    }
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and isinstance(details.get("cached_tokens"), int):
        response_usage["input_tokens_details"] = {"cached_tokens": details["cached_tokens"]}
    return {
        "id": "resp_" + str(serial),
        "object": "response",
        "created_at": int(time.time()),
        "model": requested,
        "status": "completed",
        "error": None,
        "output": [
            {
                "id": "rs_" + str(serial),
                "type": "reasoning",
                "summary": [],
            },
            {
                "id": "msg_" + str(serial),
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                    }
                ],
            },
        ],
        "output_text": content,
        "usage": response_usage,
    }


def flush_model_caches(models, *, idle_timeout=0):
    """Flush every configured SGLang service before a 19-query run starts."""
    endpoints = []
    for model in models:
        endpoint = model["endpoint"].rstrip("/")
        endpoint = endpoint.removesuffix("/v1")
        endpoints.append(
            endpoint
            + "/flush_cache"
            + (f"?timeout={idle_timeout}" if idle_timeout else "")
        )
    results = []
    for endpoint in sorted(set(endpoints)):
        body = None
        error = None
        # A remote model tunnel can reconnect between the readiness probe and
        # this reset request. Retry the reset itself; this does not retry model
        # inference and does not impose a request-rate or concurrency limit.
        for attempt in range(4):
            try:
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
                    urllib.request.Request(endpoint, method="POST"),
                    timeout=max(30, idle_timeout + 10),
                ) as response:
                    body = response.read().decode(errors="replace")
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status}")
                error = None
                break
            except (OSError, urllib.error.URLError, RuntimeError) as exc:
                error = exc
                if attempt < 3:
                    time.sleep(1.0 + attempt)
        if error is not None:
            raise RuntimeError(
                f"KV cache reset failed for {endpoint} after 4 attempts: {error}"
            ) from error
        results.append({"endpoint": endpoint, "status": "flushed", "response": body})
    return results


class PoolGateway:
    def __init__(
        self,
        models,
        *,
        port=8899,
        request_timeout=180,
        upstream_retries=3,
        upstream_concurrency_limit=32,
    ):
        self.models = {model["alias"]: model for model in models}
        self.routes = {}
        self.default_token = None
        self.lock = threading.Lock()
        self.serial = 0
        self.active = {}
        self.accounting = {}
        self.request_timeout = request_timeout
        self.upstream_concurrency_limit = (
            int(upstream_concurrency_limit)
            if upstream_concurrency_limit is not None
            else None
        )
        self.upstream_semaphore = (
            threading.BoundedSemaphore(self.upstream_concurrency_limit)
            if self.upstream_concurrency_limit
            else None
        )
        # Retry only transient upstream connection failures. This does not
        # admit, batch, pace, or cap requests; it prevents a remote inference
        # server reset from becoming a false semantic result.
        self.upstream_retries = max(0, int(upstream_retries))
        self.inflight = {}
        self.peak_inflight = {}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def reply(self, status, value):
                payload = json.dumps(value, ensure_ascii=False).encode()
                return self.write_response(status, "application/json", payload)

            def write_response(self, status, content_type, payload):
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return True
                except OSError:
                    # Headers can fail too. A disconnected native client is a
                    # response-delivery failure, not another model inference.
                    return False

            def do_GET(self):
                if self.path.endswith("/models"):
                    self.reply(
                        200,
                        {
                            "object": "list",
                            "data": [
                                {
                                    "id": alias,
                                    "object": "model",
                                    "actual_id": model["id"],
                                }
                                for alias, model in owner.models.items()
                            ],
                        },
                    )
                else:
                    self.reply(404, {"error": "Unknown gateway path"})

            def do_POST(self):
                parts = self.path.split("/")
                token = (
                    parts[2]
                    if len(parts) > 3 and parts[1] == "run"
                    else owner.default_token
                )
                with owner.lock:
                    route = owner.routes.get(token)
                    admitted = route is not None and not route["closed"]
                    if admitted:
                        folder = route["folder"]
                        owner.serial += 1
                        serial = owner.serial
                        owner.active[serial] = token
                if not admitted:
                    self.reply(410, {"error": "No active experiment route"})
                    return
                started = time.monotonic()
                record = {
                    "request_id": serial,
                    "route": token,
                    "path": self.path,
                    "submitted_unix": time.time(),
                }
                if route.get("record_request_stage"):
                    stage = self.headers.get("X-Experiment-Request-Stage")
                    # Missing or unrecognised headers stay conservative: they
                    # cannot be treated as harmless planning-stage failures.
                    record["request_stage"] = stage if stage in {"planning", "execution"} else "unknown"
                trace = {"attempts": 0, "calls": 0, "usage": {}}
                try:
                    data = json.loads(
                        self.rfile.read(int(self.headers.get("Content-Length", 0)))
                    )
                    requested = data.get("model", "")
                    alias = normalize_alias(requested)
                    alias = route["aliases"].get(alias, alias)
                    if alias not in route["allowed"] or alias not in owner.models:
                        raise ValueError(f"Unknown or forbidden model {requested!r}")
                    model = owner.models[alias]
                    record.update(
                        requested_model=requested,
                        routed_alias=alias,
                        actual_model=model["id"],
                        endpoint=model["endpoint"],
                    )
                    response_mode = self.path.endswith("/responses")
                    if not (response_mode or self.path.endswith("/chat/completions")):
                        raise ValueError(
                            "Only chat/completions and responses are supported"
                        )
                    messages = data.get("messages")
                    if messages is None:
                        value = data.get("input", "")
                        messages = (
                            [{"role": "user", "content": value}]
                            if isinstance(value, str)
                            else value
                        )
                        if data.get("instructions"):
                            messages = [
                                {"role": "system", "content": data["instructions"]},
                                *messages,
                            ]
                    messages = [
                        {
                            **message,
                            "content": [
                                {**part, "type": "text"}
                                if part.get("type") == "input_text"
                                else part
                                for part in message["content"]
                            ],
                        }
                        if isinstance(message.get("content"), list)
                        else message
                        for message in messages
                    ]
                    if response_mode and route.get("prompt_protocol"):
                        messages, record["prompt_protocol"] = make_plop_messages(
                            data, protocol=route["prompt_protocol"]
                        )
                    forwarded = {
                        "model": model["id"],
                        "messages": messages,
                        "temperature": data.get("temperature", 0),
                        "max_tokens": data.get("max_tokens")
                        or data.get("max_completion_tokens")
                        or data.get("max_output_tokens")
                        or 1024,
                        "chat_template_kwargs": {"enable_thinking": False},
                        "stream": False,
                    }
                    for key in (
                        "response_format",
                        "tools",
                        "tool_choice",
                        "stop",
                        "top_p",
                        "frequency_penalty",
                        "presence_penalty",
                        "seed",
                        "parallel_tool_calls",
                        "logprobs",
                        "top_logprobs",
                        "logit_bias",
                    ):
                        if key in data:
                            forwarded[key] = data[key]
                    if response_mode and "text" in data and "format" in data["text"]:
                        format_ = data["text"]["format"]
                        if format_.get("type") == "json_schema":
                            forwarded["response_format"] = {
                                "type": "json_schema",
                                "json_schema": {
                                    key: value
                                    for key, value in format_.items()
                                    if key != "type"
                                },
                            }
                    compact_boolean = (alias in route["compact_boolean_aliases"]
                                       and not forwarded.get("tools")
                                       and compact_keep_boolean_schema(forwarded.get("response_format")))
                    if compact_boolean:
                        # Keep the native schema for the first response. Cap a
                        # boolean JSON to avoid spending 1024 tokens on an
                        # unbounded whitespace loop; retry malformed output
                        # once with an equivalent no-whitespace grammar below.
                        forwarded["max_tokens"] = min(int(forwarded["max_tokens"]), 64)
                        record["structured_output_constraint"] = "keep_boolean_schema_with_compact_fallback_v1"
                    forwarded.update(model.get("request_overrides", {}))
                    payload = json.dumps(forwarded, ensure_ascii=False).encode()
                    owner.record(
                        folder,
                        {
                            **record,
                            "event": "submit",
                            "requested": data,
                            "forwarded": forwarded,
                            "input_sha256": hashlib.sha256(payload).hexdigest(),
                        },
                    )
                    with owner.lock:
                        owner.active[serial] = token
                    from experiments.system_comparison.transport.upstream import (
                        send_completion,
                    )

                    def normalized(response):
                        if route.get("response_transform"):
                            return route["response_transform"](data, response)[0]
                        return response

                    def infer():
                        result = send_completion(
                            owner,
                            model,
                            forwarded,
                            folder,
                            route,
                            record,
                            trace,
                            normalized,
                        )
                        if compact_boolean and not valid_keep_boolean_response(result):
                            retry = {**forwarded, "regex": r'\{"keep":(true|false)\}', "max_tokens": 32}
                            retry.pop("response_format", None)
                            owner.record(folder, {
                                **record, "event": "format_retry",
                                "reason": "malformed or incomplete boolean JSON; retry with equivalent compact grammar",
                                "forwarded": retry,
                            })
                            result = send_completion(owner, model, retry, folder, route, record, trace, normalized)
                            if not valid_keep_boolean_response(result):
                                raise ValueError("Boolean JSON remained invalid after compact-format retry")
                        return result

                    response_cache = route.get("response_cache")
                    cache_info = {}
                    if response_cache is not None:
                        cache_key = response_cache.key(model["endpoint"], forwarded)
                        result, cache_info = response_cache.resolve(
                            cache_key, infer, normalized
                        )
                    else:
                        result = infer()
                    provider_result = result
                    normalization = None
                    if route.get("response_transform"):
                        try:
                            result, normalization = route["response_transform"](
                                data, result
                            )
                        except ValueError:
                            owner.record(
                                folder,
                                {
                                    **record,
                                    "event": "rejected_response",
                                    "response": provider_result,
                                    "usage": provider_result.get("usage", {}),
                                },
                            )
                            raise
                    result = dict(result)
                    provider_usage = provider_result.get("usage", {}) if isinstance(provider_result, dict) else {}
                    # A response-cache hit never enters ``send_completion``,
                    # so its trace is empty. Retain the original usage stored
                    # with the cached response for per-request telemetry.
                    effective_usage = (
                        dict(trace["usage"])
                        if trace["usage"].get("prompt_tokens") is not None
                        else dict(provider_usage or {})
                    )
                    result["usage"] = effective_usage
                    message = result["choices"][0]["message"]
                    content = message.get("content") or ""
                    kv_observation = kv_cache_observation(effective_usage, cache_info)
                    owner.record(
                        folder,
                        {
                            **record,
                            "event": "completed",
                            "seconds": time.monotonic() - started,
                            "upstream_attempts": trace["attempts"],
                            "transport_retries": max(
                                0, trace["attempts"] - trace["calls"]
                            ),
                            "cache": cache_info,
                            "response": provider_result,
                            "usage": effective_usage,
                            **kv_observation,
                        },
                    )
                    if normalization:
                        owner.record(
                            folder,
                            {**record, "event": "normalization", **normalization},
                        )
                    if response_mode:
                        result = responses_payload(
                            serial, requested, content, result.get("usage", {})
                        )
                    else:
                        result["model"] = requested
                    if data.get("stream") and not response_mode:
                        event = {
                            "id": result.get("id", f"chatcmpl_{serial}"),
                            "object": "chat.completion.chunk",
                            "model": requested,
                            "created": int(time.time()),
                            "choices": [
                                {"index": 0, "delta": message, "finish_reason": "stop"}
                            ],
                            "usage": result.get("usage", {}),
                        }
                        delivered = self.write_response(
                            200,
                            "text/event-stream",
                            (
                                "data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n"
                            ).encode(),
                        )
                    else:
                        delivered = self.reply(200, result)
                    if not delivered:
                        owner.record(
                            folder,
                            {
                                **record,
                                "event": "delivery_error",
                                "phase": "downstream_response",
                                "seconds": time.monotonic() - started,
                                "error": "Native client disconnected before response delivery completed",
                            },
                        )
                except (
                    urllib.error.URLError,
                    ValueError,
                    KeyError,
                    TypeError,
                    OSError,
                ) as error:
                    detail = str(error)
                    if isinstance(error, urllib.error.HTTPError):
                        detail += ": " + error.read().decode(errors="replace")
                    owner.record(
                        folder,
                        {
                            **record,
                            "event": "error",
                            "seconds": time.monotonic() - started,
                            "upstream_attempts": trace["attempts"],
                            "transport_retries": max(
                                0, trace["attempts"] - trace["calls"]
                            ),
                            "usage": trace["usage"],
                            "error": detail,
                        },
                    )
                    self.reply(
                        error.code
                        if isinstance(error, urllib.error.HTTPError)
                        else 400
                        if isinstance(error, ValueError)
                        else 502,
                        {"error": {"message": detail, "type": "formal_gateway_error"}},
                    )
                finally:
                    with owner.lock:
                        owner.active.pop(serial, None)

        class GatewayServer(ThreadingHTTPServer):
            # Listen backlog accommodates native request bursts; it is not a
            # throttle on simultaneous model requests.
            request_queue_size = 256

        self.server = GatewayServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def record(self, folder, value):
        line = json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"
        with self.lock:
            counts = self.accounting.setdefault(
                str(folder),
                {
                    "model_requests": 0,
                    "completed_model_requests": 0,
                    "model_errors": 0,
                    "response_delivery_errors": 0,
                    "planning_response_delivery_errors": 0,
                    "execution_response_delivery_errors": 0,
                    "unknown_response_delivery_errors": 0,
                    "requests_by_model": {},
                    "request_seconds_sum": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "upstream_retries": 0,
                    "upstream_http_attempts": 0,
                    "kv_cache_observed_requests": 0,
                    "kv_cache_unobserved_requests": 0,
                    "kv_cache_hit_tokens": 0,
                    "kv_cache_prompt_tokens": 0,
                    "prompt_cache_complete_hits": 0,
                },
            )
            if value["event"] == "submit":
                counts["model_requests"] += 1
                model = value.get("actual_model", "unrouted")
                counts["requests_by_model"][model] = (
                    counts["requests_by_model"].get(model, 0) + 1
                )
            elif value["event"] == "completed":
                counts["completed_model_requests"] += 1
                counts["request_seconds_sum"] += value.get("seconds", 0)
                counts["prompt_tokens"] += (
                    value.get("usage", {}).get("prompt_tokens", 0) or 0
                )
                counts["completion_tokens"] += (
                    value.get("usage", {}).get("completion_tokens", 0) or 0
                )
                counts["upstream_retries"] += value.get(
                    "transport_retries",
                    max(0, int(value.get("upstream_attempts", 1)) - 1),
                )
                counts["upstream_http_attempts"] += int(
                    value.get("upstream_attempts", 1)
                )
                observation = value.get("kv_cache_observation")
                if observation in {
                    "model_usage.prompt_tokens_details.cached_tokens",
                    "prompt_cache_complete_hit",
                }:
                    counts["kv_cache_observed_requests"] += 1
                else:
                    counts["kv_cache_unobserved_requests"] += 1
                if value.get("prompt_cache_complete_hit"):
                    counts["prompt_cache_complete_hits"] += 1
                if isinstance(value.get("kv_cache_hit_tokens"), (int, float)):
                    counts["kv_cache_hit_tokens"] += int(value["kv_cache_hit_tokens"])
                if isinstance(value.get("prompt_tokens_observed"), (int, float)):
                    counts["kv_cache_prompt_tokens"] += int(value["prompt_tokens_observed"])
                cache = value.get("cache", {})
                for name, amount in (
                    ("cache_hits", int(cache.get("hit", False))),
                    ("cache_verifications", cache.get("verification_calls", 0)),
                    ("cache_vote_calls", cache.get("vote_calls", 0)),
                ):
                    counts[name] = counts.get(name, 0) + amount
            elif value["event"] == "error":
                counts["model_errors"] += 1
                counts["request_seconds_sum"] += value.get("seconds", 0)
                counts["upstream_retries"] += value.get(
                    "transport_retries",
                    max(0, int(value.get("upstream_attempts", 1)) - 1),
                )
                counts["prompt_tokens"] += (
                    value.get("usage", {}).get("prompt_tokens", 0) or 0
                )
                counts["completion_tokens"] += (
                    value.get("usage", {}).get("completion_tokens", 0) or 0
                )
                if value.get("actual_model"):
                    counts["upstream_http_attempts"] += int(
                        value.get("upstream_attempts", 1)
                    )
            elif value["event"] in {"retry", "rejected_response"}:
                # Truncated completions have known usage, even when retried.
                counts["prompt_tokens"] += (
                    value.get("usage", {}).get("prompt_tokens", 0) or 0
                )
                counts["completion_tokens"] += (
                    value.get("usage", {}).get("completion_tokens", 0) or 0
                )
            elif value["event"] == "delivery_error":
                counts["response_delivery_errors"] += 1
                stage = value.get("request_stage")
                if stage not in {"planning", "execution"}:
                    stage = "unknown"
                counts[stage + "_response_delivery_errors"] += 1
            with (folder / "gateway.jsonl").open("a") as stream:
                stream.write(line)

    def summary(self, folder):
        with self.lock:
            value = self.accounting.get(
                str(folder),
                {
                    "model_requests": 0,
                    "completed_model_requests": 0,
                    "model_errors": 0,
                    "response_delivery_errors": 0,
                    "planning_response_delivery_errors": 0,
                    "execution_response_delivery_errors": 0,
                    "unknown_response_delivery_errors": 0,
                    "requests_by_model": {},
                    "request_seconds_sum": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "upstream_retries": 0,
                    "upstream_http_attempts": 0,
                    "kv_cache_observed_requests": 0,
                    "kv_cache_unobserved_requests": 0,
                    "kv_cache_hit_tokens": 0,
                    "kv_cache_prompt_tokens": 0,
                    "prompt_cache_complete_hits": 0,
                },
            )
            return {
                **value,
                "requests_by_model": dict(value["requests_by_model"]),
                "cache_hits": value.get("cache_hits", 0),
                "cache_verifications": value.get("cache_verifications", 0),
                "cache_vote_calls": value.get("cache_vote_calls", 0),
                "kv_cache_observed_requests": value.get("kv_cache_observed_requests", 0),
                "kv_cache_unobserved_requests": value.get("kv_cache_unobserved_requests", 0),
                "kv_cache_hit_tokens": value.get("kv_cache_hit_tokens", 0),
                "kv_cache_prompt_tokens": value.get("kv_cache_prompt_tokens", 0),
                "prompt_cache_complete_hits": value.get("prompt_cache_complete_hits", 0),
                "gateway_concurrency_limit": self.upstream_concurrency_limit,
                "route_concurrency_limits": sorted(
                    {
                        route["upstream_concurrency_limit"]
                        for route in self.routes.values()
                        if str(route["folder"]) == str(folder)
                        and route.get("upstream_concurrency_limit") is not None
                    }
                ),
                "peak_upstream_requests": self.peak_inflight.get(str(folder), 0),
            }

    def register(
        self,
        token,
        folder,
        *,
        allowed=None,
        aliases=None,
        default=False,
        prompt_protocol=None,
        upstream_concurrency_limit=None,
        response_transform=None,
        response_cache=None,
        retry_unavailable_format=False,
        compact_boolean_aliases=None,
        record_request_stage=False,
    ):
        if prompt_protocol is not None and prompt_protocol not in PLOP_PROMPT_PROTOCOLS:
            raise ValueError(f"Unknown prompt protocol: {prompt_protocol}")
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with self.lock:
            if token in self.routes:
                raise ValueError("Attempt route token already exists")
            self.routes[token] = {
                "folder": folder,
                "allowed": set(allowed or self.models),
                "aliases": aliases or {},
                "closed": False,
                "default": default,
                "prompt_protocol": prompt_protocol,
                "response_transform": response_transform,
                "response_cache": response_cache,
                "retry_unavailable_format": retry_unavailable_format,
                "compact_boolean_aliases": set(compact_boolean_aliases or ()),
                "record_request_stage": record_request_stage,
                "upstream_concurrency_limit": upstream_concurrency_limit,
                "upstream_semaphore": (
                    threading.BoundedSemaphore(upstream_concurrency_limit)
                    if upstream_concurrency_limit is not None
                    else None
                ),
            }
            if default:
                self.default_token = token
        return self.url + "/run/" + token + "/v1"

    def close_run(self, token):
        with self.lock:
            self.routes[token]["closed"] = True
            if self.default_token == token:
                self.default_token = None

    def evidence(self, token):
        """Read complete records after the route is closed and all requests end."""
        from experiments.system_comparison.reporting.evidence import (
            sealed_gateway_events,
        )

        return sealed_gateway_events(self, token)

    def reopen_for_reference(self, token, folder, teacher):
        """Only an idle retained SQL session may reuse its bound URL."""
        with self.lock:
            route = self.routes[token]
            if not route["closed"] or token in self.active.values():
                raise ValueError("Cannot replace an active timed-query route")
            route.update(
                folder=Path(folder),
                allowed={teacher},
                aliases=dict.fromkeys({*self.models, *route["aliases"]}, teacher),
                prompt_protocol=None,
                closed=False,
            )
            if route.get("default"):
                self.default_token = token

    def pending(self, token=None):
        with self.lock:
            return (
                sum(value == token for value in self.active.values())
                if token
                else len(self.active)
            )

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
