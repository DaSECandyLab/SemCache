"""Strict four-model routing with uncached, complete request evidence."""

import io
import json
import os
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import urllib3

from experiments.system_comparison.transport.plop_protocol import (
    PLOP_PROMPT_PROTOCOLS,
)

from .gateway_handlers import (
    compact_keep_boolean_schema as compact_keep_boolean_schema,
)
from .gateway_handlers import (
    flush_model_caches as flush_model_caches,
)
from .gateway_handlers import (
    kv_cache_observation as kv_cache_observation,
)
from .gateway_handlers import (
    make_gateway_handler,
)
from .gateway_handlers import (
    normalize_alias as normalize_alias,
)
from .gateway_handlers import (
    responses_payload as responses_payload,
)
from .gateway_handlers import (
    valid_keep_boolean_response as valid_keep_boolean_response,
)


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
        self._http_pool = urllib3.PoolManager(
            num_pools=16,
            maxsize=max(32, self.upstream_concurrency_limit or 32),
            retries=False,
        )
        self._log_streams = {}
        owner = self
        Handler = make_gateway_handler(owner)

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
            folder_key = str(folder)
            stream = self._log_streams.get(folder_key)
            if stream is None:
                stream = (Path(folder) / "gateway.jsonl").open("a", encoding="utf-8")
                self._log_streams[folder_key] = stream
            stream.write(line)
            stream.flush()

    def http_post_json(self, model, url, payload, headers, timeout):
        has_proxy = any(
            os.environ.get(key)
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
        )
        if not model.get("disable_proxy") and has_proxy:
            request = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        try:
            response = self._http_pool.request(
                "POST",
                url,
                body=payload,
                headers=headers,
                timeout=float(timeout),
                preload_content=True,
            )
        except urllib3.exceptions.HTTPError as exc:
            cause = getattr(exc, "reason", None)
            while cause is not None and not isinstance(cause, OSError):
                cause = getattr(cause, "reason", None) or getattr(
                    cause, "__cause__", None
                )
            if isinstance(cause, OSError):
                raise urllib.error.URLError(cause) from exc
            if isinstance(exc, urllib3.exceptions.TimeoutError):
                raise TimeoutError(str(exc)) from exc
            raise urllib.error.URLError(exc) from exc
        if 200 <= response.status < 300:
            return json.loads(response.data)
        raise urllib.error.HTTPError(
            url,
            response.status,
            response.reason or "",
            response.headers,
            io.BytesIO(response.data),
        )

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
            folder_key = str(self.routes[token]["folder"])
            if self.default_token == token:
                self.default_token = None
            if not any(
                not route["closed"] and str(route["folder"]) == folder_key
                for route in self.routes.values()
            ):
                stream = self._log_streams.pop(folder_key, None)
                if stream is not None:
                    stream.flush()
                    stream.close()

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
        with self.lock:
            for stream in self._log_streams.values():
                stream.flush()
                stream.close()
            self._log_streams.clear()
        self._http_pool.clear()
