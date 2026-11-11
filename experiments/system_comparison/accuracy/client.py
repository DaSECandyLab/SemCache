"""Thread-safe HTTP client and request journal for accuracy judgments."""

import json
import os
import threading
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path


class JudgeClient:
    """Send already-batched judgments without starting a local server."""

    def __init__(self, teacher, journal, settings):
        self.teacher, self.settings = teacher, settings
        self.journal = Path(journal)
        self.lock = threading.Lock()
        self.fatal_error = None
        self.requests = self.active = self.peak_active = 0
        self.batch_sizes = Counter()
        self.usage = Counter()

    def error_message(self, error):
        message = f"{type(error).__name__}: {error}"
        key = self.teacher.api_key or os.environ.get(
            self.teacher.endpoint.get("api_key_env", "")
        )
        if key:
            message = message.replace(key, "[redacted]")
        return message[:2000]

    def http(self, messages, *, max_tokens, size):
        endpoint = self.teacher.endpoint
        body = {
            "model": endpoint["model"],
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if "enable_thinking" in endpoint:
            body["chat_template_kwargs"] = {
                "enable_thinking": endpoint["enable_thinking"]
            }
        if endpoint.get("thinking"):
            body["thinking"] = {"type": endpoint["thinking"]}
        if "reasoning_effort" in endpoint:
            body["reasoning_effort"] = endpoint["reasoning_effort"]
        headers = {"Content-Type": "application/json", **endpoint.get("headers", {})}
        key = self.teacher.api_key
        if key is None and endpoint.get("api_key_env"):
            key = os.environ.get(endpoint["api_key_env"])
            if not key:
                raise ValueError("Teacher API credential is missing")
        if key:
            headers["Authorization"] = "Bearer " + key
        base = endpoint["base_url"].rstrip("/")
        url = base + (
            "/chat/completions" if base.endswith("/v1") else "/v1/chat/completions"
        )
        with self.lock:
            # Billing/auth failures affect the entire teacher endpoint. Already
            # queued judgments must not each issue another upstream call.
            if self.fatal_error:
                raise RuntimeError(self.fatal_error)
            self.requests += 1
            self.batch_sizes[size] += 1
            serial = self.requests
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(
                urllib.request.Request(url, json.dumps(body).encode(), headers),
                timeout=self.settings["request_timeout_seconds"],
            ) as reply:
                result = json.load(reply)
            choice = result["choices"][0]
            content = choice["message"].get("content")
            with self.lock, self.journal.open("a") as stream:
                for usage_name in (
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                ):
                    self.usage[usage_name] += (result.get("usage") or {}).get(
                        usage_name, 0
                    ) or 0
                stream.write(
                    json.dumps(
                        {
                            "request": serial,
                            "teacher": self.teacher.model_id,
                            "batch_size": size,
                            "input": messages,
                            "output": content,
                            "finish_reason": choice.get("finish_reason"),
                            "usage": result.get("usage", {}),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if choice.get("finish_reason") != "stop" or not isinstance(content, str):
                raise ValueError("Teacher completion is incomplete")
            return content
        except Exception as error:
            detail = self.error_message(error)
            if isinstance(error, urllib.error.HTTPError):
                try:
                    detail += ": " + error.read(4096).decode(errors="replace")
                except (OSError, ValueError):
                    # Preserve the HTTP status even if the provider closes its
                    # error body early; the caller uses it to choose retries.
                    pass
                finally:
                    error.close()
                if key:
                    detail = detail.replace(key, "[redacted]")
            with self.lock, self.journal.open("a") as stream:
                if isinstance(error, urllib.error.HTTPError) and error.code in {
                    401,
                    402,
                    403,
                }:
                    self.fatal_error = detail
                stream.write(
                    json.dumps(
                        {
                            "request": serial,
                            "teacher": self.teacher.model_id,
                            "batch_size": size,
                            "error": detail,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if isinstance(error, urllib.error.HTTPError):
                raise RuntimeError(detail) from error
            raise
        finally:
            with self.lock:
                self.active -= 1

    def metrics(self):
        with self.lock:
            return {
                "http_requests": self.requests,
                "peak_http_concurrency": self.peak_active,
                "http_batch_sizes": dict(self.batch_sizes),
                "usage": dict(self.usage),
            }
