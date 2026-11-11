"""Reference-only model selection and safe transport configuration."""

import copy
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

KINDS = ("SEMANTIC_FILTER", "SEMANTIC_PROJECTION", "SEMANTIC_JOIN")


@dataclass(frozen=True)
class TeacherModel:
    model_id: str
    endpoint: dict
    supported_kinds: tuple = KINDS
    api_key: str | None = field(default=None, repr=False, compare=False)

    def configuration(self):
        return {
            "model_id": self.model_id,
            "endpoint": copy.deepcopy(self.endpoint),
            "supported_kinds": list(self.supported_kinds),
        }

    def transport_models(self, models):
        # 原生 HTTP 客户端通过环境变量在发送边界读取密钥；配置快照只保留变量名。
        # 只桥接调用者显式提供的 JSON 密钥，不读取任何外部凭据文件。
        if self.api_key is not None:
            os.environ[self.endpoint["api_key_env"]] = self.api_key
        return {**models, self.model_id: copy.deepcopy(self.endpoint)}

    def apply_profile(self, profile):
        profile["judge_model"] = self.model_id
        profile["teacher_models"] = {
            self.model_id: {"supported_kinds": list(self.supported_kinds)}
        }


def teacher_supports(profile, op, model):
    return model in op.get("models", []) or op["kind"] in profile.get(
        "teacher_models", {}
    ).get(model, {}).get("supported_kinds", [])


def _read_config(config=None):
    if config is None:
        raise ValueError("Supply --teacher-config for an external teacher model")
    source = config
    document = (
        json.loads(Path(source).read_text())
        if isinstance(source, (str, Path))
        else source
    )
    if not isinstance(document, dict):
        raise ValueError("Teacher configuration must be a JSON object")
    # 单个 teacher 直接使用 API 模型名作为内部标识，无需重复维护别名和默认项。
    if "model" in document:
        model = document["model"]
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Teacher model must be a nonempty string")
        return model, {model: document}
    entries = document.get("models", document)
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Teacher configuration requires a nonempty model registry")
    if any(not isinstance(endpoint, dict) for endpoint in entries.values()):
        raise ValueError("Teacher model entries must be JSON objects")
    return document.get("default_model") if "models" in document else None, entries


def default_teacher_model(config=None):
    default, entries = _read_config(config)
    if default is None and len(entries) == 1:
        default = next(iter(entries))
    if not isinstance(default, str) or default not in entries:
        raise ValueError("Set default_model in teacher JSON or specify --teacher-model")
    return default


def load_teacher(model_id, candidate_models, config=None):
    """Resolve a candidate teacher or an external endpoint without adding actions."""
    if model_id is None:

        def size(name):
            entry = candidate_models[name]
            match = re.search(
                r"(\d+(?:\.\d+)?)B(?:\b|[-_])", entry.get("model", ""), re.I
            )
            value = entry.get("parameters") or (
                float(match[1]) * 1e9 if match else None
            )
            if value is None:
                raise ValueError("Teacher selection requires model parameter counts")
            return value

        model_id = max(candidate_models, key=size)
    if config is None and model_id in candidate_models:
        endpoint, kinds = copy.deepcopy(candidate_models[model_id]), KINDS
    else:
        _, entries = _read_config(config)
        if model_id not in entries:
            raise ValueError(
                f"Unknown teacher model: {model_id}; supply --teacher-config"
            )
        endpoint = copy.deepcopy(entries[model_id])
        kinds = tuple(endpoint.pop("supported_kinds", KINDS))
    if any(key.lower() == "authorization" for key in endpoint):
        raise ValueError("Teacher credentials must use api_key or api_key_env")
    headers = endpoint.get("headers")
    if headers is not None:
        if not isinstance(headers, dict) or not headers:
            raise ValueError("headers must map header names to string values")
        for name, value in headers.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(value, str)
                or not value
                or any(ord(c) <= 32 or ord(c) >= 127 for c in value)
            ):
                raise ValueError("headers must map header names to printable values")
            if name.lower() == "authorization":
                raise ValueError("Authorization must use api_key or api_key_env")
    # 配置使用 max_output_tokens；原生 Chat Completions transport 使用 max_tokens。
    if "max_output_tokens" in endpoint:
        limit = endpoint.pop("max_output_tokens")
        if "max_tokens" in endpoint:
            raise ValueError("Specify only max_output_tokens, not both token limits")
        if type(limit) is not int or not 1 <= limit <= 262144:
            raise ValueError("max_output_tokens must be an integer in [1, 262144]")
        endpoint["max_tokens"] = limit
    api_key = endpoint.pop("api_key", None)
    if api_key is not None:
        if "api_key_env" in endpoint:
            raise ValueError("Specify only one of api_key and api_key_env")
        if (
            not isinstance(api_key, str)
            or not api_key
            or any(ord(c) <= 32 or ord(c) >= 127 for c in api_key)
        ):
            raise ValueError("api_key must be a nonempty HTTP header value")
        identity = model_id + "\0" + endpoint.get("base_url", "")
        suffix = hashlib.sha256(identity.encode()).hexdigest()[:24].upper()
        endpoint["api_key_env"] = "SEMCACHE_TEACHER_KEY_" + suffix
    if not kinds or set(kinds) - set(KINDS):
        raise ValueError("Unsupported teacher semantic kinds")
    name = endpoint.get("api_key_env")
    if name is not None and (
        not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
    ):
        raise ValueError("api_key_env must be an environment variable name")
    return TeacherModel(model_id, endpoint, kinds, api_key)
