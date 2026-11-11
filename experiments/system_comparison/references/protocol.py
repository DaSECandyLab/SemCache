"""Lossless serialization for native Teacher response protocols.

WorkBuddy returns ordinary chat text even when the native caller asks for a
JSON string. Supply missing string quotes or serialize standalone boolean
tokens; never reinterpret the task, repair facts, or call another model.
"""

import json
import re
from copy import deepcopy


def complete_response(request, response):
    reason = response["choices"][0].get("finish_reason")
    if reason != "stop":
        raise ValueError(f"Incomplete Teacher response: finish_reason={reason}")
    return response, None


def lotus_boolean_response(request, response):
    """Serialize standalone YES/NO replies only for LOTUS's boolean filter."""
    complete_response(request, response)
    messages = request.get("messages", [])
    filter_instruction = (
        "The user will provide a claim and some relevant context. "
        "Your job is to determine whether the claim is true for the given context. "
        "Use the following format to provide your answer: "
        "Answer: <Your answer here. The answer should be either True or False>"
    )
    if not messages or messages[0].get("role") != "system":
        return response, None
    system_content = messages[0].get("content")
    user_content = messages[-1].get("content")
    if (
        not isinstance(system_content, str)
        or " ".join(system_content.split()) != filter_instruction
        or messages[-1].get("role") != "user"
        or not isinstance(user_content, str)
        or not user_content.startswith("Context:\n")
        or "\nClaim: " not in user_content
    ):
        return response, None
    content = response["choices"][0]["message"].get("content")
    if not isinstance(content, str):
        return response, None
    match = re.fullmatch(r"\s*(?:Answer\s*:\s*)?(YES|NO)\s*", content, re.IGNORECASE)
    if match is None:
        return response, None
    serialized = "Answer: " + ("True" if match.group(1).casefold() == "yes" else "False")
    adjusted = deepcopy(response)
    adjusted["choices"][0]["message"]["content"] = serialized
    return adjusted, {
        "protocol": "lotus-native-filter-boolean-v1",
        "original_content": content,
        "serialized_content": serialized,
        "prompt_changed": False,
    }


def sema_string_response(request, response):
    complete_response(request, response)
    schema = request.get("response_format", {}).get("json_schema", {}).get("schema", {})
    choice = response["choices"][0]
    content = choice["message"].get("content")
    if schema.get("type") == "boolean" and isinstance(content, str):
        serialized = {
            "true": "true",
            "false": "false",
            "yes": "true",
            "no": "false",
        }.get(content.strip().casefold())
        if serialized is not None and serialized != content:
            adjusted = deepcopy(response)
            adjusted["choices"][0]["message"]["content"] = serialized
            return adjusted, {
                "protocol": "sema-native-json-boolean-v1",
                "original_content": content,
                "serialized_content": serialized,
                "prompt_changed": False,
            }
    if schema.get("type") != "string" or not isinstance(content, str):
        return response, None
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, str):
        return response, None
    adjusted = deepcopy(response)
    adjusted["choices"][0]["message"]["content"] = json.dumps(
        content, ensure_ascii=False
    )
    return adjusted, {
        "protocol": "sema-native-json-string-v1",
        "original_content": content,
        "serialized_content": adjusted["choices"][0]["message"]["content"],
        "prompt_changed": False,
    }
