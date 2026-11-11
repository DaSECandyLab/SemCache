"""Opt-in transport adaptation of PLOP's native STRING output instruction.

The exact trailing STRING suffix is the discriminator. No task text, output
type, or business rule is inferred from the original request.
"""

import hashlib
from copy import deepcopy

PLOP_PROMPT_PROTOCOLS = ("plop-task-output-v1", "plop-task-output-v2")
PLOP_TASK_OUTPUT_PROTOCOL = "plop-task-output-v2"
STRING_SUFFIX = "\n Return a single word or phrase, do not contain any other words."
V1_TASK_OUTPUT_INSTRUCTION = (
    "Follow the original task in the user message to determine the requested "
    "value, format, and output length, including complete sentences, summaries, "
    "or lists of multiple values when requested. Output only the requested "
    "value, without additional explanation or commentary. Preserve the original "
    "task's missing-value tokens and empty-string requirements. Follow any "
    "additional system instructions about the response format."
)
TASK_OUTPUT_INSTRUCTION = V1_TASK_OUTPUT_INSTRUCTION + (
    " If the original task explicitly defines a missing-value token and the "
    "requested information is absent, output that exact token alone. Do not "
    "replace it with an explanatory sentence or invent the missing information. "
    "If the original task defines no missing-value convention, follow its "
    "instructions without introducing one."
)
_PROTOCOL_INSTRUCTIONS = {
    "plop-task-output-v1": V1_TASK_OUTPUT_INSTRUCTION,
    "plop-task-output-v2": TASK_OUTPUT_INSTRUCTION,
}


def _responses_messages(data):
    """Match the gateway's existing Responses-to-chat conversion, on a copy."""
    messages = deepcopy(data.get("messages"))
    if messages is None:
        value = deepcopy(data.get("input", ""))
        messages = (
            [{"role": "user", "content": value}] if isinstance(value, str) else value
        )
        if data.get("instructions"):
            messages = [
                {"role": "system", "content": deepcopy(data["instructions"])},
                *messages,
            ]
    if not isinstance(messages, list):
        return messages
    return [
        {
            **message,
            "content": [
                {**part, "type": "text"} if part.get("type") == "input_text" else part
                for part in message["content"]
            ],
        }
        if isinstance(message.get("content"), list)
        else message
        for message in messages
    ]


def make_plop_messages(data, protocol=None):
    """Return independent messages and protocol evidence without changing data.

    ``None`` preserves the gateway's current conversion. The named protocol
    removes exactly one native STRING suffix from a string ``input``; BOOL,
    numeric, structured-input, and explicit-message requests pass through.
    Existing Responses ``instructions`` remain system messages after the
    neutral protocol instruction so their specific formatting rules prevail.
    """
    if protocol is not None and protocol not in PLOP_PROMPT_PROTOCOLS:
        raise ValueError(f"Unknown PLOP prompt protocol {protocol!r}")

    messages = _responses_messages(data)
    evidence = {"protocol": protocol, "status": "unchanged", "reason": "disabled"}
    if protocol is None:
        return messages, evidence
    instruction = _PROTOCOL_INSTRUCTIONS[protocol]
    evidence["policy_sha256"] = hashlib.sha256(instruction.encode("utf-8")).hexdigest()
    if data.get("messages") is not None:
        evidence["reason"] = "explicit_messages"
        return messages, evidence
    value = data.get("input", "")
    if not isinstance(value, str):
        evidence["reason"] = "non_string_input"
        return messages, evidence
    if not value.endswith(STRING_SUFFIX):
        evidence["reason"] = "no_native_string_suffix"
        return messages, evidence

    # Slice only the known final appendage. Never strip whitespace, replace
    # interior occurrences, or rewrite the original task and record text.
    messages[-1]["content"] = value[: -len(STRING_SUFFIX)]
    messages.insert(0, {"role": "system", "content": instruction})
    evidence.update(status="adapted", reason="native_string_suffix_removed")
    return messages, evidence


def adapt_plop_messages(data):
    """Apply the explicitly selected current task-output protocol (v2)."""
    return make_plop_messages(data, protocol=PLOP_TASK_OUTPUT_PROTOCOL)
