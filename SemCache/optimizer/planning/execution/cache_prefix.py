"""Static token lengths and field-reuse interactions, without inference calls."""

import json
import math
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def tokenizer():
    from tokenizers import Tokenizer

    path = (
        Path(__file__).resolve().parents[4]
        / "sglang/models/gemma-4-E2B-it-qat-w4a16-ct/tokenizer.json"
    )
    return Tokenizer.from_file(str(path))


@lru_cache(maxsize=4096)
def token_count(text):
    return len(tokenizer().encode(text, add_special_tokens=False).ids)


def prefix_context(node, op, action, profile, semantic):
    from planning.observations.statistics import prefix

    if not action["model_id"].startswith("gemma4-"):
        return {"prompt_token_context": [0.0] * 16}
    system = profile.get("system_prefix")
    spec = op.get("operator_specification") or profile.get(
        "operator_specifications", {}
    ).get(op["kind"])
    if not isinstance(system, str) or not isinstance(spec, str):
        raise ValueError("Cache prefix predictor requires frozen prompt specifications")
    namespace = json.loads(prefix(op, action, system, spec))
    task = op["task"]
    blocks = {
        "OPERATOR_SPECIFICATION": spec,
        "TASK_DESCRIPTION": task,
        "DATA_BLOCK": "",
    }
    full = "\n\n".join(
        f"<{block}>\n{blocks[block]}\n</{block}>"
        for block in action["layout"]["block_order"]
    )
    static = token_count(system) + token_count(full)
    before = token_count(system) + token_count(namespace[2])
    tail = max(0, static - before)
    field = semantic["semantic_field_context"]
    first_length = math.expm1(field[0])
    field_count = math.expm1(field[-1])
    total_length = math.expm1(field[24]) * field_count
    own = math.expm1(node["numeric"][-3])
    history = node.get("l_existing")
    existing = math.expm1(history[1]) if history and history[13] == 1 else 0.0
    previous = max(
        (math.expm1(v) for v in node.get("l_previous", {}).values() if v is not None),
        default=0.0,
    )
    ratio = own / max(1, first_length)
    self_full = min(1, ratio)
    cross_full = min(1, max(existing, previous) / max(1, first_length))
    values = [
        *[
            math.log1p(v)
            for v in (
                token_count(system),
                token_count(spec),
                token_count(task),
                static,
                before,
                tail,
            )
        ],
        math.log1p(first_length),
        math.log1p(total_length),
        self_full,
        cross_full,
        math.log1p(self_full * tail),
        math.log1p(cross_full * tail),
        1 - min(1, field[2]) if field[8] else 0.0,
        min(8, ratio),
        math.log1p(before + own / 4 + max(self_full, cross_full) * tail),
        1.0,
    ]
    return {"prompt_token_context": values}
