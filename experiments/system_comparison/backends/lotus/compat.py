"""Process-local compatibility for Boolean text and token probabilities."""

import importlib
import math
import re
import sys
from types import SimpleNamespace


def normalize_numeric_columns(frame, schema):
    """Decode LOTUS's scalar text outputs using the declared numeric contract.

    A map's permitted JSON null arrives as the Python string 'null'. Keep
    text fields untouched and reject other invalid numeric answers.
    """
    import pandas as pd

    from experiments.system_comparison.reporting.results import value_for_type

    result = frame.copy()
    columns = {}
    for name, kind in schema:
        if name not in result.columns or not (
            kind.upper() in {"DOUBLE", "FLOAT", "REAL"} or kind.upper().startswith("DECIMAL")
        ):
            continue
        values, null_tokens = [], 0
        for value in result[name]:
            if isinstance(value, str) and value.strip().casefold() == "null":
                value = None
                null_tokens += 1
            elif not isinstance(value, str) and pd.isna(value):
                value = None
            if isinstance(value, bool):
                raise ValueError(f"Boolean in LOTUS numeric output column {name}")
            values.append(value_for_type(value, kind))
        result[name] = pd.Series(values, index=result.index, dtype=object)
        columns[name] = {"type": kind, "rows": len(values), "decoded_null_tokens": null_tokens}
    return result, {"rule": "declared numeric columns only; JSON null token becomes SQL NULL", "columns": columns}


def install_boolean_protocol():
    from lotus.models import LM

    if getattr(LM, "_comparison_boolean_protocol", False):
        return
    original_scores = LM.format_logprobs_for_filter_cascade
    postprocessors = importlib.import_module("lotus.sem_ops.postprocessors")
    original_postprocess = postprocessors.filter_postprocess

    def scores(self, logprobs, output_tokens=("True", "False")):
        def token(value):
            stripped = value.strip().casefold()
            if stripped in {output_tokens[0].casefold(), "yes"}:
                return output_tokens[0]
            if stripped in {output_tokens[1].casefold(), "no"}:
                return output_tokens[1]
            return value

        def normalize_item(item):
            groups = {}
            for top in item.top_logprobs:
                groups.setdefault(token(top.token), []).append(top.logprob)
            tops = []
            for value, probabilities in groups.items():
                maximum = max(probabilities)
                combined = maximum + math.log(sum(math.exp(p - maximum) for p in probabilities))
                tops.append(SimpleNamespace(token=value, logprob=combined))
            return SimpleNamespace(token=token(item.token), logprob=item.logprob, top_logprobs=tops)

        normalized = [[normalize_item(item) for item in response] for response in logprobs]
        return original_scores(self, normalized, output_tokens=output_tokens)

    def postprocess(answers, *args, **kwargs):
        normalized = [
            re.sub(r"(?im)(^|\bAnswer:\s*)(YES|NO)\s*[.!]?\s*$",
                   lambda match: match[1] + ("True" if match[2].casefold() == "yes" else "False"),
                   answer) if isinstance(answer, str) else answer
            for answer in answers
        ]
        return original_postprocess(normalized, *args, **kwargs)

    LM.format_logprobs_for_filter_cascade = scores
    for name, module in list(sys.modules.items()):
        if name.startswith("lotus.sem_ops") and getattr(module, "filter_postprocess", None) is original_postprocess:
            module.filter_postprocess = postprocess
    postprocessors.filter_postprocess = postprocess
    LM._comparison_boolean_protocol = True
