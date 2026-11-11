"""Small compatibility fixes for local Abacus model responses.

The upstream parser trims a one-to-one answer to the *last* closing brace.
Small local models sometimes repeat the JSON object after a markdown code block,
so ``json.loads`` sees two adjacent objects and raises ``Extra data``.  The
first object is the answer requested by Abacus; trailing explanation or a
duplicate rendering is formatting noise.  This adapter keeps the upstream
parser unchanged and only applies a balanced JSON decode as a fallback.
"""

from __future__ import annotations

import json
import math
import re
from contextlib import contextmanager
from datetime import date, datetime
from decimal import Decimal
from types import UnionType
from typing import Any, Union, get_args, get_origin
from unittest.mock import patch


def install_short_output_prompts():
    """Keep Abacus output schemas/strategies while requesting concise answers."""
    from palimpzest.constants import PromptStrategy
    from palimpzest.prompts import PromptFactory

    if getattr(PromptFactory, "_comparison_short_outputs", False):
        return
    system = dict(PromptFactory.BASE_SYSTEM_PROMPT_MAP)
    user = dict(PromptFactory.BASE_USER_PROMPT_MAP)
    for name in ("MAP", "FILTER", "JOIN", "AGG"):
        strategy = getattr(PromptStrategy, name)
        direct = getattr(PromptStrategy, name + "_NO_REASONING")
        system[strategy] = system[direct]
        user[strategy] = "Return only the requested answer; do not include an explanation.\n" + user[direct]
    for name in ("MAP_MOA_PROPOSER", "FILTER_MOA_PROPOSER"):
        strategy = getattr(PromptStrategy, name)
        for prompts in (system, user):
            text = prompts[strategy]
            text = text.replace("a detailed and succinct analysis describing what you believe is the correct value for each output field", "the requested values concisely")
            text = text.replace("a detailed and succinct analysis describing whether you believe the input satisfies the filter condition", "a concise TRUE or FALSE answer")
            text = text.replace("Be sure to cite information from the context as evidence of why your answers are correct. Do not hallucinate evidence.", "Use only the supplied context; do not invent facts.")
            text = text.replace("Be sure to cite information from the context as evidence of why your determination is correct. Do not hallucinate evidence.", "Use only the supplied context; do not invent facts.")
            text = text.replace("Let's think step-by-step in order to answer the question.", "Give the requested answer directly without an explanation.")
            prompts[strategy] = text
    for name in ("MAP_MOA_AGG", "FILTER_MOA_AGG"):
        strategy = getattr(PromptStrategy, name)
        text = user[strategy]
        text = text.replace("Let's think step-by-step in order to answer the question.\n\nREASONING: ",
                            "Return only the requested answer, without an explanation.\n\nANSWER: ")
        user[strategy] = text
    PromptFactory.BASE_SYSTEM_PROMPT_MAP = system
    PromptFactory.BASE_USER_PROMPT_MAP = user
    PromptFactory._comparison_short_outputs = True


@contextmanager
def native_join_sampling():
    """Keep ordinary joins stateless in sentinel trials and exactly scored.

    Upstream's relational join rule omits the ``is_validation`` handling used
    by its semantic join rules. Retained records then leak between concurrent
    trials, producing extra outputs which the pairwise validator never scores.
    Only deterministic equality joins receive quality 1; semantic operators
    continue through the unmodified native Validator.
    """
    from palimpzest.query.execution.execution_strategy import SentinelExecutionStrategy
    from palimpzest.query.execution.mab_execution_strategy import OpFrontier
    from palimpzest.query.operators.join import RelationalJoin
    from palimpzest.query.optimizer.rules import RelationalJoinRule

    original_score = SentinelExecutionStrategy._score_quality
    original_inputs = OpFrontier.get_frontier_op_inputs

    def frontier_inputs(self, source_indices_to_sample, max_quality_op):
        # Exclude samples pruned to None before native Join indexing.
        def available(indices):
            if isinstance(indices, str):
                return indices.split("---")[0] in self.root_dataset_ids
            return indices is not None and any(available(item) for item in indices)

        usable = {indices for indices in source_indices_to_sample if available(indices)}
        return original_inputs(self, usable, max_quality_op)

    def substitute(cls, logical_expression, **runtime_kwargs):
        return cls._perform_substitution(
            logical_expression,
            RelationalJoin,
            runtime_kwargs,
            {
                "retain_inputs": not runtime_kwargs["is_validation"],
                "join_parallelism": runtime_kwargs["join_parallelism"],
            },
        )

    def score(self, validator, groups):
        semantic = {}
        for indices, pairs in groups.items():
            pending = []
            for record_set, op in pairs:
                if isinstance(op, RelationalJoin):
                    for stats in record_set.record_op_stats:
                        stats.quality = 1.0
                else:
                    pending.append((record_set, op))
            semantic[indices] = pending
        _, generation_stats = original_score(self, validator, semantic)
        return groups, generation_stats

    with (
        patch.object(RelationalJoinRule, "substitute", classmethod(substitute)),
        patch.object(SentinelExecutionStrategy, "_score_quality", score),
        patch.object(OpFrontier, "get_frontier_op_inputs", frontier_inputs),
    ):
        yield


def install_tolerant_json_parser():
    """Install local-only Abacus parsing and hashing compatibility fixes.

    The paper implementation assumes that every value in a ``DataRecord`` is
    JSON serializable.  Our frozen TPC-H inputs intentionally preserve SQL
    ``DECIMAL`` and ``DATE`` values, so the upstream record hash can otherwise
    fail before the native plan starts.  The conversion workload has a second
    harmless format issue: local models sometimes return a numeric answer with
    an explanation instead of the requested one-field JSON object.  The
    adapters below repair representation only; they do not change the native
    optimizer, operator graph, or model policy.
    """
    from palimpzest.query.generators import generators

    if not getattr(generators, "_semcache_tolerant_json", False):
        original = generators.get_json_from_answer

        def tolerant(answer, model, cardinality):
            try:
                return original(answer, model, cardinality)
            except json.JSONDecodeError:
                repaired = _repair_json_escapes(answer)
                if repaired != answer:
                    try:
                        return original(repaired, model, cardinality)
                    except json.JSONDecodeError:
                        pass
                # Abacus's convert prompts use an object for one-to-one outputs
                # and an array for one-to-many outputs.  Decode exactly the
                # first complete JSON value, preserving strings and escaped
                # braces.  Numeric, field-aware recovery is installed below
                # because this function does not receive the output schema.
                opener = (
                    "{" if getattr(cardinality, "name", "") == "ONE_TO_ONE" else "["
                )
                start = answer.find(opener)
                if start < 0:
                    raise
                value, _ = json.JSONDecoder().raw_decode(answer[start:])
                if not isinstance(value, (dict, list)):
                    raise
                return value

        generators.get_json_from_answer = tolerant
        generators._semcache_tolerant_json = True

    # DataRecord.__hash__ is used for native memoization and sentinel deduping.
    # Keep SQL values intact in the actual operators, but use a tagged stable
    # representation only for that hash serialization.
    from palimpzest.core.elements.records import DataRecord

    if not getattr(DataRecord, "_semcache_json_hash_serializer", False):

        def json_default(value):
            if isinstance(value, Decimal):
                return {"$decimal": str(value)}
            if isinstance(value, (datetime, date)):
                return {"$date": value.isoformat()}
            # DuckDB/Pandas occasionally leave NumPy scalar values in an
            # object column.  Import lazily so normal runs do not require NumPy
            # in this adapter's module namespace.
            try:
                import numpy as np

                if isinstance(value, np.generic):
                    return value.item()
            except ImportError:
                pass
            raise TypeError(
                f"Object of type {type(value).__name__} is not JSON serializable"
            )

        def hash_json(
            self,
            include_bytes=True,
            bytes_to_str=False,
            project_cols=None,
            sorted=False,
        ):
            record = self.to_dict(include_bytes, bytes_to_str, project_cols, sorted)
            return json.dumps(record, indent=2, default=json_default)

        DataRecord.to_json_str = hash_json
        DataRecord._semcache_json_hash_serializer = True

    # Generator._check_convert_answer_text has the output fields available,
    # unlike get_json_from_answer.  Use that schema to recover a numeric scalar
    # from the common local-model form ``Price: 32000000`` or
    # ``Answer: 250000000`` after the native parser rejects the surrounding
    # explanation.  It is intentionally restricted to numeric fields.
    if not getattr(generators.Generator, "_semcache_numeric_recovery", False):
        original_check = generators.Generator._check_convert_answer_text

        def tolerant_check(self, answer_text, fields, throw_exception=False):
            original_error = None
            try:
                result = original_check(
                    self, answer_text, fields, throw_exception=throw_exception
                )
                if result is not None:
                    return result
            except Exception as error:
                original_error = error
            recovered = _recover_numeric_fields(answer_text, fields)
            if (
                recovered is None
                and getattr(self.cardinality, "name", "") == "ONE_TO_ONE"
            ):
                recovered = _recover_text_fields(answer_text, fields)
            if recovered is not None:
                return self._prepare_field_answers(recovered, fields)
            if original_error is not None and throw_exception:
                if isinstance(original_error, json.JSONDecodeError):
                    # Native format retries only catch OutputTypeError. A
                    # malformed/truncated JSON response needs the same format
                    # correction instead of silently yielding None fields.
                    raise generators.OutputTypeError(
                        "Incomplete or invalid JSON output; return the exact requested "
                        f"fields {list(fields)} in one complete JSON object"
                    ) from original_error
                raise original_error
            return None

        generators.Generator._check_convert_answer_text = tolerant_check
        generators.Generator._semcache_numeric_recovery = True
        generators.Generator._semcache_text_recovery = True

    if not getattr(generators.Generator, "_semcache_boolean_format_retry", False):
        original_bool = generators.Generator._check_bool_answer_text

        def retryable_bool(self, answer_text, throw_exception=False):
            try:
                return original_bool(self, answer_text, throw_exception=throw_exception)
            except Exception as error:
                lower = answer_text.casefold()
                if re.search(
                    r"(?:does not|did not|not)\s+(?:satisfy|meet)|not\s+satisfied",
                    lower,
                ):
                    return {"passed_operator": False}
                if re.search(
                    r"(?:satisfies|satisfied|meets)\s+(?:the\s+)?(?:filter\s+)?condition",
                    lower,
                ):
                    return {"passed_operator": True}
                if throw_exception:
                    raise generators.OutputTypeError(
                        "Missing explicit boolean answer. Preserve your previous decision "
                        "and output ANSWER: TRUE or ANSWER: FALSE."
                    ) from error
                return None

        generators.Generator._check_bool_answer_text = retryable_bool
        generators.Generator._semcache_boolean_format_retry = True


_NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?"
_UNIT = r"(?:\s*(?:k|thousand|m|million|b|billion))?"
_LABEL_NUMBER = re.compile(
    rf"(?im)(?:^|[\n])\s*(?:[-*•]\s*)?(?:\*\*)?"
    rf"(?P<label>[A-Za-z_][\w .-]*?)(?:\*\*)?\s*:\s*"
    rf"(?P<number>{_NUMBER}){_UNIT}(?:\s|$|[.,])"
)


def _repair_json_escapes(answer: str) -> str:
    """Remove backslashes that are not legal JSON string escapes.

    Local completions occasionally emit ``\\washing`` when they intended a
    separator before a word.  Removing only the invalid slash preserves the
    text and leaves standard escapes (``\\n``, ``\\"``, ``\\uXXXX``) intact.
    """
    repaired = re.sub(r"\\(?![\"\\/bfnrtu])", "", answer)
    repaired = re.sub(r"\\u(?![0-9a-fA-F]{4})", "u", repaired)
    return repaired


def _number_value(number: str, unit: str | None = None):
    raw = number.replace(",", "")
    value = float(raw) if any(c in raw for c in ".eE") else int(raw)
    if unit:
        multipliers = {
            "k": 1_000,
            "thousand": 1_000,
            "m": 1_000_000,
            "million": 1_000_000,
            "b": 1_000_000_000,
            "billion": 1_000_000_000,
        }
        value *= multipliers[unit.casefold()]
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return value


def _recover_numeric_fields(answer: str, fields):
    """Return a field->scalar object for a numeric one-to-one conversion."""
    if not isinstance(fields, dict) or not fields:
        return None
    names = list(fields)
    numeric_names = []
    for name, info in fields.items():
        annotation = getattr(info, "annotation", None)
        text = str(annotation).casefold()
        if any(
            token in text for token in ("int", "float", "decimal", "double", "number")
        ):
            numeric_names.append(name)
    if not numeric_names:
        return None

    # Prefer an exact output-field label, preserving field spelling/case.
    found = {}
    for match in _LABEL_NUMBER.finditer(answer):
        label = match.group("label").strip().strip("*").strip()
        key = next(
            (name for name in names if name.casefold() == label.casefold()), None
        )
        if key is None:
            # Natural-language labels such as "Price" are useful when the
            # schema has a single numeric output field.
            if len(numeric_names) == 1 and label.casefold() in {
                "answer",
                "value",
                "result",
                "price",
                "amount",
            }:
                key = numeric_names[0]
        if key is not None and key in numeric_names:
            suffix = match.group(0)[match.end("number") :]
            unit_match = re.search(r"(?i)\b(k|thousand|m|million|b|billion)\b", suffix)
            found[key] = _number_value(
                match.group("number"), unit_match.group(1) if unit_match else None
            )
    if found:
        return found

    # Inline ``Answer: 123`` is common in short completions without a line
    # break.  Restrict this fallback to one numeric output field.
    if len(numeric_names) == 1:
        match = re.search(
            rf"(?i)\b(?:answer|value|result|price|amount)\s*[:=-]\s*(?P<number>{_NUMBER}){_UNIT}",
            answer,
        )
        if match:
            suffix = match.group(0)[match.end("number") :]
            unit_match = re.search(r"(?i)\b(k|thousand|m|million|b|billion)\b", suffix)
            return {
                numeric_names[0]: _number_value(
                    match.group("number"), unit_match.group(1) if unit_match else None
                )
            }

        # MAP_MOA_AGG concatenates several model responses.  Local models
        # commonly put the final scalar in a markdown emphasis span without a
        # field label (``**500000000**``).  Prefer the last such scalar; the
        # preceding reasoning contains the source values and unit examples.
        bold = list(re.finditer(rf"\*\*\s*(?P<number>{_NUMBER}){_UNIT}\s*\*\*", answer))
        if bold:
            match = bold[-1]
            suffix = match.group(0)[match.end("number") :]
            unit_match = re.search(r"(?i)\b(k|thousand|m|million|b|billion)\b", suffix)
            return {
                numeric_names[0]: _number_value(
                    match.group("number"), unit_match.group(1) if unit_match else None
                )
            }

        # A plain scalar on its own line is the next safest one-field form.
        standalone = list(
            re.finditer(rf"(?m)^\s*(?P<number>{_NUMBER}){_UNIT}\s*$", answer)
        )
        if standalone:
            match = standalone[-1]
            suffix = match.group(0)[match.end("number") :]
            unit_match = re.search(r"(?i)\b(k|thousand|m|million|b|billion)\b", suffix)
            return {
                numeric_names[0]: _number_value(
                    match.group("number"), unit_match.group(1) if unit_match else None
                )
            }
    return None


def _recover_text_fields(answer: str, fields):
    """Wrap an explicit final field value; never infer a value from prose."""
    if not isinstance(fields, dict) or len(fields) != 1:
        return None
    name, info = next(iter(fields.items()))
    annotation = getattr(info, "annotation", None)
    parts = (
        get_args(annotation)
        if get_origin(annotation) in (Union, UnionType)
        else (annotation,)
    )
    if str not in parts or any(part not in (str, Any, type(None)) for part in parts):
        return None
    # A complete one-field object may contain a quoted movie title followed
    # by unquoted prose: {"plot": "88 Minutes" focuses on a professor.}.
    # Preserve that entire literal field value, including the title quotes.
    # An unfinished object or an additional field is never repaired this way.
    object_match = re.fullmatch(
        rf'\s*\{{\s*"{re.escape(name)}"\s*:\s*(.*?)\s*\}}\s*', answer, re.DOTALL
    )
    if object_match:
        value = object_match.group(1)
        try:
            prefix, end = json.JSONDecoder().raw_decode(value)
        except json.JSONDecodeError:
            return None
        tail = value[end:].strip()
        if (
            isinstance(prefix, str)
            and re.match(r"[A-Za-z]", tail)
            and not tail.endswith('"')
            and not re.search(r'[,{}\[\]]\s*"|"\s*:', tail)
        ):
            return {name: value}
        return None
    # Exact field name at the beginning of a final paragraph, e.g.
    # BACKGROUND: ... . Do not select category words from an explanation or
    # turn an unlabelled/truncated completion into a supposedly valid answer.
    match = re.search(rf"(?im)^\s*{re.escape(name)}\s*:\s*", answer)
    if match is None:
        return None
    value = answer[match.end() :].strip()
    value = re.sub(r"\n---\s*$", "", value).strip()
    if not value or value.startswith(("{", "[", "```")):
        return None
    if value == "null":
        return {name: None}
    if value.startswith('"'):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
        if not isinstance(value, str):
            return None
    return {name: value}
