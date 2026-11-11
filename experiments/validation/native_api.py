"""Native semantic operator backends for the explicit SQL conversion adapter."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
from pathlib import Path

from experiments.core.paths import ROOT


def load_adapter(system):
    path = ROOT / f"systems/{system}/validate.py"
    spec = importlib.util.spec_from_file_location(f"validation_{system}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def instruction(task, names):
    return re.sub(r"\{(\d+)\}", lambda m: names[int(m[1])], task)


def condition_instruction(task):
    return (
        "Evaluate whether the following condition holds for the provided inputs. "
        "Judge the condition; do not assume it is true merely because it is stated.\n"
        + task
    )


def boolean_instruction(task):
    """Translate selected SQL boolean answer tokens to Abacus's native format."""
    return re.sub(r"\b(YES|NO)\b", lambda m: {"YES": "TRUE", "NO": "FALSE"}[m[0]], task)


def parse_native_tags(result, tags, *, predicate):
    """Accept explicit equivalent scalar output; never infer a missing answer."""
    if predicate:
        value = result.get("output")
        if value is None:
            value = result.get("raw_output")
        if isinstance(value, bool):
            return {**result, "output": str(value)}
        if not isinstance(value, str) or value.strip().lower() not in {
            "true",
            "false",
            "yes",
            "no",
        }:
            raise ValueError("Nirvana predicate did not return an explicit boolean")
        return {
            **result,
            "output": "True" if value.strip().lower() in {"true", "yes"} else "False",
        }
    if any(result.get(tag) is None for tag in tags):
        if (
            len(tags) != 1
            or not isinstance(result.get("raw_output"), str)
            or not result["raw_output"].strip()
            or re.search(r"[<>\[\]{}]", result["raw_output"])
        ):
            raise ValueError("Nirvana map did not return its required fields")
        return {**result, tags[0]: result["raw_output"].strip()}
    return result


class NativeAPI:
    def __init__(self, system, job):
        self.system, self.job = system, job
        self.folder = Path(job["output"])
        self.serial = 0
        self.adapter = load_adapter(system)
        self.stats = []
        if system == "nirvana":
            self.adapter.configure_client(job)

    def __call__(self, kind, task, rows, fields):
        self.serial += 1
        folder = self.folder / f"stage-{self.serial:02d}"
        folder.mkdir()
        names = [f"arg{i}" for i in range(len(rows[0]))]
        records = [
            {**dict(zip(names, row)), "adapter_id": i} for i, row in enumerate(rows)
        ]
        (folder / "contract.json").write_text(
            json.dumps(
                {"kind": kind, "task": task, "fields": fields, "records": records},
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        if self.system == "abacus":
            answers = self.abacus(kind, task, records, names, fields, folder)
        elif self.system == "moar":
            answers = self.moar(kind, task, records, names, fields, folder)
        elif self.system == "nirvana":
            answers = self.nirvana(kind, task, records, names, fields, folder)
        else:
            raise ValueError(self.system)
        self.stats.append(
            {
                "stage": self.serial,
                "kind": kind,
                "inputs": len(rows),
                "outputs": len(answers) if kind == "projection" else sum(answers),
            }
        )
        return answers

    def abacus(self, kind, task, records, names, fields, folder):

        errors = self.adapter.NativeErrors()
        logging.getLogger().addHandler(errors)
        try:
            import pandas as pd

            ds = self.adapter.source_dataset(
                pd.DataFrame(records), f"{self.job['id']}:stage:{self.serial}"
            )
            text = instruction(task, names)
            if kind == "projection":
                types = {
                    "VARCHAR": str,
                    "TEXT": str,
                    "DOUBLE": float,
                    "BIGINT": int,
                    "INTEGER": int,
                    "BOOLEAN": bool,
                }
                cols = [
                    {"name": f["name"], "type": types[f["type"]] | None, "desc": text}
                    for f in fields
                ]
                ds = ds.sem_map(cols, desc=text, depends_on=names)
            else:
                text = boolean_instruction(text)
                ds = ds.sem_filter(text, depends_on=names)
            (folder / "native-task.txt").write_text(text + "\n")
            result = ds.run(config=self.adapter.native_config(self.job))
            if errors.messages:
                raise ValueError("; ".join(errors.messages[:3]))
            rows = [
                {
                    "adapter_id": record["adapter_id"],
                    **{field["name"]: record[field["name"]] for field in fields},
                }
                for record in result
            ]
            stats = (
                result.execution_stats.model_dump(mode="json")
                if result.execution_stats
                else {}
            )
            (folder / "native-stats.json").write_text(
                json.dumps(stats, default=str, ensure_ascii=False) + "\n"
            )
        finally:
            logging.getLogger().removeHandler(errors)
        indexed = self.indexed(rows, len(records), kind)
        return (
            [
                {f["name"]: indexed[i][f["name"]] for f in fields}
                for i in range(len(records))
            ]
            if kind == "projection"
            else [i in indexed for i in range(len(records))]
        )

    def moar(self, kind, task, records, names, fields, folder):
        import yaml
        from docetl.runner import DSLRunner

        os.environ["OPENAI_API_BASE"] = self.job["endpoint"]
        os.environ["OPENAI_API_KEY"] = "local-validation"
        os.environ["DOCETL_HOME_DIR"] = str(folder / "cache")
        source = folder / "input.json"
        source.write_text(json.dumps(records, ensure_ascii=False) + "\n")
        schema = (
            {f["name"]: "string" for f in fields}
            if kind == "projection"
            else {"_short_explanation": "string", "keep": "boolean"}
        )
        prompt = instruction(task, ["{{ input." + name + " }}" for name in names])
        if kind != "projection":
            # DocETL filters inherit map's system prompt. Explicitly evaluate the
            # supplied condition instead of treating its wording as an assertion.
            # _short_explanation is a supported native Filter output field.
            prompt = (
                condition_instruction(prompt)
                + "\nGive one concise sentence explaining the truth value in "
                "_short_explanation; encode that truth value as the boolean keep. "
                "Preserve all comparisons and conditions exactly."
            )
        op = {
            "name": "semantic_stage",
            "type": "map" if kind == "projection" else "filter",
            "model": self.job.get("native_model_alias", "gpt-4o-mini"),
            "prompt": prompt,
            "output": {"mode": "structured_output", "schema": schema},
            "bypass_cache": True,
            "optimize": False,
            "timeout": int(self.job.get("timeout", 120)),
            "max_retries_per_timeout": 0,
            "litellm_completion_kwargs": {
                "api_key": "local-validation",
                "temperature": 0,
                "num_retries": 0,
                "max_tokens": 1024,
            },
        }
        config = {
            "default_model": op["model"],
            "default_lm_api_base": self.job["endpoint"],
            "bypass_cache": True,
            "plan_rewrites": False,
            "datasets": {
                "input": {"type": "file", "path": str(source), "source": "local"}
            },
            "operations": [op],
            "pipeline": {
                "steps": [
                    {
                        "name": "stage",
                        "input": "input",
                        "operations": ["semantic_stage"],
                    }
                ],
                "output": {"type": "file", "path": str(folder / "output.json")},
            },
        }
        (folder / "pipeline.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
        )
        runner = DSLRunner(config, max_threads=2)
        with self.adapter.LLMAudit(folder / "llm-validation.jsonl") as audit:
            result, _ = runner.run()
        if audit.summary()["validated_calls"] != audit.summary()["calls"]:
            raise ValueError("Native DocETL returned unvalidated semantic outputs")
        indexed = self.indexed(result, len(records), kind)
        return (
            [
                {f["name"]: indexed[i][f["name"]] for f in fields}
                for i in range(len(records))
            ]
            if kind == "projection"
            else [i in indexed for i in range(len(records))]
        )

    @staticmethod
    def indexed(rows, count, kind):
        result = {}
        for row in rows:
            identifier = row.get("adapter_id")
            if (
                isinstance(identifier, bool)
                or not isinstance(identifier, int)
                or not 0 <= identifier < count
                or identifier in result
            ):
                raise ValueError(
                    "Native operator corrupted candidate identity/bag multiplicity"
                )
            result[identifier] = row
        if kind == "projection" and len(result) != count:
            raise ValueError("Native projection dropped candidates")
        return result

    def nirvana(self, kind, task, records, names, fields, folder):
        import asyncio

        import pandas as pd
        from nirvana.ops.base import BaseOperation
        from nirvana.ops.filter import FilterOperation
        from nirvana.ops.map import MapOperation

        original = BaseOperation.llm
        normalized = []

        class CompatibleScalarOutput:
            default_model = original.default_model

            async def __call__(self, messages, **kwargs):
                answer = await original(messages, **kwargs)
                parsed = parse_native_tags(
                    answer, kwargs.get("tags", []), predicate=kind != "projection"
                )
                normalized.append(
                    {
                        "raw_output": answer.get("raw_output"),
                        "parsed": {
                            k: v
                            for k, v in parsed.items()
                            if k not in {"cost", "raw_output"}
                        },
                    }
                )
                return parsed

        BaseOperation.set_llm(CompatibleScalarOutput())
        try:
            data = pd.DataFrame(records, dtype=object)
            text = task
            if kind != "projection":
                text = boolean_instruction(text)
            if kind != "projection":
                text += "\nReturn only <output>True</output> or <output>False</output>, without explanation."
            (folder / "native-task-template.txt").write_text(text + "\n")

            class BoundSQLTask:
                async def _execute_by_plain_llm(
                    self, data, user_instruction, dtypes, **kwargs
                ):
                    # Bind positional SQL prompt arguments to actual input values,
                    # then retain the complete native operator/client execution.
                    row = records[data.name]
                    data = pd.Series({name: row[name] for name in names}, dtype=object)
                    bound = instruction(
                        user_instruction, [str(row[name]) for name in names]
                    )
                    return await super()._execute_by_plain_llm(
                        data, bound, dtypes, **kwargs
                    )

            class SQLFilter(BoundSQLTask, FilterOperation):
                pass

            class SQLMap(BoundSQLTask, MapOperation):
                def _postprocess_map_output(self, result, output_columns):
                    # Upstream collapses an explicit empty XML value into NULL.
                    # A SQL VARCHAR distinguishes the two; preserve the value.
                    return {
                        name: None
                        if result[name] is None
                        or result[name].strip().lower() == "none"
                        else result[name]
                        for name in output_columns
                    }

            args = {
                "user_instruction": text,
                "input_columns": names,
                "rate_limit": 4,
            }
            if kind == "projection":
                operation = SQLMap(**args, output_columns=[f["name"] for f in fields])
                result = asyncio.run(operation.execute(data))
                answers = [
                    {f["name"]: result.outputs[f["name"]][i] for f in fields}
                    for i in range(len(records))
                ]
            else:
                operation = SQLFilter(**args)
                result = asyncio.run(operation.execute(data))
                answers = [bool(value) for value in result.output]
        finally:
            BaseOperation.set_llm(original)
            (folder / "output-format.json").write_text(
                json.dumps(normalized, ensure_ascii=False, indent=2) + "\n"
            )
        return answers
