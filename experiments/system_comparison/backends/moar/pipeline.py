"""Lower frozen queries to a complete, shared DocETL pipeline."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

from experiments.core.io import save
from experiments.system_comparison.backends.moar.runtime import records
from experiments.validation.native_api import instruction
from experiments.validation.runtime.native_graph import Compiler, describe
from experiments.validation.workload.cte import inline_single_use


AMENITIES_COMPACT_INSTRUCTION = (
    " List all distinct amenities as short generic labels separated by commas."
    " Omit quantities, brands, model numbers and detailed specifications."
    " Preserve every named amenity, including a swimming pool or gym if present;"
    " do not invent amenities."
)


def semantic(kind, task, columns, name, outputs=None):
    prompt = instruction(task, ['{{ input[' + json.dumps(column) + '] }}' for column in columns])
    if not any("{{ input[" + json.dumps(column) + "] }}" in prompt for column in columns):
        prompt += "\nInputs:\n" + "\n".join(column + ': {{ input[' + json.dumps(column) + '] }}' for column in columns)
    return {
        "name": name, "type": kind, "prompt": prompt,
        "output": {"mode": "structured_output", "schema": outputs or {"keep": "boolean"}},
        "bypass_cache": True, "timeout": 180, "max_retries_per_timeout": 0,
        "litellm_completion_kwargs": {"temperature": 0, "num_retries": 0, "max_tokens": 1024},
    }


class Builder:
    def __init__(self, job, tables):
        self.job, self.tables = job, tables
        self.folder = Path(job["output"])
        self.datasets, self.steps, self.ops, self.memo = {}, [], [], {}
        self.serial = 0
        self.full_paths, self.sample_paths = {}, {}
        self.sample_ids = None

    def name(self, prefix):
        self.serial += 1
        return f"{prefix}_{self.serial}"

    def source(self, frame, fields, label, sample=False):
        name = self.name("source")
        path = self.folder / (name + "-full.json")
        rows = records(frame, fields)
        save(path, rows)
        self.datasets[name] = {"type": "file", "source": "local", "path": str(path)}
        self.full_paths[name] = str(path)
        if sample:
            # Sample selection does not inspect teacher membership or results.
            subset = frame.sample(n=min(40, len(frame)), random_state=42).sort_index()
            small = self.folder / (name + "-sample.json")
            save(small, records(subset, fields))
            self.sample_paths[name] = str(small)
            key = {"estate": "Unnamed: 0", "movies": "id", "movies_sema": "movie_id"}.get(self.job["dataset"])
            self.sample_ids = (key, set(subset[key])) if key else None
            if self.job["dataset"] == "appstore":
                self.sample_ids = ("id", set(subset.index))
        return name

    def step(self, source, operations):
        name = self.name("step")
        self.ops.extend(operations)
        self.steps.append({"name": name, "input": source, "operations": [op["name"] for op in operations]})
        return name

    def emit(self, node):
        if id(node) in self.memo:
            return self.memo[id(node)]
        if node.op == "scan":
            ref = self.source(node.data, node.fields, node.options["name"],
                              sample=self.job["dataset"] == "appstore" and node.options["name"] == "user_reviews")
        else:
            children = [self.emit(child) for child in node.children]
            name = self.name(node.op)
            if node.op == "filter" and node.options["semantic"]:
                ref = self.step(children[0], [semantic("filter", node.options["task"], node.options["columns"], name)])
            elif node.op == "join":
                op = {"name": name, "type": "equijoin", "optimize": False}
                if node.options["semantic"]:
                    columns = node.options["columns"]
                    op.update(comparison_prompt=instruction(node.options["task"], [
                        '{{ ' + ('left' if column in node.children[0].fields else 'right') + '[' + json.dumps(column) + '] }}'
                        for column in columns
                    ]), blocking_conditions=["True"], timeout=180,
                    litellm_completion_kwargs={"temperature": 0, "max_tokens": 1024})
                else:
                    op.update(comparison_prompt="Ordinary SQL key join: {{ left }} / {{ right }}", comparison_sql_join={
                        "keys": node.options["keys"], "left_fields": node.children[0].fields, "right_fields": node.children[1].fields,
                    })
                ref = self.name("step")
                self.ops.append(op)
                self.steps.append({"name": ref, "operations": [{name: {"left": children[0], "right": children[1]}}]})
            else:
                spec = {"op": node.op, "options": node.options, "fields": node.fields, "input_fields": node.children[0].fields}
                operations = [
                    {"name": name, "type": "code_reduce", "reduce_key": "_all", "concurrent_thread_count": 16,
                     "code": "from experiments.system_comparison.backends.moar.runtime import ordinary_rows\ndef transform(rows):\n    return ordinary_rows(rows, " + repr(spec) + ")\n"},
                    {"name": name + "_rows", "type": "unnest", "unnest_key": "__sql_rows", "keep_empty": False,
                     "expand_fields": list(node.fields), "comparison_sql_expand": True},
                ]
                ref = self.step(children[0], operations)
        self.memo[id(node)] = ref
        return ref

    def build(self):
        text = Path(self.job["query_file"]).read_text()
        tree = ast.parse(text)
        assignments = {target.id: ast.literal_eval(stmt.value) for stmt in tree.body
                       if isinstance(stmt, ast.Assign) for target in stmt.targets
                       if isinstance(target, ast.Name) and isinstance(stmt.value, (ast.Constant, ast.Dict))}
        self.aggregate = assignments.get("AGGREGATE")
        if "RELATIONAL_SQL" in assignments:
            sql, report = inline_single_use(assignments["RELATIONAL_SQL"])
            save(self.folder / "cte-simplification.json", report)
            compiler = Compiler(self.tables)
            relation = compiler.compile(sql)
            save(self.folder / "logical-graph.json", describe(relation.node))
            source = self.emit(relation.node)
            self.output_mapping = relation.columns
            self.output_schema = self.job["output_schema"]
        else:
            dataset = {"estate": "nirvana_estate", "movies": "nirvana_movies", "movies_sema": "movie"}[self.job["dataset"]]
            frame = self.tables[dataset]
            fields = dict(frame.attrs["duckdb_schema"])
            source = self.source(frame, fields, dataset, sample=True)
            calls = sorted((n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                            and n.func.attr in {"semantic_filter", "semantic_map"}), key=lambda n: n.lineno)
            operations = []
            for call in calls:
                args = {keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords}
                kind = "filter" if call.func.attr == "semantic_filter" else "map"
                outputs = None
                if kind == "map":
                    outputs = {name: "number?" if name in {"Price", "RuntimeMinutes"} else "string" for name in args["output_columns"]}
                    fields.update({name: "DOUBLE" if value == "number?" else "VARCHAR" for name, value in outputs.items()})
                    if args["output_columns"] == ["Amenities"]:
                        args["user_instruction"] += AMENITIES_COMPACT_INSTRUCTION
                operations.append(semantic(kind, args["user_instruction"], args["input_columns"], self.name("semantic_" + kind), outputs))
            source = self.step(source, operations)
            self.output_schema = list(fields.items()) if self.aggregate else self.job["output_schema"]
            self.output_mapping = [(name, name) for name, _ in self.output_schema]
        # Project only declared output fields when reading final rows. This
        # also removes DocETL bookkeeping without changing the pipeline.
        return {
            "default_model": "openai/semantic-large", "bypass_cache": True, "plan_rewrites": False,
            "default_lm_api_base": self.job["endpoint"], "datasets": self.datasets,
            "operations": self.ops, "pipeline": {"steps": self.steps, "output": {"type": "file", "path": str(self.folder / "native-output.json")}},
        }
