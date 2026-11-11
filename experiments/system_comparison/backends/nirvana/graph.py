"""Lower shared SQL graphs to native Nirvana operators."""

import copy
from pathlib import Path

import pandas as pd

from experiments.system_comparison.execution.operators import (
    Predicate,
    SQLJoin,
    context_frame,
    python_type,
    reference_counts,
)
from experiments.system_comparison.reporting.timing import stage
from experiments.validation.native_api import boolean_instruction, instruction
from experiments.validation.runtime.native_graph import Scalar


class Nirvana:
    def __init__(self, job):
        self.job = job
        self.serial = 0
        self.memo = {}
        self.shared_counts = None
        self.root_id = None
        self.active_shared = set()

    def source(self, data, fields):
        from unittest.mock import patch

        import nirvana as nv
        import nirvana.dataframe.frame as module

        declared = getattr(module, "_comparison_text_columns", set())
        declared.update(n for n, t in fields.items() if python_type(t) is str)
        module._comparison_text_columns = declared

        # The frozen SQL schema already supplies each type. Nirvana's object
        # inference otherwise applies .str to object-stored numbers and treats
        # text URLs as media, changing the declared workload.
        def declared_type(column):
            return column, python_type(fields[column.name])

        with patch.object(module, "infer_and_convert_dtype", declared_type):
            return nv.DataFrame(data.astype(object).where(pd.notna(data), None))

    def emit(self, node):
        identity = id(node)
        if (
            self.shared_counts
            and self.shared_counts.get(identity, 0) > 1
            and identity != self.root_id
            and identity not in self.active_shared
            and node.op != "scan"
        ):
            cached = self.memo.get(identity)
            if cached is None:
                self.active_shared.add(identity)
                try:
                    cached = self.execute(node)
                finally:
                    self.active_shared.remove(identity)
            return self.source(cached, node.fields)
        if node.op == "scan":
            return self.source(node.data, node.fields)
        if node.op == "context":
            child_id = id(node.children[0])
            self.active_shared.add(child_id)
            try:
                data = self.execute(node.children[0])
            finally:
                self.active_shared.discard(child_id)
            with stage("host", "explicit SQL context boundary"):
                data = context_frame(data, node.options)
            return self.source(data, node.fields)
        children = [self.emit(child) for child in node.children]
        df = copy.copy(children[0])
        if node.op == "project":
            df.leaf_node = copy.copy(df.leaf_node)
            df.leaf_node.node_fields = copy.copy(df.leaf_node.node_fields)
            df.leaf_node.node_fields.output_fields = list(node.fields)
            return df
        if node.op == "map":
            scalar = Scalar(node.options["expressions"], node.children[0].fields)
            df.semantic_map(
                user_instruction="Evaluate deterministic SQL expressions.",
                input_columns=list(scalar.fields),
                output_columns=list(node.options["expressions"]),
                func=scalar,
            )
        elif node.op == "filter":
            if node.options["semantic"]:
                columns = node.options["columns"]
                task = boolean_instruction(
                    instruction(
                        node.options["task"], ["the value of " + n for n in columns]
                    )
                )
                df.semantic_filter(user_instruction=task, input_columns=columns)
            else:
                predicate = Predicate(node.options["expression"], node.fields)
                df.semantic_filter(
                    user_instruction="Evaluate the SQL predicate: "
                    + node.options["expression"],
                    input_columns=list(predicate.scalar.fields),
                    func=predicate,
                )
        else:
            if node.options["semantic"]:
                task = boolean_instruction(
                    instruction(
                        node.options["task"],
                        ["the value of " + n for n in node.options["columns"]],
                    )
                )
                kwargs = {
                    "user_instruction": task,
                    "left_on": node.options["left_columns"],
                    "right_on": node.options["right_columns"],
                    "how": "inner",
                    "strategy": "nest",
                }
            else:
                tool = SQLJoin(
                    node.options["keys"],
                    left_columns=tuple(node.children[0].fields),
                    right_columns=tuple(node.children[1].fields),
                )
                kwargs = {
                    "user_instruction": "Evaluate an ordinary SQL inner join.",
                    "left_on": list(node.children[0].fields),
                    "right_on": list(node.children[1].fields),
                    "how": "inner",
                    "strategy": "nest",
                    "tool": tool,
                }
            df.add_operator(
                op_name="join",
                op_kwargs=kwargs,
                data_kwargs={
                    "left_input_fields": list(node.children[0].fields),
                    "right_input_fields": list(node.children[1].fields),
                    "output_fields": list(node.fields),
                },
                other=children[1],
            )
        df.leaf_node.node_fields.output_fields = list(node.fields)
        return df

    def execute(self, node):
        if self.shared_counts is None:
            self.shared_counts = reference_counts(node)
            self.root_id = id(node)
        cached = self.memo.get(id(node))
        if cached is not None:
            return cached.copy()
        from experiments.system_comparison.backends.nirvana.optimizer import (
            optimize_and_execute,
        )

        self.serial += 1
        folder = (
            Path(self.job["output"]) / "native-regions" / f"region-{self.serial:02d}"
        )
        folder.mkdir(parents=True, exist_ok=True)
        with stage("planning", "construct native Nirvana region"):
            df = self.emit(node)
        output, _, _, _ = optimize_and_execute(df, {**self.job, "output": str(folder)})
        output = output[list(node.fields)].copy()
        self.memo[id(node)] = output
        return output.copy()
