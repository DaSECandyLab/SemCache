"""Lower shared SQL graphs to native Lotus operators."""

from experiments.system_comparison.execution.operators import (
    Predicate,
    SQLJoin,
    context_frame,
)
from experiments.system_comparison.reporting.timing import stage
from experiments.validation.native_api import instruction
from experiments.validation.runtime.native_graph import Scalar


class Lotus:
    def __init__(self, job):
        self.job = job
        self.memo = {}

    def execute(self, node):
        cached = self.memo.get(id(node))
        if cached is not None:
            return cached.copy()
        if node.op == "scan":
            return node.data.copy()
        children = [self.execute(child) for child in node.children]
        frame = children[0]
        if node.op == "context":
            with stage("host", "explicit SQL context boundary"):
                output = context_frame(frame, node.options)
                self.memo[id(node)] = output
                return output.copy()
        if node.op == "project":
            output = frame[node.options["columns"]].copy()
            self.memo[id(node)] = output
            return output.copy()
        if node.op == "map":
            with stage("host", "ordinary SQL expressions"):
                scalar = Scalar(node.options["expressions"], node.children[0].fields)
                frame = frame.copy()
                frame[list(node.options["expressions"])] = scalar.frame(frame)
                self.memo[id(node)] = frame
                return frame.copy()
        if node.op == "filter":
            if node.options["semantic"]:
                task = instruction(
                    node.options["task"],
                    ["{" + n + "}" for n in node.options["columns"]],
                )
                return frame.sem_filter(task)
            with stage("host", "ordinary SQL predicate"):
                predicate = Predicate(node.options["expression"], node.fields)
                return frame.loc[predicate.scalar.frame(frame)["keep"].eq(True)].copy()
        if node.options["semantic"]:
            from experiments.system_comparison.backends.lotus.join import join

            return join(frame, children[1], node.options, self.job)
        with stage("host", "ordinary SQL join"):
            return SQLJoin(node.options["keys"]).frames(frame, children[1])
