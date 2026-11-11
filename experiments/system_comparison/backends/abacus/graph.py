"""Lower shared SQL graphs to native Abacus operators."""

from pathlib import Path

import pandas as pd

from experiments.core.io import save
from experiments.system_comparison.execution.operators import (
    Predicate,
    context_frame,
    python_type,
    reference_counts,
)
from experiments.system_comparison.reporting.timing import stage
from experiments.validation.native_api import boolean_instruction, instruction
from experiments.validation.runtime.native_graph import Scalar, q


class Abacus:
    def __init__(self, job):
        self.job = job
        self.serial = 0
        self.memo = {}
        self.shared_counts = None
        self.root_id = None
        self.active_shared = set()

    def source(self, data, fields, name):
        import palimpzest as pz

        schema = [
            {"name": n, "type": python_type(t) | None, "desc": "Frozen SQL input " + n}
            for n, t in fields.items()
        ]
        clean = data.astype(object).where(pd.notna(data), None)
        return pz.MemoryDataset(
            id=self.job["id"] + ":" + name, vals=clean, schema=schema
        )

    @staticmethod
    def columns(fields, names):
        return [
            {
                "name": n,
                "type": python_type(fields[n]) | None,
                "desc": "Deterministic SQL expression",
            }
            for n in names
        ]

    def emit(self, node):
        identity = id(node)
        if (
            self.shared_counts
            and self.shared_counts.get(identity, 0) > 1
            and identity != self.root_id
            and identity not in self.active_shared
            and node.op != "scan"
        ):
            # Native lineage objects are trees, while a SQL CTE is a DAG edge.
            # Materialize a shared CTE once, then expose its frozen output as a
            # new scan for each consumer.  This prevents stochastic semantic
            # predicates from being called twice for the same CTE rows.
            cached = self.memo.get(identity)
            if cached is None:
                self.active_shared.add(identity)
                try:
                    cached = self.execute(node)
                finally:
                    self.active_shared.remove(identity)
            return self.source(cached, node.fields, f"cte:{identity}")
        if node.op == "scan":
            return self.source(
                node.data,
                node.fields,
                node.options["name"] + ":" + next(iter(node.fields)),
            )
        if node.op == "context":
            child_id = id(node.children[0])
            self.active_shared.add(child_id)
            try:
                data = self.execute(node.children[0])
            finally:
                self.active_shared.discard(child_id)
            with stage("host", "explicit SQL context boundary"):
                data = context_frame(data, node.options)
            return self.source(data, node.fields, "context:" + next(iter(node.fields)))
        children = [self.emit(child) for child in node.children]
        ds = children[0]
        if node.op == "map":
            scalar = Scalar(node.options["expressions"], node.children[0].fields)
            return ds.map(
                scalar,
                cols=self.columns(node.fields, node.options["expressions"]),
                depends_on=list(scalar.fields),
            )
        if node.op == "project":
            return ds.project(node.options["columns"])
        if node.op == "filter":
            if node.options["semantic"]:
                task = boolean_instruction(
                    instruction(node.options["task"], node.options["columns"])
                )
                return ds.sem_filter(task, depends_on=node.options["columns"])
            predicate = Predicate(node.options["expression"], node.fields)
            return ds.filter(predicate, depends_on=list(predicate.scalar.fields))
        if node.options["semantic"]:
            task = boolean_instruction(
                instruction(node.options["task"], node.options["columns"])
            )
            return ds.sem_join(
                children[1], condition=task, depends_on=node.options["columns"]
            )
        left, right = children
        keys = node.options["keys"]
        names = []
        for a, b in keys or [(None, None)]:
            self.serial += 1
            name = f"joinKey{self.serial}"
            names.append(name)
            for side, field in (("left", a), ("right", b)):
                expression = q(field) if field else "1::BIGINT"
                original = node.children[0 if side == "left" else 1].fields
                scalar = Scalar({name: expression}, original)
                kind = original[field] if field else "BIGINT"
                target = left if side == "left" else right
                if field:
                    predicate = Predicate(q(field) + " IS NOT NULL", original)
                    target = target.filter(predicate, depends_on=[field])
                target = target.map(
                    scalar,
                    cols=self.columns({name: kind}, [name]),
                    depends_on=list(scalar.fields),
                )
                if side == "left":
                    left = target
                else:
                    right = target
        return left.join(right, on=names, how="inner")

    def execute(self, node):
        if self.shared_counts is None:
            self.shared_counts = reference_counts(node)
            self.root_id = id(node)
        cached = self.memo.get(id(node))
        if cached is not None:
            return cached.copy()
        from experiments.system_comparison.backends.abacus.optimizer import (
            optimize_and_run,
        )
        from experiments.system_comparison.backends.abacus.sampling import (
            EmptyCandidates,
            training_set,
        )
        from experiments.system_comparison.execution.worker import abacus_config

        self.serial += 1
        folder = (
            Path(self.job["output"]) / "native-regions" / f"region-{self.serial:02d}"
        )
        folder.mkdir(parents=True, exist_ok=True)
        region = {**self.job, "output": str(folder)}
        with stage("planning", "construct native Abacus region"):
            ds = self.emit(node)
        samples = None
        if region.get("abacus_optimizer", {}).get("sampling_enabled", True):
            try:
                samples = training_set(ds, region)
            except EmptyCandidates:
                save(
                    folder / "mechanism.json",
                    {
                        "completed": True,
                        "proven_empty_relational_candidates": True,
                        "sampling_not_applicable": True,
                        "proof": "all semantic predicates set TRUE; ordinary SQL candidate set empty",
                    },
                )
                output = pd.DataFrame(columns=list(node.fields), dtype=object)
                self.memo[id(node)] = output
                return output.copy()
        result = optimize_and_run(
            ds, region, config=abacus_config(region), train_dataset=samples
        )
        rows = [{name: record[name] for name in node.fields} for record in result]
        output = pd.DataFrame(rows, columns=list(node.fields), dtype=object)
        self.memo[id(node)] = output
        return output.copy()
