"""Reuse frozen answers only for explicitly validated SQL-equivalent rewrites."""

import copy
import hashlib
import json
from pathlib import Path

from reference_answers.storage.files import file_digest

from .answers import TrainingAnswers, sql_digest


class EquivalentTrainingAnswers:
    def __init__(self, root, workloads, databases, validation):
        path = Path(validation)
        evidence = json.loads(path.read_text())
        if evidence.get("status") != "completed":
            raise ValueError("SQL equivalence validation is incomplete")
        rows = {row["query"]: row for row in evidence["rows"]}
        frozen = {
            w["id"]: w
            for w in json.loads((Path(root) / "workloads.json").read_text())["workloads"]
        }
        self.validation_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.originals, self.execution_hashes, self.rewritten = {}, {}, set()
        originals = []
        for workload in workloads:
            key = workload["id"]
            original = frozen[key]
            source = original["source_sql"]
            execution_hash = sql_digest(workload["sql"])
            if workload["sql"].strip() != source.strip():
                row = rows.get(key, {})
                if (
                    row.get("original_sql_sha256") != sql_digest(source)
                    or row.get("rewritten_sql_sha256") != execution_hash
                    or not row.get("sampling_templates_bound")
                    or row.get("equivalence_cases", 0) < 1
                    or evidence.get("database_sha256")
                    != file_digest(databases[workload["dataset"]])
                ):
                    raise ValueError(f"No validated equivalent reference for {key}")
                self.rewritten.add(key)
            old = copy.deepcopy(workload)
            old.update(sql=source, sql_sha256=sql_digest(source))
            self.originals[key] = old
            self.execution_hashes[key] = execution_hash
            originals.append(old)
        self.base = TrainingAnswers(root, originals, databases)
        self.system = self.base.system

    def load(self, workload):
        key = workload["id"]
        if sql_digest(workload["sql"]) != self.execution_hashes[key]:
            raise ValueError(f"Execution SQL changed after equivalence validation: {key}")
        answer = self.base.load(self.originals[key])
        if key in self.rewritten:
            answer = {
                **answer,
                "execution_sql_sha256": self.execution_hashes[key],
                "reference_sql_sha256": self.originals[key]["sql_sha256"],
                "equivalence_validation_sha256": self.validation_sha256,
            }
        return answer

    def check_generator(self):
        return self.base.check_generator()
