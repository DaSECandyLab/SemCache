"""Read frozen teacher answers produced by SemCache or the legacy Sema runner."""

import hashlib
import json
from pathlib import Path

from experiments.system_comparison.accuracy.artifacts import result_digest, typed_result
from reference_answers.semcache import PROTOCOL as SEMCACHE_REFERENCE_PROTOCOL
from reference_answers.sql.aggregate import capture_plan
from reference_answers.storage.files import file_digest


def sql_digest(sql):
    return hashlib.sha256(sql.encode()).hexdigest()


def sql_matches_reference(workload, reference):
    """Accept a validated SemCache rewrite paired with its source answer."""
    if workload["sql"].strip() == reference["source_sql"].strip():
        return True
    equivalence = workload.get("sql_equivalence")
    if not isinstance(equivalence, dict):
        return False
    if equivalence.get("original_sql_sha256") != reference["sql_sha256"]:
        return False
    validation = equivalence.get("validation")
    if not validation:
        return False
    path = Path(validation)
    if not path.is_file():
        return False
    expected = equivalence.get("validation_sha256")
    return expected is None or file_digest(path) == expected


class TrainingAnswers:
    def __init__(self, root, workloads, databases, *, allow_subset=False):
        self.root = Path(root).resolve()
        self.contract = json.loads((self.root / "contract.json").read_text())
        self.system = self.contract.get("system")
        if (
            self.system not in ("semcache", "sema")
            or self.contract.get("aggregate_input_protocol")
            != "final_aggregate_input_v1"
            or not isinstance(self.contract.get("teacher"), dict)
        ):
            raise ValueError(
                "Expected SemCache or Sema teacher references with actual aggregate input snapshots"
            )
        if (
            self.system == "semcache"
            and self.contract.get("response_protocol") != SEMCACHE_REFERENCE_PROTOCOL
        ):
            raise ValueError(
                "SemCache teacher references require the heuristic_requests protocol "
                + SEMCACHE_REFERENCE_PROTOCOL
            )
        self.stats = {}
        for name, path in databases.items():
            path = Path(path)
            if file_digest(path) != self.contract["database_sha256"].get(name):
                raise ValueError(f"Reference source database differs: {name}")
            stat = path.stat()
            self.stats[name] = (path, stat.st_size, stat.st_mtime_ns)
        frozen = json.loads((self.root / "workloads.json").read_text())["workloads"]
        self.workloads = {w["id"]: w for w in frozen}
        self.teachers = self.contract.get(
            "teachers", {"original": self.contract["teacher"]}
        )
        self.query_teachers = self.contract.get(
            "query_teachers", dict.fromkeys(self.workloads, "original")
        )
        self.query_teacher_configs = self.contract.get("query_teacher_configs", {})
        if (
            not isinstance(self.query_teacher_configs, dict)
            or not set(self.query_teacher_configs) <= set(self.workloads)
            or any(
                not isinstance(value, dict)
                or not isinstance(value.get("model_id"), str)
                or not value["model_id"]
                or not isinstance(value.get("endpoint"), dict)
                for value in self.query_teacher_configs.values()
            )
        ):
            raise ValueError("Historical query teacher configurations are invalid")
        if (
            not isinstance(self.teachers, dict)
            or not self.teachers
            or any(not isinstance(value, dict) for value in self.teachers.values())
            or not isinstance(self.query_teachers, dict)
            or set(self.query_teachers) != set(self.workloads)
            or any(
                not isinstance(name, str) or name not in self.teachers
                for name in self.query_teachers.values()
            )
        ):
            raise ValueError("Reference teacher routes differ from the query set")
        selected_ids = {w["id"] for w in workloads}
        reference_ids = set(self.workloads)
        if (
            len(self.workloads) != len(frozen)
            or len(selected_ids) != len(workloads)
            or not selected_ids
            or not selected_ids <= reference_ids
            or (not allow_subset and selected_ids != reference_ids)
            or set(self.contract["queries"]) != reference_ids
        ):
            raise ValueError("Training and reference query sets differ")
        self.bases = {}
        for workload in workloads:
            reference = self.workloads[workload["id"]]
            source = reference["source_sql"]
            source_hash = sql_digest(source)
            if (
                not sql_matches_reference(workload, reference)
                or reference["dataset"] != workload["dataset"]
                or reference["dataset"] not in self.stats
                or self.contract["queries"][workload["id"]] != reference["sql_sha256"]
                or sql_digest(reference["sql"]) != reference["sql_sha256"]
                or (
                    "source_sql_sha256" in reference
                    and reference["source_sql_sha256"] != source_hash
                )
                or (
                    "sql_sha256" in workload
                    and workload["sql_sha256"] != sql_digest(workload["sql"])
                )
                or (
                    self.system == "semcache"
                    and (
                        source != reference["sql"]
                        or reference.get("source_sql_sha256") != source_hash
                    )
                )
            ):
                raise ValueError(f"Training and reference SQL differ: {workload['id']}")
            self.bases[workload["id"]] = (
                "pre_aggregate_rows"
                if capture_plan(reference["sql"]) is not None
                else "final_output"
            )

    def load(self, workload):
        reference = self.workloads[workload["id"]]
        if (
            workload["dataset"] != reference["dataset"]
            or not sql_matches_reference(workload, reference)
        ):
            raise ValueError(f"Training and reference SQL differ: {workload['id']}")
        path, size, mtime = self.stats[workload["dataset"]]
        current = path.stat()
        if (current.st_size, current.st_mtime_ns) != (size, mtime):
            raise ValueError("Source database changed during measured training")
        answer_path = self.root / workload["id"] / "answer.json"
        if not answer_path.is_file():
            raise FileNotFoundError(f"Reference pending: {workload['id']}")
        answer_bytes = answer_path.read_bytes()
        answer = json.loads(answer_bytes)
        if (
            answer.get("status") != "completed"
            or answer.get("system") != self.system
            or (
                self.system == "semcache"
                and answer.get("protocol") != SEMCACHE_REFERENCE_PROTOCOL
            )
            or answer.get("teacher")
            != self.query_teacher_configs.get(
                workload["id"], self.teachers[self.query_teachers[workload["id"]]]
            )
            or answer.get("teacher_route", self.query_teachers[workload["id"]])
            != self.query_teachers[workload["id"]]
            or answer.get("sql_sha256") != reference["sql_sha256"]
            or answer.get("sql_sha256") != self.contract["queries"][workload["id"]]
            or sql_digest(reference["sql"]) != reference["sql_sha256"]
            or answer.get("result_sha256") != result_digest(answer)
        ):
            raise ValueError(f"Invalid {self.system} reference answer: {answer_path}")
        basis = self.bases[workload["id"]]
        if answer.get("accuracy_basis") != basis:
            raise ValueError(
                f"Reference accuracy basis differs from SQL: {answer_path}"
            )
        pre = answer.get("pre_aggregate")
        if basis == "pre_aggregate_rows" and not isinstance(pre, dict):
            raise ValueError(f"Reference lacks required aggregate input: {answer_path}")
        if basis == "final_output" and pre is not None:
            raise ValueError(f"Unexpected reference aggregate input: {answer_path}")
        captured = None
        if basis == "pre_aggregate_rows":
            assert isinstance(pre, dict)
            if pre.get("protocol") != self.contract["aggregate_input_protocol"]:
                raise ValueError(
                    f"Invalid reference aggregate input protocol: {answer_path}"
                )
            snapshot_path = Path(pre["path"])
            if not snapshot_path.is_absolute():
                snapshot_path = self.root / snapshot_path
            try:
                payload = json.loads(snapshot_path.read_text())
            except FileNotFoundError as error:
                # Only a missing answer is pending; a published answer must
                # already have its complete aggregate snapshot.
                raise ValueError(
                    f"Missing reference aggregate input: {answer_path}"
                ) from error
            if (
                result_digest(payload) != pre["result_sha256"]
                or len(payload["rows"]) != pre["row_count"]
            ):
                raise ValueError(f"Invalid reference aggregate input: {answer_path}")
            captured = typed_result(payload, pre["columns"])
        return {
            "system": self.system,
            "teacher": answer["teacher"],
            "teacher_route": self.query_teachers[workload["id"]],
            "result": typed_result(answer),
            "pre_aggregate": captured,
            "basis": basis,
            "answer_path": str(answer_path),
            "answer_sha256": hashlib.sha256(answer_bytes).hexdigest(),
            "artifact": answer["artifact"],
        }

    def check_generator(self):
        progress = json.loads((self.root / "progress.json").read_text())
        if progress["status"] not in ("running", "completed", "completed_with_skips"):
            raise RuntimeError(
                f"Reference generator is {progress['status']}; pending answers cannot be scored"
            )
        return progress
