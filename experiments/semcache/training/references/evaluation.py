"""Fixed-layout heuristic teacher references, saved once for both test preferences."""

import hashlib
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path

from planning.backends.execution import DuckDBEvaluator
from planning.settings.configuration import load_config
from reference_answers.execution.semcache import HeuristicTeacherPolicy
from reference_answers.sql.aggregate import capture_plan

from training.artifacts import read, save
from training.references.scoring import score_execution_pair


def policy(teacher, settings=None):
    settings = settings or load_config()
    return {
        "protocol": "semcache-calibrated-teacher-gotdf-v3",
        "planner": "heuristic_requests",
        "sampling": "independent_base_operator_selectivity",
        "operator_sample_rows": settings["sampling"].get("operator_sample_rows", 20),
        "relational_sample_rows": settings["sampling"].get(
            "relational_sample_rows", 200
        ),
        "search": {
            "beam_width": settings["search"]["beam_width"],
            **settings["search"]["inference"],
        },
        "objective_units": "seconds",
        "operator_time_calibration": settings["calibration"]["operator_time"],
        "request_granularity": "one_record_or_pair",
        "layout": "GOTDF",
        "teacher": teacher.configuration(),
    }


def relocated_reference(result, root, key):
    """Resolve a saved attempt within this query after its answer root moves."""
    artifact = Path(result["artifact"])
    query_parts = Path(key).parts
    suffix = artifact.parts[-3:]
    if (
        len(artifact.parts) < len(query_parts) + 3
        or artifact.parts[-len(query_parts) - 3 : -3] != query_parts
        or len(suffix) != 3
        or re.fullmatch(r"attempt-[1-9][0-9]*", suffix[0]) is None
        or suffix[1] != "sql"
        or re.fullmatch(r"run-[1-9][0-9]*", suffix[2]) is None
    ):
        raise ValueError(f"Invalid saved reference artifact for {key}: {artifact}")
    root = Path(root).resolve()
    current = (root / Path(*suffix)).resolve()
    if not current.is_relative_to(root):
        raise ValueError(f"Saved reference artifact leaves query directory: {current}")
    if not current.is_dir():
        raise FileNotFoundError(f"Missing saved reference artifact: {current}")
    return {**result, "artifact": str(current)}


@contextmanager
def retry_environment():
    """Only the reference subprocess gets a bounded external API backoff."""
    defaults = {
        "SEMCACHE_RATE_LIMIT_RETRIES": "8",
        "SEMCACHE_SERVER_ERROR_RETRIES": "3",
    }
    added = [name for name in defaults if name not in os.environ]
    try:
        for name in added:
            os.environ[name] = defaults[name]
        yield
    finally:
        for name in added:
            os.environ.pop(name, None)


def reference(
    output,
    key,
    sql,
    database,
    teacher,
    build,
    settings,
    timeout=None,
    *,
    reuse_only=False,
):
    root = Path(output) / key
    contract = {
        "sql_sha256": hashlib.sha256(sql.encode()).hexdigest(),
        "database": str(database),
        **policy(teacher, settings),
    }
    with Path(database).open("rb") as stream:
        contract["database_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    saved = root / "reference.json"
    if saved.exists():
        previous = read(saved)
        if previous["contract"] != contract:
            raise ValueError("Frozen teacher reference inputs or policy changed")
        if previous["result"]["status"] == "completed":
            return relocated_reference(previous["result"], root, key), True
    if reuse_only:
        raise FileNotFoundError(f"No completed shared reference for {key}: {saved}")
    attempt = 1
    while (root / f"attempt-{attempt}").exists():
        attempt += 1
    folder = root / f"attempt-{attempt}"
    folder.mkdir(parents=True)
    planner = HeuristicTeacherPolicy(
        teacher, folder, build=build, database=database, settings=settings
    )
    with DuckDBEvaluator(
        build,
        teacher.transport_models({}),
        sql,
        database=database,
        output=folder / "sql",
        timeout=timeout,
        parallelism=settings["execution"]["cpu_parallelism"],
        max_concurrent_requests_per_model=settings["execution"][
            "max_concurrent_requests_per_model"
        ],
        enforce_model_operator_order=settings["execution"][
            "enforce_model_operator_order"
        ],
    ) as evaluator:
        # Teacher answers do not consume or evict the candidate cache.
        evaluator.response_cache = {
            **evaluator.response_cache,
            "directory": str(Path(output) / ".teacher-response-cache"),
        }
        with retry_environment():
            result = evaluator.run(
                policy_override=planner,
                allow_timeout=True,
                capture_aggregate_input=capture_plan(sql) is not None,
                include_rows=False,
            )
    for key_to_remove in ("events", "csv", "rows"):
        result.pop(key_to_remove, None)
    result["execution_timeout_seconds"] = timeout
    save(saved, {"contract": contract, "result": result})
    return result, False


def complete_quality(
    entry,
    directory,
    *,
    output,
    key,
    sql,
    database,
    teacher,
    teacher_config,
    build,
    settings,
    timeout,
    stage,
    reuse_only=False,
    saved_references=None,
):
    entry.setdefault("reference_attempts", []).append(
        {
            k: entry[k]
            for k in (
                "reference_status",
                "reference_error",
                "reference_artifact",
                "accuracy",
                "reference_policy",
            )
            if k in entry
        }
    )
    entry.pop("reference_error", None)
    entry["reference_policy"] = (
        saved_references.policy
        if saved_references is not None
        else policy(teacher, settings)
    )
    try:
        entry["reference_timeout_seconds"] = timeout
        stage("复用参考" if saved_references is not None else "参考执行")
        if saved_references is not None:
            expected, reused = saved_references.reference(key, sql, database)
            entry.update(
                reference_source_run=expected["source_run"],
                reference_pre_aggregate_path=expected.get("pre_aggregate_path"),
                quality_basis="saved reference run agreement on non-free-text rows; aggregate inputs before aggregation; not human ground truth",
            )
        else:
            expected, reused = reference(
                output,
                key,
                sql,
                database,
                teacher,
                build,
                settings,
                timeout,
                reuse_only=reuse_only,
            )
        entry.update(
            reference_status=expected["status"],
            reference_artifact=expected["artifact"],
            reference_seconds=expected["seconds"],
            reference_reused=reused,
            reference_timeout_seconds=expected.get(
                "execution_timeout_seconds", timeout
            ),
        )
        stage("评分中")
        started = time.monotonic()
        entry["accuracy"] = (
            score_execution_pair(
                {"artifact": entry["result_artifact"]},
                expected,
                key.removeprefix("translated/semcache/"),
                output=directory / "accuracy-judgments",
                teacher_config=teacher_config,
            )
            if expected["status"] == "completed"
            else None
        )
        entry["scoring_seconds"] = time.monotonic() - started
    except (OSError, ValueError, KeyError, RuntimeError, TimeoutError) as error:
        entry.update(
            reference_status="failed", reference_error=str(error), accuracy=None
        )
