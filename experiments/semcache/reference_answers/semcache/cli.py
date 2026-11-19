"""Collect training references using SemCache, a teacher and fixed GOTDF."""

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path

from planning.settings.configuration import (
    ROOT,
    activate_native,
    load_config,
    response_cache_settings,
    runtime_path,
    teacher_path,
)
from reference_answers.sql.training_sources import load_training_sources
from reference_answers.storage.files import file_digest
from training.artifacts import save

from .collector import collect, configured_teacher, timestamp

PROTOCOL = "semcache-calibrated-teacher-gotdf-v3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workloads", type=Path)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("movies_nirvana", "estate", "appstore", "movies_sema", "tpch"),
    )
    parser.add_argument("--teacher-config", type=Path, action="append")
    parser.add_argument("--allow-teacher-routing-update", action="store_true")
    parser.add_argument("--parallelism", type=int)
    parser.add_argument("--rate-limit-retries", type=int, default=8)
    parser.add_argument("--server-error-retries", type=int, default=3)
    parser.add_argument("--allow-transport-update", action="store_true")
    parser.add_argument(
        "--allow-workload-update", action="store_true",
        help="Update query membership while retaining saved answers for unchanged SQL",
    )
    parser.add_argument(
        "--allow-planner-update", action="store_true",
        help="Record a planner-code update while keeping SQL, teacher and answer semantics fixed",
    )
    parser.add_argument("--allow-cache-policy-update", action="store_true",
                        help="Apply a changed second-use verification policy and regenerate answers from the old policy")
    parser.add_argument(
        "--query-workers",
        type=int,
        help="Queries to execute concurrently; no shared HTTP concurrency cap",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable local answer caching and identical-request reuse",
    )
    args = parser.parse_args()
    root = args.output.resolve()
    config = load_config()
    verification = config["reference_answers"]["semcache"].get("verify_on_second_use")
    if verification is not None:
        config["response_cache"]["verify_on_second_use"] = verification
    settings = copy.deepcopy(config["reference_answers"]["semcache"])
    settings["runtime_configuration"] = copy.deepcopy(config)
    settings["planning_contract"] = {
        "search": copy.deepcopy(config["search"]),
        "sampling": copy.deepcopy(config["sampling"]),
    }
    settings["enforce_model_operator_order"] = config["execution"][
        "enforce_model_operator_order"
    ]
    settings["cache_enabled"] = not args.no_cache
    settings["verify_on_second_use"] = response_cache_settings(config)[
        "verify_on_second_use"
    ]
    settings["boolean_output_protocol"] = "last-boolean-token-icase-v1"
    settings["http_metadata_protocol"] = "string-values-v1"
    if not 0 <= args.rate_limit_retries <= 20:
        parser.error("--rate-limit-retries must be in [0,20]")
    settings["rate_limit_retries"] = args.rate_limit_retries
    os.environ["SEMCACHE_RATE_LIMIT_RETRIES"] = str(args.rate_limit_retries)
    if not 0 <= args.server_error_retries <= 10:
        parser.error("--server-error-retries must be in [0,10]")
    settings["server_error_retries"] = args.server_error_retries
    os.environ["SEMCACHE_SERVER_ERROR_RETRIES"] = str(args.server_error_retries)
    settings["query_workers"] = args.query_workers or 1
    settings["selected_datasets"] = (
        sorted(set(args.datasets)) if args.datasets else None
    )
    if args.query_workers is not None and not 1 <= args.query_workers <= 32:
        parser.error("--query-workers must be in [1,32]")
    settings["response_cache_directory"] = (
        str(root / ".cache" / "teacher") if settings["cache_enabled"] else None
    )
    if args.parallelism is not None:
        settings["parallelism"] = args.parallelism
    if (
        type(settings["parallelism"]) is not int
        or not 1 <= settings["parallelism"] <= 128
    ):
        parser.error("--parallelism must be in [1,128]")
    if settings["planner"] != "heuristic_requests" or settings["layout"] != "GOTDF":
        parser.error("Training references require heuristic_requests and fixed GOTDF")
    paths = args.teacher_config or [teacher_path(config)]
    routes = {
        path.stem.removeprefix("teacher."): configured_teacher(path, settings)
        for path in paths
    }
    if len(routes) != len(paths):
        parser.error("Teacher configuration names must be distinct")
    if len(routes) > 1:
        settings["query_workers"] = len(routes)
    teacher = next(iter(routes.values()))
    databases = runtime_path(config, "source_databases")
    workloads, source = load_training_sources(
        args.workloads or ROOT / config["training"]["workload_manifest"], databases
    )
    build = runtime_path(config, "build")
    assert isinstance(build, Path)
    native_bindings = runtime_path(config, "native_bindings")
    assert isinstance(native_bindings, Path)
    binaries = [
        build / "duckdb",
        build / "src/libduckdb.so",
        build / "extension/semantic/semantic.duckdb_extension",
        build / "extension/beam_search/beam_search.duckdb_extension",
    ]
    bindings = sorted((native_bindings / "planning/kernels").glob("_search_core*.so"))
    if not bindings:
        raise FileNotFoundError("Configured native planner binding is not built")
    binaries.extend(bindings)
    contract = {
        "version": 5,
        "system": "semcache",
        "response_protocol": PROTOCOL,
        "aggregate_input_protocol": "final_aggregate_input_v1",
        "teacher": teacher.configuration(),
        "settings": settings,
        "database_sha256": {k: file_digest(v) for k, v in databases.items()},
        "native_sha256": {str(p): file_digest(p) for p in binaries},
        "planner_sha256": {
            name: file_digest(
                ROOT.parent / "experiments/semcache" / name
                if name.startswith("reference_answers/")
                else ROOT / "optimizer" / name
            )
            for name in (
                "reference_answers/execution/semcache.py",
                "planning/search/heuristics.py",
                "planning/search/beam.py",
                "planning/search/config.py",
                "planning/cost/requests.py",
                "planning/cost/row_time.py",
                "planning/cost/estimates.py",
                "planning/backends/sampling.py",
                "planning/backends/relational_estimates.py",
                "planning/kernels/raw_batch.py",
                "planning/search/results.py",
                "planning/state/problem/__init__.py",
                "planning/state/problem/actions.py",
                "planning/state/problem/construction.py",
                "planning/state/problem/replay.py",
                "planning/state/problem/transitions.py",
                "planning/state/trees.py",
                "planning/settings/modes.py",
            )
        },
        "queries": {w["id"]: w["sql_sha256"] for w in workloads},
        "training_workload": source,
    }
    if len(routes) > 1:
        from reference_answers.execution.lanes import assign_teachers

        contract["teachers"] = {
            name: model.configuration() for name, model in routes.items()
        }
        contract["active_teacher_routes"] = list(routes)
        contract["query_teachers"] = assign_teachers(
            root,
            workloads,
            routes,
            allow_routing_update=args.allow_teacher_routing_update,
        )
        overrides = {}
        for workload in workloads:
            saved = root / workload["id"] / "answer.json"
            if saved.exists():
                original = json.loads(saved.read_text())["teacher"]
                name = contract["query_teachers"][workload["id"]]
                if name not in contract["teachers"]:
                    contract["teachers"][name] = original
                if original != contract["teachers"][name]:
                    overrides[workload["id"]] = original
        if overrides:
            contract["query_teacher_configs"] = overrides
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".generation.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = root / "contract.json"
        if previous.exists() and json.loads(previous.read_text()) != contract:
            frozen = json.loads(previous.read_text())
            if args.allow_workload_update:
                changed = [
                    identity
                    for identity in frozen["queries"].keys() & contract["queries"].keys()
                    if frozen["queries"][identity] != contract["queries"][identity]
                ]
                if changed:
                    raise ValueError(
                        "Workload membership update cannot change existing SQL: "
                        + ", ".join(sorted(changed))
                    )

            def answer_contract(value):
                value = copy.deepcopy(value)
                # Planning, transport and response semantics are already frozen
                # explicitly below; unrelated training settings may change.
                value["settings"].pop("runtime_configuration", None)
                value["settings"].pop("query_workers", None)
                value["settings"].pop("selected_datasets", None)
                value["settings"].pop("rate_limit_retries", None)
                if args.allow_workload_update:
                    value.pop("queries", None)
                    value.pop("training_workload", None)
                if args.allow_planner_update:
                    value.pop("planner_sha256", None)
                    beam_extension = str(
                        build / "extension/beam_search/beam_search.duckdb_extension"
                    )
                    value["native_sha256"][beam_extension] = contract["native_sha256"][
                        beam_extension
                    ]
                if args.allow_cache_policy_update:
                    value["settings"].pop("verify_on_second_use", None)
                if args.allow_teacher_routing_update:
                    value.pop("teachers", None)
                    value.pop("query_teachers", None)
                    value.pop("query_teacher_configs", None)
                    value.pop("active_teacher_routes", None)
                if args.allow_transport_update:
                    value["settings"].pop("server_error_retries", None)
                    value["settings"].pop("boolean_output_protocol", None)
                    value["settings"].pop("http_metadata_protocol", None)
                    extension = str(
                        build / "extension/semantic/semantic.duckdb_extension"
                    )
                    value["native_sha256"][extension] = contract["native_sha256"][
                        extension
                    ]
                return value

            if answer_contract(frozen) != answer_contract(contract):
                raise ValueError(
                    "Reference contract changed; restore the saved configuration "
                    "or clear incompatible results in the existing output directory"
                )
            if args.allow_workload_update and frozen["queries"] != contract["queries"]:
                with (root / "workload-changes.jsonl").open("a") as log:
                    log.write(json.dumps({
                        "time": timestamp(),
                        "retained": sorted(frozen["queries"].keys() & contract["queries"].keys()),
                        "added": sorted(contract["queries"].keys() - frozen["queries"].keys()),
                        "removed_from_selection": sorted(frozen["queries"].keys() - contract["queries"].keys()),
                        "saved_answer_policy": "reuse unchanged SQL; retain historical files",
                    }) + "\n")
            if args.allow_planner_update and (
                frozen["planner_sha256"] != contract["planner_sha256"]
                or frozen["native_sha256"].get(str(build / "extension/beam_search/beam_search.duckdb_extension"))
                != contract["native_sha256"].get(str(build / "extension/beam_search/beam_search.duckdb_extension"))
            ):
                with (root / "planner-changes.jsonl").open("a") as log:
                    log.write(json.dumps({
                        "time": timestamp(),
                        "planner_before": frozen["planner_sha256"],
                        "planner_after": contract["planner_sha256"],
                        "saved_answer_policy": "reuse unchanged SQL and teacher; new answers use current planner",
                    }) + "\n")
            if frozen["settings"].get("verify_on_second_use") != settings["verify_on_second_use"]:
                with (root / "cache-policy-changes.jsonl").open("a") as log:
                    log.write(json.dumps({
                        "time": timestamp(),
                        "verify_on_second_use_before": frozen["settings"].get("verify_on_second_use"),
                        "verify_on_second_use_after": settings["verify_on_second_use"],
                        "old_policy_answers": "regenerate_using_existing_prompt_cache",
                    }) + "\n")
            if frozen["native_sha256"] != contract["native_sha256"]:
                with (root / "transport-changes.jsonl").open("a") as log:
                    log.write(
                        json.dumps(
                            {
                                "time": timestamp(),
                                "reason": "Native transport/execution update; prompts/model unchanged",
                                "native_before": frozen["native_sha256"],
                                "native_after": contract["native_sha256"],
                                "rate_limit_retries": settings["rate_limit_retries"],
                                "server_error_retries": settings[
                                    "server_error_retries"
                                ],
                                "boolean_output_protocol": settings[
                                    "boolean_output_protocol"
                                ],
                            }
                        )
                        + "\n"
                    )
            if frozen.get("teachers") != contract.get("teachers") or frozen.get(
                "query_teachers"
            ) != contract.get("query_teachers"):
                with (root / "teacher-routing-changes.jsonl").open("a") as log:
                    log.write(
                        json.dumps(
                            {
                                "time": timestamp(),
                                "teachers_before": frozen.get(
                                    "teachers", {"original": frozen["teacher"]}
                                ),
                                "teachers_after": contract.get("teachers"),
                                "query_teachers": contract.get("query_teachers"),
                            }
                        )
                        + "\n"
                    )
            with (root / "scheduling-changes.jsonl").open("a") as log:
                log.write(
                    json.dumps(
                        {
                            "time": timestamp(),
                            "query_workers_before": frozen["settings"].get(
                                "query_workers", 1
                            ),
                            "query_workers_after": settings["query_workers"],
                            "teacher_http_limit_per_query": settings["parallelism"],
                            "global_teacher_http_limit": None,
                        }
                    )
                    + "\n"
                )
        save(previous, contract)
        save(
            root / "workloads.json",
            {"suite": "training_semcache", **source, "workloads": workloads},
        )
        save(
            root / "resources.json",
            {
                "teacher_http_parallelism": settings["parallelism"],
                "teacher_http_parallelism_scope": "per_query",
                "query_workers": settings["query_workers"],
                "teacher_routes": list(routes),
                "max_active_queries_per_teacher_route": 1
                if len(routes) > 1
                else settings["query_workers"],
                "global_teacher_http_limit": None,
                "rate_limit_retries": settings["rate_limit_retries"],
                "server_error_retries": settings["server_error_retries"],
                "cpu_parallelism": settings["cpu_parallelism"],
                "enforce_model_operator_order": settings[
                    "enforce_model_operator_order"
                ],
                "effective_batch_size": 1,
                "response_cache_enabled": settings["cache_enabled"],
                "identical_request_reuse": settings["cache_enabled"],
                "second_teacher_check": settings["cache_enabled"]
                and settings["verify_on_second_use"],
                "majority_vote": settings["cache_enabled"]
                and settings["verify_on_second_use"],
            },
        )
        (root / "launcher.pid").write_text(str(os.getpid()) + "\n")
        save(
            ROOT / config["training"]["reference_answers_dir"] / "latest.json",
            {"root": str(root), "system": "semcache", "protocol": PROTOCOL},
        )
        activate_native(config)
        from reference_answers.execution.semcache import execute

        collect(
            root,
            workloads,
            databases,
            build,
            teacher,
            settings,
            contract,
            execute=execute,
            teacher_routes=routes if len(routes) > 1 else None,
        )


if __name__ == "__main__":
    main()
