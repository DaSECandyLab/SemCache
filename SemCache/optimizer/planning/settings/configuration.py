"""Shared training/inference settings; teacher credentials stay in a separate file."""

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Literal, overload

from .documents import read_configuration, read_json

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "config/system/config.json"
TEACHER_CONFIG_DIR = ROOT / "config/teacher"
RESPONSE_CACHE_LIMITS = {
    "max_disk_bytes": 268435456,
    "max_memory_bytes": 16777216,
    "max_entries": 32768,
    "max_entry_bytes": 262144,
    "max_correction_retries": 5,
}


def activate_native(settings):
    """Select the configured binding before the first native module import."""
    from planning import kernels

    os.environ["SEMCACHE_RESPONSE_CACHE"] = json.dumps(
        response_cache_settings(settings)
    )
    path = str(runtime_path(settings, "native_bindings") / "planning/kernels")
    if path not in kernels.__path__:
        kernels.__path__.insert(0, path)


def response_cache_settings(settings=None):
    """Resolve shared response storage independently of the caller's cwd.

    Native sampling inherits this JSON through activate_native; direct SQL
    evaluators use the same settings explicitly. A null directory disables
    the transport result cache and its shared flights. Typed query-local reuse
    is separate and remains controlled by semantic_request_reuse.
    """
    if settings is None:
        inherited = os.environ.get("SEMCACHE_RESPONSE_CACHE")
        if inherited is not None:
            settings = {"response_cache": json.loads(inherited)}
        else:
            settings = load_config()
    section = settings.get("response_cache", {})
    if not isinstance(section, dict):
        raise ValueError("response_cache must be a JSON object")
    unknown = (
        section.keys()
        - RESPONSE_CACHE_LIMITS.keys()
        - {"directory", "verify_on_second_use"}
    )
    if unknown:
        raise ValueError(
            "Unknown response_cache settings: " + ", ".join(sorted(unknown))
        )
    directory = section.get("directory")
    if directory is not None and (
        not isinstance(directory, str) or not directory.strip()
    ):
        raise ValueError("response_cache.directory must be a directory path or null")
    result: dict[str, Any] = {
        "directory": str((ROOT / Path(directory).expanduser()).resolve())
        if directory is not None
        else None
    }
    verification = section.get("verify_on_second_use", True)
    if type(verification) is not bool:
        raise ValueError("response_cache.verify_on_second_use must be a boolean")
    result["verify_on_second_use"] = verification
    for key, default in RESPONSE_CACHE_LIMITS.items():
        value = section.get(key, default)
        lower = 0 if key in ("max_memory_bytes", "max_correction_retries") else 1
        if type(value) is not int or value < lower:
            requirement = (
                "a nonnegative integer" if lower == 0 else "a positive integer"
            )
            raise ValueError(f"response_cache.{key} must be {requirement}")
        result[key] = value
    if result["max_entry_bytes"] < 56:
        raise ValueError("response_cache.max_entry_bytes must be >= 56 (file header)")
    if result["max_correction_retries"] > 64:
        raise ValueError("response_cache.max_correction_retries must be in [0,64]")
    return result


def load_sampling_config(path, settings):
    """Allow an explicit override file or the assembled sampling section."""
    if path is None:
        return dict(settings["sampling"])
    document = read_json(path)
    return document.get("sampling", document)


def _validate_operational_settings(settings):
    """Validate shared non-search defaults before a launcher starts any work."""

    def integer(section, key, lower: int | None = 1):
        value = settings[section].get(key)
        if type(value) is not int or (lower is not None and value < lower):
            requirement = "an integer" if lower is None else f"an integer >= {lower}"
            raise ValueError(f"{section}.{key} must be {requirement}")
        return value

    def number(section, key, lower: float = 0, *, inclusive=False):
        value = settings[section].get(key)
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or (value < lower if inclusive else value <= lower)
        ):
            relation = ">=" if inclusive else ">"
            raise ValueError(f"{section}.{key} must be finite and {relation} {lower}")
        return value

    def boolean(section, key):
        if type(settings[section].get(key)) is not bool:
            raise ValueError(f"{section}.{key} must be a boolean")

    def path(section, key):
        value = settings[section].get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{section}.{key} must be a nonempty path")

    for key in ("python", "build", "native_bindings", "run_root"):
        path("runtime", key)
    number("runtime", "shutdown_timeout_seconds")
    training = settings["training"]
    integer("training", "pretrain_rounds")
    integer("training", "rounds", 0)
    for key in ("repeats", "epochs"):
        integer("training", key)
    integer("training", "seed", None)
    boolean("training", "collect_only")
    boolean("training", "evaluate_selected")
    number("training", "timeout")
    minimum = number("training", "minimum_timeout")
    if minimum > training["timeout"]:
        raise ValueError("training.minimum_timeout must not exceed training.timeout")
    number("training", "timeout_slack", 1, inclusive=True)
    fraction = training.get("validation_fraction")
    if type(fraction) not in (int, float) or fraction != 0:
        raise ValueError("training.validation_fraction must be 0 for closed workloads")
    families = training.get("train_families")
    if families is not None and (
        not isinstance(families, list)
        or not families
        or any(not isinstance(name, str) or not name.strip() for name in families)
        or len(set(families)) != len(families)
    ):
        raise ValueError("training.train_families must be distinct names or null")
    relations = training.get("max_relations")
    if relations is not None and (type(relations) is not int or relations < 1):
        raise ValueError("training.max_relations must be a positive integer or null")
    checkpoints = training.get("checkpoint_epochs")
    if (
        not isinstance(checkpoints, list)
        or any(type(epoch) is not int or epoch < 1 for epoch in checkpoints)
        or checkpoints != sorted(set(checkpoints))
    ):
        raise ValueError(
            "training.checkpoint_epochs must be increasing positive integers"
        )
    learning = settings["learning"]
    for policy_name in ("round_update", "final_update"):
        if policy_name not in training:
            continue
        from planning.learning.convergence import validate_update_policy

        validate_update_policy(training[policy_name], f"training.{policy_name}")
    if learning.get("algorithm") not in ("balsa_monte_carlo", "supervised"):
        raise ValueError("learning.algorithm must be balsa_monte_carlo or supervised")
    if (
        not isinstance(learning.get("device"), str)
        or re.fullmatch(r"[a-z][a-z0-9_]*(?::[0-9]+)?", learning["device"]) is None
        or learning["device"] == "auto"
    ):
        raise ValueError("learning.device must be an explicit PyTorch device")
    integer("learning", "batch_size")
    integer("learning", "patience", 0)
    boolean("learning", "balance_datasets")
    for key in (
        "learning_rate",
        "accuracy_weight",
        "censor_weight",
        "estimated_censor_weight",
        "complete_state_weight",
        "partial_state_weight",
        "time_scale",
    ):
        number("learning", key, 1e-12, inclusive=True)
    if number("learning", "warmup_fraction", 0, inclusive=True) >= 1:
        raise ValueError("learning.warmup_fraction must be in [0,1)")
    if number("learning", "discount", 0, inclusive=True) > 1:
        raise ValueError("learning.discount must be in [0,1]")
    width = integer("value_model", "width")
    heads = integer("value_model", "heads")
    integer("value_model", "layers")
    boolean("value_model", "use_cache_features")
    if width % heads:
        raise ValueError("value_model.width must be divisible by value_model.heads")
    if settings["inference"].get("plan_generation", "heuristic_requests") not in (
        "heuristic_requests",
        "heuristic_value",
    ):
        raise ValueError("Unknown inference.plan_generation policy")
    path("search", "fixed_model_id")
    if settings["search"].get("exploration_fraction", 0) != 0:
        raise ValueError("The two priority policies use exploration_fraction=0")
    if settings["inference"].get("preference") not in ("fastest", "accurate"):
        raise ValueError("inference.preference must be fastest or accurate")
    integer("calibration", "cohort_seed", None)
    for key in ("refresh_seconds", "health_interval", "stalled_after"):
        number("monitoring", key)
    number("monitoring", "stop_query_after", 0, inclusive=True)


def load_config(path=None):
    source = Path(path) if path is not None else DEFAULT_CONFIG
    settings = read_configuration(source)
    if not isinstance(settings, dict):
        raise ValueError("Optimizer configuration must be a JSON object")
    sections = (
        "runtime",
        "statistics",
        "models",
        "execution",
        "sampling",
        "calibration",
        "search",
        "training",
        "learning",
        "value_model",
        "inference",
        "monitoring",
        "reference_answers",
    )
    for name in sections:
        if not isinstance(settings.get(name), dict):
            raise ValueError(f"Configuration requires an object section: {name}")
    # Fixed policy metadata is excluded from editable configuration and supplied in loaded snapshots.
    settings["search"].setdefault("exploration_fraction", 0.0)
    settings["search"].setdefault("workers", 1)
    settings["training"].setdefault("validation_fraction", 0.0)
    settings["learning"].setdefault("algorithm", "balsa_monte_carlo")
    reference = settings["reference_answers"].get("semcache")
    if isinstance(reference, dict):
        reference.setdefault("planner", "heuristic_requests")
        reference.setdefault("layout", "GOTDF")
    _validate_operational_settings(settings)
    cpu_limit = settings["runtime"].get("max_cpu_threads")
    if type(cpu_limit) is not int or not 1 <= cpu_limit <= 1024:
        raise ValueError("runtime.max_cpu_threads must be an integer in [1,1024]")
    for section, key in (
        ("execution", "cpu_parallelism"),
        ("sampling", "cpu_parallelism"),
        ("learning", "threads"),
        ("search", "workers"),
    ):
        value = settings[section].get(key)
        if type(value) is not int or not 1 <= value <= cpu_limit:
            raise ValueError(
                f"{section}.{key} must be between 1 and runtime.max_cpu_threads"
            )
    inference = settings["inference"]
    if (
        not isinstance(inference.get("device"), str)
        or re.fullmatch(r"auto|cpu|cuda(?::[0-9]+)?", inference["device"]) is None
    ):
        raise ValueError("inference.device must be auto, cpu, cuda or cuda:<index>")
    if (
        type(inference.get("gpu_batch_size")) is not int
        or inference["gpu_batch_size"] < 1
    ):
        raise ValueError("inference.gpu_batch_size must be a positive integer")
    response_cache_settings(settings)
    training = settings["training"]
    if training.get("rounds", 0) and (
        not isinstance(training.get("reference_answers_dir"), str)
        or not training["reference_answers_dir"].strip()
    ):
        raise ValueError(
            "training.reference_answers_dir must name the saved teacher answers"
        )
    estimated_timeout = training.get("estimated_timeout")
    if (
        type(estimated_timeout) not in (int, float)
        or not math.isfinite(estimated_timeout)
        or estimated_timeout <= 0
    ):
        raise ValueError("training.estimated_timeout must be positive and finite")
    minimum_results = training.get("minimum_result_rows", 10)
    if type(minimum_results) is not int or minimum_results < 1:
        raise ValueError("training.minimum_result_rows must be a positive integer")
    minimum_inputs = training.get("minimum_distinct_semantic_inputs")
    if type(minimum_inputs) is not int or minimum_inputs < 1:
        raise ValueError(
            "training.minimum_distinct_semantic_inputs must be a positive integer"
        )
    manifest = training.get("workload_manifest")
    if not isinstance(manifest, str) or not manifest.strip():
        raise ValueError(
            "training.workload_manifest must name the frozen query manifest"
        )
    source_databases = settings["runtime"].get("source_databases")
    expected_sources = {"movies_nirvana", "movies_sema", "estate", "appstore", "tpch"}
    selected_dataset = training.get("dataset")
    if selected_dataset is not None and (
        not isinstance(selected_dataset, str)
        or selected_dataset not in expected_sources
    ):
        raise ValueError("training.dataset must be a configured dataset name or null")
    if "teacher_config" in training and (
        not isinstance(training["teacher_config"], str)
        or not training["teacher_config"].strip()
    ):
        raise ValueError("training.teacher_config must name the training teacher file")
    query_counts = training.get("queries_per_dataset")
    counts = (
        dict.fromkeys(expected_sources, query_counts)
        if type(query_counts) is int
        else query_counts
    )
    if (
        not isinstance(counts, dict)
        or set(counts) != expected_sources
        or any(type(count) is not int or count < 1 for count in counts.values())
    ):
        raise ValueError(
            "training.queries_per_dataset must give a positive count for each dataset"
        )
    if (
        not isinstance(source_databases, dict)
        or set(source_databases) != expected_sources
        or any(
            not isinstance(path, str) or not path.strip()
            for path in source_databases.values()
        )
    ):
        raise ValueError(
            "runtime.source_databases must map the five dataset names to database paths"
        )
    if len({(ROOT / path).resolve() for path in source_databases.values()}) != 5:
        raise ValueError("Each dataset must use a distinct source database file")
    obsolete_sampling = {
        "max_model_calls",
        "max_fallback_model_calls",
        "parallelism",
        "initial_size",
        "batch_size",
        "budget",
        "min_survivors",
        "accuracy_size",
        "max_strata",
        "min_depth_samples",
        "selectivity_batch_size",
        "selectivity_batch_max_bytes",
        "accuracy_batch_size",
        "accuracy_batch_max_bytes",
        "initial_source_rows",
        "source_refill_rows",
        "source_row_budget",
        "minimum_survivor_rows",
        "max_source_strata",
        "minimum_depth_evidence",
        "max_fallback_llm_requests_per_operator",
        "selectivity_estimator",
        "share_state_statistics",
        "relational_fallback_sampling",
    } & settings["sampling"].keys()
    if obsolete_sampling:
        raise ValueError(
            "Replace obsolete sampling keys with the names in SemCache/README.md: "
            + ", ".join(sorted(obsolete_sampling))
        )
    if (
        not isinstance(settings.get("teacher_config"), str)
        or not settings["teacher_config"].strip()
    ):
        raise ValueError("teacher_config must name a separate teacher JSON file")
    cycle = settings["training"].get("feedback_cycle")
    if (
        not isinstance(cycle, list)
        or not cycle
        or any(
            not isinstance(item, dict)
            or item.get("plan_generation")
            not in ("heuristic_requests", "heuristic_value", "value")
            or item.get("query_order") not in ("forward", "reverse", "shuffled")
            for item in cycle
        )
    ):
        raise ValueError(
            "training.feedback_cycle requires plan_generation="
            "heuristic_requests/heuristic_value/value "
            "and query_order=forward/reverse/shuffled"
        )
    fraction = settings["training"].get("exploration_fraction", 0.0)
    if (
        type(fraction) not in (int, float)
        or not math.isfinite(fraction)
        or not 0 <= fraction < 1
    ):
        raise ValueError("training.exploration_fraction must be in [0,1)")
    if (
        type(settings["training"].get("feedback_rollouts_per_query", 2)) is not int
        or settings["training"].get("feedback_rollouts_per_query", 2) != 2
    ):
        raise ValueError(
            "training.feedback_rollouts_per_query must be 2 (two searches and executions)"
        )
    retry = settings["training"].get("network_retry", {})
    if (
        not isinstance(retry, dict)
        or type(retry.get("max_retries", 2)) is not int
        or not 0 <= retry.get("max_retries", 2) <= 10
        or type(retry.get("backoff_seconds", 5)) not in (int, float)
        or not math.isfinite(retry.get("backoff_seconds", 5))
        or not 0 <= retry.get("backoff_seconds", 5) <= 30
    ):
        raise ValueError(
            "training.network_retry requires max_retries in [0,10] and backoff_seconds in [0,30]"
        )
    endpoints = settings["models"].get("endpoints")
    if (
        not isinstance(endpoints, list)
        or not endpoints
        or any(not isinstance(url, str) or not url for url in endpoints)
    ):
        raise ValueError("models.endpoints must contain API base URLs")
    limit = settings["models"].get("max_output_tokens")
    if type(limit) is not int or not 1 <= limit <= 262144:
        raise ValueError("max_output_tokens must be an integer in [1, 262144]")
    execution_limit = settings["execution"].get("max_concurrent_requests_per_model")
    if type(execution_limit) is not int or not 1 <= execution_limit <= 1024:
        raise ValueError(
            "execution.max_concurrent_requests_per_model must be an integer in [1, 1024]"
        )
    if type(settings["execution"].get("enforce_model_operator_order")) is not bool:
        raise ValueError("execution.enforce_model_operator_order must be a boolean")
    build_width = settings.get("statistics", {}).get("build_parallelism")
    if type(build_width) is not int or not 1 <= build_width <= 64:
        raise ValueError("statistics.build_parallelism must be an integer in [1,64]")
    from planning.settings.parallelism import sampling_parallelism

    settings["sampling"].setdefault("max_accuracy_batches_per_operator", 1)
    sampling_parallelism(settings["sampling"])
    if (
        type(settings["calibration"].get("sample_rows")) is not int
        or settings["calibration"]["sample_rows"] < 1
    ):
        raise ValueError("calibration.sample_rows must be a positive integer")
    timing = settings["calibration"].setdefault(
        "operator_time",
        {"sample_rows": 10, "relational_repeats": 3, "seed": 42},
    )
    if not isinstance(timing, dict):
        raise ValueError("calibration.operator_time must be an object")
    for key in ("sample_rows", "relational_repeats"):
        if type(timing.get(key)) is not int or timing[key] < 1:
            raise ValueError("calibration.operator_time." + key + " must be positive")
    if type(timing.get("seed")) is not int:
        raise ValueError("calibration.operator_time.seed must be an integer")
    cache_dir = settings["calibration"].get("cache_dir")
    if cache_dir is not None and (
        not isinstance(cache_dir, str) or not cache_dir.strip()
    ):
        raise ValueError("calibration.cache_dir must be a directory path or null")
    reference = settings["reference_answers"]
    semcache = reference.get("semcache")
    if not isinstance(semcache, dict):
        raise ValueError("reference_answers.semcache must be an object")
    if "verify_on_second_use" in semcache and type(semcache["verify_on_second_use"]) is not bool:
        raise ValueError("reference_answers.semcache.verify_on_second_use must be a boolean")
    if (
        semcache.get("planner") != "heuristic_requests"
        or semcache.get("layout") != "GOTDF"
    ):
        raise ValueError("SemCache references require heuristic_requests and GOTDF")
    for key, lower, upper in (
        ("parallelism", 1, 128),
        ("cpu_parallelism", 1, 64),
        ("max_output_tokens", 1, 4096),
    ):
        value = semcache.get(key)
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(
                f"reference_answers.semcache.{key} must be in [{lower},{upper}]"
            )
    if {"reasoning_effort", "thinking", "enable_thinking"} & semcache.keys():
        raise ValueError(
            "Configure teacher thinking and reasoning_effort only in "
            "config/teacher/teacher.*.json, not reference_answers.semcache"
        )
    limit = semcache.get("timeout_seconds")
    if (
        not isinstance(limit, (int, float))
        or isinstance(limit, bool)
        or not math.isfinite(limit)
        or limit <= 0
    ):
        raise ValueError("SemCache reference timeout must be positive and finite")
    unknown = reference.keys() - {"semcache"}
    if unknown:
        raise ValueError(
            "Unknown reference_answers settings: " + ", ".join(sorted(unknown))
        )
    return settings


@overload
def runtime_path(settings, name: Literal["source_databases"]) -> dict[str, Path]: ...


@overload
def runtime_path(
    settings, name: Literal["python", "build", "native_bindings", "run_root"]
) -> Path: ...


@overload
def runtime_path(settings, name: str) -> Path | dict[str, Path]: ...


def runtime_path(settings, name: str) -> Path | dict[str, Path]:
    """Runtime paths are relative to the SemCache root, independent of cwd."""
    value = settings["runtime"][name]
    if isinstance(value, dict):
        return {key: (ROOT / path).resolve() for key, path in value.items()}
    return (ROOT / value).resolve()


def teacher_path(settings, config=None):
    """Resolve teacher names in their directory; explicit paths stay explicit."""
    teacher = Path(settings["teacher_config"]).expanduser()
    if teacher.is_absolute():
        return teacher.resolve()
    if teacher.parent == Path("."):
        return (TEACHER_CONFIG_DIR / teacher).resolve()
    return (
        (Path(config) if config else DEFAULT_CONFIG).resolve().parent
        / teacher
    ).resolve()


def config_from_argv():
    """Read shared defaults before building command parsers; explicit flags win."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args, _ = parser.parse_known_args()
    settings = load_config(args.config)
    activate_native(settings)
    return args.config, settings
