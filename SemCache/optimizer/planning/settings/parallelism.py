"""CPU resource caps; independent HTTP model lanes do not use a global cap."""

import os
from pathlib import Path


def configure_cpu_resources(settings, *, configure_torch=False):
    """Limit only this process and its children, retaining unrestricted HTTP IO.

    CPU affinity bounds actual CPU capacity even when independent transports
    need more OS threads. Existing library threads receive the same mask.
    Call at startup, before creating query clients or doing model inference.
    """
    limit = positive_parallelism(settings["runtime"]["max_cpu_threads"])
    cpus = sorted(os.sched_getaffinity(0))[:limit]
    for task in Path("/proc/self/task").iterdir():
        try:
            os.sched_setaffinity(int(task.name), cpus)
        except ProcessLookupError:
            pass
    count = len(cpus)
    for name in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(count)
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
    result = {"max_cpu_threads": limit, "cpu_affinity": cpus}
    if configure_torch:
        import torch

        torch.set_num_threads(min(settings["learning"]["threads"], count))
        # Set once before inter-op work starts. A single inter-op executor
        # avoids multiplying the configured intra-op worker pool.
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        result.update(
            torch_threads=torch.get_num_threads(),
            torch_interop_threads=torch.get_num_interop_threads(),
        )
    return result


def positive_parallelism(value):
    if type(value) is not int or not 1 <= value <= 1024:
        raise ValueError("parallelism must be an integer between 1 and 1024")
    return value


def sampling_parallelism(config=None, *, override=None):
    settings = dict(config or {})
    obsolete_accuracy = {
        "max_accuracy_llm_requests",
        "max_accuracy_model_calls",
    } & settings.keys()
    if obsolete_accuracy:
        raise ValueError(
            "Replace obsolete accuracy HTTP limits "
            + ", ".join(sorted(obsolete_accuracy))
            + " with max_accuracy_batches_per_operator (positive batch count; default 1)"
        )
    batches = settings.setdefault("max_accuracy_batches_per_operator", 1)
    if type(batches) is not int or batches < 1:
        raise ValueError("max_accuracy_batches_per_operator must be a positive integer")
    rows = settings.get("operator_sample_rows", settings.get("initial_size", 20))
    if type(rows) is not int or rows < 1:
        raise ValueError("operator_sample_rows must be a positive integer")
    maximum = settings.setdefault("max_operator_sample_rows", 50)
    if type(maximum) is not int or maximum < rows:
        raise ValueError(
            "max_operator_sample_rows must be an integer >= operator_sample_rows"
        )
    relational_rows = settings.get("relational_sample_rows", 200)
    if type(relational_rows) is not int or relational_rows < 1:
        raise ValueError("relational_sample_rows must be a positive integer")
    cardinality_rows = settings.setdefault("relational_cardinality_sample_rows", 2000)
    if type(cardinality_rows) is not int or cardinality_rows < 1:
        raise ValueError("relational_cardinality_sample_rows must be a positive integer")
    source_rows = max(rows, relational_rows)
    settings.setdefault("ensure_prefix_depth_rates", True)
    for key in (
        "sample_depth_selectivity",
        "sample_accuracy",
        "ensure_prefix_depth_rates",
        "selectivity_parallel_models",
    ):
        if key in settings and type(settings[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    # Sampling is always one record/pair per prompt. Reject stale packing
    # configurations instead of silently changing their requested behavior.
    for name in (
        "selectivity_prompt_batch_size",
        "accuracy_prompt_batch_size",
        "selectivity_batch_size",
        "accuracy_batch_size",
    ):
        if name in settings and settings.pop(name) != 1:
            raise ValueError(
                "Sampling prompts contain exactly one record/pair; remove " + name
            )
    for name in (
        "selectivity_prompt_max_bytes",
        "accuracy_prompt_max_bytes",
        "selectivity_batch_max_bytes",
        "accuracy_batch_max_bytes",
    ):
        settings.pop(name, None)
    # Public names describe the unit being bounded. The native sampler keeps
    # its shorter internal names after this one small normalization step.
    for public, internal in {
        "max_llm_requests": "max_model_calls",
        "max_fallback_llm_requests_per_operator": "max_fallback_model_calls",
        "accuracy_sample_units": "accuracy_size",
    }.items():
        if public in settings:
            settings[internal] = settings.pop(public)
    # Each operator draws its own bounded base cohort. Refill never conditions
    # on survivors; a binary operator samples pairs, not rows squared.
    settings.update(
        operator_sample_rows=rows,
        relational_sample_rows=relational_rows,
        initial_size=source_rows,
        batch_size=source_rows,
        budget=source_rows,
        join_pair_budget=rows,
        min_survivors=1,
        max_strata=1,
        min_depth_samples=min(4, rows),
    )
    limit = positive_parallelism(
        settings.get("cpu_parallelism", 1) if override is None else override
    )
    workers = positive_parallelism(settings.get("cpu_workers", min(limit, 64)))
    if workers > 64:
        raise ValueError("cpu_workers must be between 1 and 64")
    # `parallelism` is the native sampler's internal transport field. The
    # public configuration name is explicit about this being CPU parallelism.
    settings.update(
        cpu_parallelism=limit, parallelism=limit, cpu_workers=min(workers, limit)
    )
    return settings


def parallelism_argument(value):
    return positive_parallelism(int(value))
