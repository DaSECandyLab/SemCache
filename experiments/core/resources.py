"""Resource settings for the current comparison, separate from model/plan choices."""

from __future__ import annotations

CONCURRENCY = 16
BATCH_SIZE = 4
PROFILE = "native-workers16-batch4-v1"
GATEWAY_REQUEST_LIMIT = 32


def job_resources(batch_size=BATCH_SIZE, *, request_concurrency=CONCURRENCY):
    if type(batch_size) is not int or not 1 <= batch_size <= 64:
        raise ValueError("batch_size must be an integer in [1,64]")
    if type(request_concurrency) is not int or request_concurrency < 1:
        raise ValueError("request_concurrency must be a positive integer")
    return {
        "resource_profile": f"native-workers{request_concurrency}-cpu{CONCURRENCY}-batch{batch_size}-v2",
        "native_concurrency": request_concurrency,
        "max_workers": request_concurrency,
        "join_parallelism": request_concurrency,
        "duckdb_threads": CONCURRENCY,
        "batch_size": batch_size,
    }


def native_settings(system, batch_size=BATCH_SIZE, *, request_concurrency=CONCURRENCY, abacus_queue_batch_size=None):
    """Record units explicitly: a prompt batch is not a worker count."""
    settings = {
        "plop": {
            "NUM_LLM_WORKERS": CONCURRENCY,
            "threads": CONCURRENCY,
            "prompt_rows_per_request": 1,
            "native_batch_size": None,
        },
        "sema": {
            "threads": CONCURRENCY,
            "semantic_batch_size": batch_size,
            "semantic_filter_batch_size": batch_size,
            "llm_rate_limit": 0,
            "rate_limit_unit": "requests/second; 0=unlimited",
            "native_http_concurrency_option": None,
        },
        "abacus": {
            "max_workers": request_concurrency,
            "join_parallelism": request_concurrency,
            "batch_size": abacus_queue_batch_size if abacus_queue_batch_size is not None else batch_size,
            "batch_unit": "operator queue admission",
            "prompt_rows_per_request": 1,
        },
        "nirvana": {
            "rate_limit": request_concurrency,
            "batch_size_for_join": batch_size,
            "map_filter_native_batch_size": None,
            "prompt_rows_per_request": 1,
        },
        "lotus": {
            "max_batch_size": request_concurrency,
            "parallel_groupby_max_threads": CONCURRENCY,
            "rate_limit": None,
            "rate_limit_unit": "requests/minute; None=unlimited",
            "max_batch_size_unit": "concurrent independent HTTP requests",
            "prompt_rows_per_request": 1,
        },
        "moar": {
            "max_threads": request_concurrency,
            "max_concurrent_agents": 3,
            "prompt_rows_per_request": 1,
            "native_batch_size": 1,
        },
    }
    return dict(settings[system])


def resource_snapshot(
    system, batch_size=BATCH_SIZE, *, gateway_request_limit=GATEWAY_REQUEST_LIMIT,
    request_concurrency=CONCURRENCY, abacus_queue_batch_size=None,
):
    return {
        **job_resources(batch_size, request_concurrency=request_concurrency),
        "system": system,
        "native": native_settings(system, batch_size, request_concurrency=request_concurrency,
                                  abacus_queue_batch_size=abacus_queue_batch_size),
        "gateway_concurrency_limit": gateway_request_limit,
        "concurrency_scope": "native workers plus a gateway cap shared across all models in this configuration",
        "batch_scope": "native options only; unsupported options remain unset",
    }


def configure_open_file_limit():
    """Allow native client connections without changing model concurrency."""
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = 65536 if hard == resource.RLIM_INFINITY else min(65536, hard)
    if soft != resource.RLIM_INFINITY and soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    return {
        "open_files_soft": resource.getrlimit(resource.RLIMIT_NOFILE)[0],
        "open_files_hard": hard,
    }
