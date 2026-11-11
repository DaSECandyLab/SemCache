"""Independent startup probes: parallel across models, serial within a model."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager
from urllib.parse import urlsplit

from planning.backends.execution import SQLBackend
from planning.prompt_layouts import BLOCK_ORDERS
from planning.settings.parallelism import positive_parallelism


@contextmanager
def calibration_requests(build, models, parallelism):
    """Yield cells with their telemetry; the caller alone publishes history.

    These generic probes already have fully materialized inputs and need no
    source catalog. Each model owns an in-memory database, stage socket and
    telemetry file. Aliases of the same endpoint/model share a serial lane.
    """
    lanes = {}
    for model, endpoint in models.items():
        address = urlsplit(endpoint["base_url"])
        path = address.path[:-1] if address.path.endswith("/") else address.path
        key = (
            address.scheme,
            address.hostname,
            address.port or (443 if address.scheme == "https" else 80),
            path,
            endpoint["model"],
        )
        lanes.setdefault(key, []).append(model)
    # `parallelism` limits the CPU work used by the surrounding SQL backend.
    # Calibration measures atomic request latency: one serial lane per physical
    # model, with different lanes allowed to overlap. The execution request
    # window is independent of these startup probes.
    positive_parallelism(parallelism)
    limit = max(1, len(lanes))
    with ExitStack() as resources:
        executor = resources.enter_context(ThreadPoolExecutor(max_workers=limit))
        backends = {}

        def run_lane(aliases, op, records, layout):
            result = []
            for model in aliases:
                backend = backends[model]
                backend.reset_observations()
                rows = backend(op, (records,), model, layout)
                result.append((model, rows, list(backend.events)))
            return result

        def measure(op, records):
            # Create backends on the owner thread. Reuse them for every kind
            # and layout; cached restarts never call this function.
            for model, endpoint in models.items():
                if model not in backends:
                    backends[model] = resources.enter_context(
                        SQLBackend(
                            build,
                            {model: endpoint},
                            collect_telemetry=True,
                            field_types={"quality_result": "VARCHAR"},
                            parallelism=1,
                            # These probes measure fresh model latency; reusing
                            # a stored response cannot supply that observation.
                            setup="SET semantic_response_cache='{}';",
                        )
                    )
            for order in BLOCK_ORDERS:
                layout = {**op["default_layout"], "block_order": list(order)}
                futures = [
                    executor.submit(run_lane, aliases, op, records, layout)
                    for aliases in lanes.values()
                ]
                for future in as_completed(futures):
                    for model, rows, events in future.result():
                        yield ":".join(order), model, rows, events
                # A layout barrier also prevents overlapping the next request
                # sequence of the same model, regardless of executor scheduling.

        try:
            yield measure
        finally:
            # Drain workers before their SQL connections and sockets close,
            # including errors raised during caller-side event processing.
            executor.shutdown(wait=True, cancel_futures=True)
