"""Configure and own resident sampling sessions, history, and inspection views."""

import math
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from ..sampling.rows import SampleRow
from ..state import canonical
from .materialization import materialize_state


def sample_row(row):
    return {
        "id": row.identity,
        "values": dict(row.values),
        "lineage": [list(pair) for pair in row.lineage],
    }


def build_sampler_config(problem, *, sources, populations, sampling, history):
    """Capture one isolated ingress; resident sampling never calls this adapter."""
    operators = []
    for oid, op in problem.all_ops.items():
        operators.append(dict(op, operator_id=oid))
    config = {
        "query": problem.query,
        "profile": problem.profile,
        "operators": operators,
        "sources": {
            str(relation): [
                sample_row(row) if isinstance(row, SampleRow) else row for row in rows
            ]
            for relation, rows in sources.items()
        },
        "populations": {
            str(relation): count for relation, count in populations.items()
        },
        "sampling": sampling,
        "history": history,
        "fields": list(problem._native_owner.fields),
    }
    from planning.settings.configuration import response_cache_settings

    config["sampling_epoch"] = time.time_ns()
    config["prompt_answers_enabled"] = (
        response_cache_settings()["directory"] is not None
    )
    return config


def prepare_profile(problem):
    """Import frozen priors once for native search without online sampling."""
    from . import _search_core

    problem.native_problem
    config = build_sampler_config(
        problem, sources={}, populations={}, sampling={}, history=problem.cache_history
    )
    config["initial_statistics"] = problem.initial.statistics
    environment = _search_core.ProfileEnvironment(config)
    return NativeSamplingProxy(environment, None, config)


def apply_context(problem, profile, history):
    """Publish native observations without discarding Python model metadata."""
    previous_models = problem.profile.get("models", {})
    exported_models = profile.get("models")
    if isinstance(exported_models, dict):
        profile = dict(profile)
        profile["models"] = {
            name: {**previous_models.get(name, {}), **entry}
            for name, entry in exported_models.items()
        }
    problem.profile.update(profile)
    problem.catalog = problem.profile.get("models", {})
    problem.constraints = problem.profile.get("constraints", {})
    problem.cache_history = tuple(history)


class NativeBackendView:
    """Inspect archived events after the sampler drains its native backend."""

    def __init__(self, sampler, backend):
        self._sampler, self._backend = sampler, backend
        self._archive = None

    def __getattr__(self, name):
        return getattr(self._backend, name)

    @property
    def events(self):
        return (
            self._archive["events"]
            if self._archive is not None
            else self._sampler.config()["events"]
        )

    @property
    def observations(self):
        return (
            self._archive["observations"]
            if self._archive is not None
            else self._sampler.config()["observations"]
        )

    def archive(self):
        if self._archive is None and self._sampler is not None:
            snapshot = (
                self._sampler.telemetry()
                if hasattr(self._sampler, "telemetry")
                else self._sampler.config()
            )
            self._archive = {
                "events": snapshot.get("events", []),
                "observations": snapshot.get("observations", []),
            }
            self._sampler = None


class NativeSamplingProxy:
    """Own statistics preparation and expose the frozen search environment."""

    def __init__(self, native_sampler, backend, config):
        self.native_sampler = native_sampler
        self.backend = NativeBackendView(native_sampler, backend)
        self.sketch_config = dict(config.get("sampling", {}).get("adaptive_sketch", {}))
        self.timings = SimpleNamespace(values={})
        self._config = {"sampling": dict(config.get("sampling", {}))}

    @property
    def calls(self):
        return self.native_sampler.calls()

    @property
    def population_sketch_plans(self):
        return self.native_sampler.profile().get("base_sketch_plans", {})

    @property
    def admitted(self):
        if hasattr(self.native_sampler, "config"):
            return self.native_sampler.config().get("admitted", {})
        return {}

    def request_latency(self, *, start_ns=None, end_ns=None):
        events = self.backend.events
        values, intervals = [], []
        model_requests = Counter()
        for event in events:
            if event.get("event") in ("sampling_llm_submit", "sampling_llm_retransmit"):
                model_requests[event.get("model_id", "?")] += 1
            if event.get("event") != "semantic_row":
                continue
            lo, hi = event.get("submit_ns"), event.get("response_ns")
            if type(lo) is int and type(hi) is int and hi >= lo:
                lo = max(lo, start_ns) if start_ns is not None else lo
                hi = min(hi, end_ns) if end_ns is not None else hi
                if hi <= lo:
                    continue
                intervals.append((lo, hi))
                values.append((hi - lo) / 1e9)
            elif start_ns is None and end_ns is None:
                value = event.get("latency_ms")
                if type(value) in (int, float) and math.isfinite(value) and value >= 0:
                    values.append(value / 1000)
        merged = []
        for lo, hi in sorted(intervals):
            if merged and lo <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        return {
            "reported_requests": len(values),
            "seconds": sum(values) if values else None,
            "wall_seconds": sum(hi - lo for lo, hi in merged) / 1e9,
            "source": "semantic_row request intervals; seconds sum overlaps, wall_seconds unions overlaps",
            "http_requests_by_model": dict(model_requests),
        }

    def sync_profile(self, problem, exported=None):
        profile = (
            self.native_sampler.profile() if exported is None else exported["profile"]
        )
        history = (
            self.native_sampler.history() if exported is None else exported["history"]
        )
        apply_context(problem, profile, history)

    def state(self, problem, handle):
        self.sync_profile(problem)
        return materialize_state(
            problem,
            {
                "state": handle,
                "statistics": self.native_sampler.full_statistics(handle),
            },
        )

    def selectivity(self, problem, state):
        self.native_sampler.observe(state.native_handle)
        return self.state(problem, state.native_handle)

    def accuracy(self, problem, state):
        self.native_sampler.accuracy(state.native_handle)
        return self.state(problem, state.native_handle)

    def finish(self, problem, raw):
        self.sync_profile(problem, raw)
        if "_environment" in raw:
            self.backend.archive()
            self.native_sampler = raw["_environment"]
        self.export_timings = raw.get("export_timings", {})
        self.state_diagnostics = raw.get("state_diagnostics", [])
        self.driver_timings = {
            name: {
                "wall_seconds": value[0],
                "calls": int(value[1]),
                **(
                    {"spans_ns": raw["timing_spans"][name]}
                    if name in raw.get("timing_spans", {})
                    else {}
                ),
            }
            for name, value in raw.get("timings", {}).items()
        }
        self.operation_timings = {
            name: {
                "wall_seconds": value["seconds"],
                "calls": value["calls"],
                **({"spans_ns": value["spans_ns"]} if "spans_ns" in value else {}),
            }
            for name, value in raw.get("sampling_timings", {}).items()
        }
        self.timings.values = {**self.driver_timings, **self.operation_timings}


def persist_history(history, native_sampler):
    """Publish the resident request ledger once without re-counting requests."""
    exported = native_sampler.config()
    entries = native_sampler.history()
    loads = exported["model_loads"]
    now = time.time()
    records = [
        (
            entry["id"],
            entry["model_id"],
            entry["pre"],
            str(entry["operator"]),
            entry["first_field"],
            entry.get("started_at", now - entry.get("age_seconds", 0)),
            canonical(entry["summary"]),
            entry.get("requests_before"),
            entry.get("characters_before"),
        )
        for entry in entries
    ]
    with history.lock, history.db:
        history.db.execute("DELETE FROM history")
        history.db.executemany(
            "INSERT INTO history VALUES (?,?,?,?,?,?,?,?,?)", records
        )
        history.db.executemany(
            "INSERT OR REPLACE INTO model_load VALUES (?,?,?)",
            [(model, value[0], value[1]) for model, value in loads.items()],
        )


@contextmanager
def prepare_native(factory, problem):
    """Open the pinned native backend and capture static configuration once."""
    from . import _search_core

    problem.native_problem
    from ..backends.direct_sampling import load_direct_sources
    from ..backends.sampling import prepare_stages

    transport_models = (
        factory.teacher.transport_models(factory.models)
        if factory.teacher
        else factory.models
    )
    # Keep the frozen training catalog (parameter counts/context limits) in the
    # profile sent to C++; transport endpoint settings remain separate.
    if factory.catalog:
        problem.profile["models"] = {
            name: {
                **factory.catalog.get(name, {}),
                **problem.profile.get("models", {}).get(name, {}),
            }
            for name in factory.catalog
        }
    if factory.teacher:
        factory.teacher.apply_profile(problem.profile)
    types = prepare_stages(problem, transport_models, sampling=factory.config)
    backend = _search_core.NativeSQLBackend(
        {
            "build": str(Path(factory.build).resolve()),
            "models": transport_models,
            "database": str(Path(factory.database).resolve())
            if factory.database is not None
            else None,
            "setup": factory.setup,
            "parallelism": factory.config["parallelism"],
            "cpu_workers": factory.config["cpu_workers"],
            # Beam selection/accuracy observations are independent. Keep the
            # normal SQL execution lane policy separate from this backend.
            "semantic_unrestricted": True,
            "field_types": types,
            "telemetry": True,
            "adaptive": factory.config.get("adaptive_sketch"),
            "trace_timings": factory.config.get("trace_timings", False),
        }
    )
    sampler = None
    previous_history_provider = problem.history_provider
    try:
        sources, populations, direct_sources = load_direct_sources(
            problem, factory.config, lambda rid, sql: backend.load(str(rid), sql)
        )
        offline_statistics = factory.offline_statistics(problem)
        for relation in populations:
            if str(relation) in offline_statistics:
                populations[relation] = offline_statistics[str(relation)][
                    "cooccurrence"
                ]["row_count"]
        ingress_history = factory.history.snapshot()
        captured_now = time.time()
        for entry in ingress_history:
            entry["started_at"] = captured_now - entry.get("age_seconds", 0)
        config = build_sampler_config(
            problem,
            sources=sources,
            populations=populations,
            sampling=factory.config,
            history=ingress_history,
        )
        config["operator_time_calibration"] = factory.calibration_config
        # Rows are serialized once in sources. Cohorts carry only stable IDs,
        # avoiding another copy of every long text cell in the native ingress.
        config["direct_sources"] = {
            relation: {
                limit: [row["id"] for row in rows] for limit, rows in cohorts.items()
            }
            for relation, cohorts in direct_sources.items()
        }
        config["model_loads"] = {
            name: list(factory.history.load(name)) for name in problem.catalog
        }
        config["offline_statistics"] = offline_statistics
        config["history_capacity"] = factory.history.capacity
        if "history_now" in factory.config:
            config["history_now"] = factory.config["history_now"]
        sampler = _search_core.NativeSampler(config, backend)
        proxy = NativeSamplingProxy(sampler, backend, config)
        # contextmanager generators keep locals alive across yield. The native
        # constructor has copied this ingress; release Python's source banks.
        del config, sources, direct_sources, ingress_history, offline_statistics
        problem.history_provider = sampler.history
        proxy.sync_profile(problem)
        yield proxy
    finally:
        problem.history_provider = previous_history_provider
        try:
            if sampler is not None:
                persist_history(factory.history, sampler)
        finally:
            backend.close()
