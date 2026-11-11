"""Bound QueryIR stages -> shared beam sampling, using a DuckDB read snapshot."""

import copy
import math
import re
from contextlib import contextmanager

from planning.settings.parallelism import sampling_parallelism

from ..state import field_key
from .execution import SQLBackend


class NativeSamplingFactory:
    def __init__(
        self,
        build,
        models,
        *,
        database=None,
        setup="",
        config=None,
        history=None,
        teacher=None,
        catalog=None,
        calibration_config=None,
    ):
        from planning.settings.configuration import load_config

        settings = load_config()
        self.calibration_config = {
            **settings["calibration"]["operator_time"],
            "parallelism": settings["execution"]["cpu_parallelism"],
            "max_concurrent_requests_per_model": settings["execution"][
                "max_concurrent_requests_per_model"
            ],
            **(calibration_config or {}),
        }
        self.build, self.models, self.database = build, models, database
        self.catalog = catalog or {}
        self.setup, self.config = setup, sampling_parallelism(config)
        self.config["max_concurrent_requests_per_model"] = settings["execution"][
            "max_concurrent_requests_per_model"
        ]
        self.offline_store = None
        if database is not None and self.config.get("offline_statistics_dir"):
            from planning.observations.offline_statistics import OfflineStatisticsStore

            self.offline_store = OfflineStatisticsStore(
                self.config["offline_statistics_dir"], database
            )
        self.teacher = teacher
        from planning.observations.cache import CacheHistory

        self._owns_history = history is None
        self.history = (
            history
            if history is not None
            else CacheHistory(
                ":memory:", capacity=self.config.get("history_capacity", 256)
            )
        )

    def __call__(self, problem):
        from ..kernels.sampling import prepare_native

        return prepare_native(self._for_problem(problem), problem)

    def _for_problem(self, problem):
        """Resolve policy before opening transports or choosing sample models."""
        from planning.settings.modes import (
            DEFAULT_FIXED_MODEL,
            REQUESTS,
            sampling_for_mode,
            validate_mode,
        )

        mode = validate_mode(
            problem.profile.get(
                "plan_generation", self.config.get("plan_generation", REQUESTS)
            )
        )
        fixed = problem.profile.get(
            "fixed_model_id", self.config.get("fixed_model_id", DEFAULT_FIXED_MODEL)
        )
        configured = copy.copy(self)
        configured.calibration_config = dict(self.calibration_config)
        environment = problem.query.get("environment", {})
        for exported, setting in (
            ("duckdb_threads", "parallelism"),
            ("scheduler_max_inflight_per_model", "max_concurrent_requests_per_model"),
        ):
            value = environment.get(exported)
            if type(value) is int and value > 0:
                configured.calibration_config[setting] = value
        configured.config = sampling_parallelism(
            sampling_for_mode(self.config, mode, fixed_model_id=fixed)
        )
        configured.config["max_concurrent_requests_per_model"] = (
            configured.calibration_config["max_concurrent_requests_per_model"]
        )
        problem.profile.update(
            plan_generation=mode, sampling_strategy="independent_base"
        )
        if mode == REQUESTS:
            if fixed not in self.models:
                raise ValueError(f"Fixed sampling model is unavailable: {fixed}")
            configured.models = {fixed: self.models[fixed]}
            configured.catalog = (
                {fixed: self.catalog[fixed]} if fixed in self.catalog else {}
            )
        candidates = set(configured.models)
        for oid, op in problem.ops.items():
            names = [name for name in op["models"] if name in candidates]
            if not names:
                raise ValueError(f"No sampling model for semantic operator {oid}")
            op["models"] = names
            problem.all_ops[oid]["models"] = list(names)
        return configured

    def offline_statistics(self, problem):
        required = self.config.get("sample_depth_selectivity", True)
        if not required:
            return {}
        if self.offline_store is None:
            raise ValueError("Value-guided sampling requires an offline statistics store")
        return self.offline_store.for_query(problem.query, required=True)

    def reference(self, problem):
        return self._for_problem(problem)._reference(problem)

    @contextmanager
    def _reference(self, problem):
        """Explicit differential reference; production uses native sampling."""
        from planning.execution.feedback import SampleCalibration

        from ..sampling import SamplingCoordinator

        transport_models = (
            self.teacher.transport_models(self.models) if self.teacher else self.models
        )
        types = prepare_stages(problem, transport_models, sampling=self.config)
        calibration = SampleCalibration(problem.profile)
        with SQLBackend(
            self.build,
            transport_models,
            database=self.database,
            setup=self.setup,
            field_types=types,
            collect_telemetry=True,
            history_sink=self.history,
            observation_sink=calibration.observe,
            adaptive_sketch=self.config.get("adaptive_sketch"),
            parallelism=self.config["parallelism"],
            cpu_workers=self.config["cpu_workers"],
        ) as backend:
            from .direct_sampling import load_direct_sources

            sources, populations, direct_sources = load_direct_sources(
                problem, self.config, backend.load
            )
            offline_statistics = self.offline_statistics(problem)
            for relation in populations:
                if str(relation) in offline_statistics:
                    populations[relation] = offline_statistics[str(relation)][
                        "cooccurrence"
                    ]["row_count"]
            problem.history_provider = self.history.snapshot
            coordinator = SamplingCoordinator(
                sources,
                backend,
                population_counts=populations,
                direct_sources=direct_sources,
                **{
                    k: v
                    for k, v in self.config.items()
                    if k
                    not in {
                        "cpu_workers",
                        "cpu_parallelism",
                        "parallelism",
                        "relational_fallback_sampling",
                        "selectivity_batch_size",
                        "accuracy_batch_size",
                        "accuracy_batch_max_bytes",
                        "max_fallback_model_calls",
                        "offline_statistics_dir",
                        "selectivity_batch_max_bytes",
                        "plan_generation",
                        "fixed_model_id",
                        "trace_timings",
                        "collect_calibration",
                        "relational_cardinality_sample_rows",
                        "max_concurrent_requests_per_model",
                    }
                },
            )
            coordinator.offline_statistics = offline_statistics
            from planning.observations.cache import PrefixSummary

            for relation, offline in coordinator.offline_statistics.items():
                rid = next(r for r in sources if str(r) == relation)
                for name, summary in offline["summaries"].items():
                    coordinator.population_summaries[(rid, name)] = PrefixSummary.load(
                        summary
                    )
            problem.profile["adaptive_sketch"] = coordinator.sketch_config
            problem.profile["base_sketch_plans"] = coordinator.population_sketch_plans
            yield coordinator

    def close(self):
        if self._owns_history:
            self.history.close()


def _validate_estimate_only_joins(problem):
    left_joins = {
        node_id
        for node_id, node in problem.logical.items()
        if node.get("join_type") == "LEFT"
    }
    if not left_joins:
        return
    if problem.profile.get("plan_generation") != "heuristic_requests":
        raise ValueError(
            "LEFT JOIN online sampling supports only heuristic_requests cardinality "
            "estimates; value sampling with NULL-extended identities is unsupported"
        )
    for op in problem.ops.values():
        pending = list(problem.logical[op["node_id"]].get("children", []))
        visited = set()
        while pending:
            node_id = pending.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            if node_id in left_joins:
                raise ValueError(
                    "LEFT JOIN output values cannot feed online semantic sampling; "
                    "NULL-extended sample identities are unsupported"
                )
            pending.extend(problem.logical[node_id].get("children", []))


def _prepare_correlated_terminal_estimates(problem):
    """Keep unused correlated tails as priors, never as sampled value evidence.

    DELIM_GET is a correlated input supplied by DuckDB at execution time, not
    an independently readable table. Only request-count planning may leave its
    terminal relational consumers unmaterialized. Any semantic consumer still
    requires a real bound population and is rejected by the normal contracts.
    """
    logical = problem.logical
    if not any(node.get("type") == "DELIM_GET" for node in logical.values()):
        return
    if problem.profile.get("plan_generation") != "heuristic_requests":
        raise ValueError(
            "Correlated online sampling supports only terminal cardinality estimates in heuristic_requests"
        )
    parents = {nid: set() for nid in logical}
    for nid, node in logical.items():
        for child in node.get("children", []):
            parents[child].add(nid)

    def closure(start, edges):
        reached, pending = set(), list(start)
        while pending:
            nid = pending.pop()
            if nid not in reached:
                reached.add(nid)
                pending.extend(edges(nid))
        return reached

    virtual = {}
    terminal = set()
    for nid, node in logical.items():
        if node.get("type") != "DELIM_GET":
            continue
        consumers = closure([nid], lambda current: parents[current])
        if any(
            logical[current].get("type", "").startswith("SEMANTIC_")
            for current in consumers
        ):
            raise ValueError(
                "Correlated sampling values for a semantic consumer are unsupported"
            )
        owners = []
        for owner in consumers:
            candidate = logical[owner]
            children = candidate.get("children", [])
            if candidate.get("type") != "DELIM_JOIN" or len(children) != 2:
                continue
            right = closure(
                [children[1]], lambda current: logical[current].get("children", [])
            )
            if nid in right:
                owners.append(owner)
        # Nested correlation needs exported ownership, not an inferred alias.
        if len(owners) != 1:
            raise ValueError(
                "Correlated sampling input has no unambiguous DELIM_JOIN owner"
            )
        owner = logical[owners[0]]
        left = logical[owner["children"][0]]
        population = left.get("estimated_rows")
        if (
            type(population) not in (int, float)
            or not math.isfinite(population)
            or population < 0
        ):
            raise ValueError(
                "Correlated sampling domain has no finite left cardinality prior"
            )
        virtual[str(nid)] = {
            "estimated_rows": population,
            "owner_node_id": owners[0],
            "cardinality_basis": "correlated input domain upper bound from DuckDB left cardinality; no sampled values",
        }
        terminal |= consumers
    if not virtual:
        return
    problem.profile["cardinality_only_sources"] = virtual
    for nid, prior in virtual.items():
        problem.profile.setdefault("node_statistics", {}).setdefault(nid, {}).update(
            output_rows=prior["estimated_rows"],
            cardinality_basis=prior["cardinality_basis"],
            sampling_cardinality_only=True,
        )
    for oid, op in problem.all_ops.items():
        if op["node_id"] not in terminal:
            continue
        op["sampling_cardinality_only"] = True
        node = logical[op["node_id"]]
        node["sampling_cardinality_only"] = True
        if node.get("type") == "DELIM_JOIN":
            join_type = node.get("parameters", {}).get("Join Type")
            if join_type != "MARK":
                raise ValueError(
                    "Cardinality-only correlated planning requires a MARK join"
                )
            op["join_type"] = node["join_type"] = join_type


def prepare_stages(problem, models, *, sampling=None):
    """Shared bound-stage contracts for resident and reference sampling."""
    from .population_samples import population_probe_sql, prepare_population_samples
    from .relational_estimates import prepare_population_estimates

    _validate_estimate_only_joins(problem)
    _prepare_correlated_terminal_estimates(problem)
    prepare_population_estimates(problem)
    for name, config in models.items():
        # External teacher endpoints are registered in the transport registry,
        # but are never candidate sampling models or catalog entries.
        if name not in problem.catalog and name in problem.profile.get(
            "teacher_models", {}
        ):
            continue
        info = problem.catalog.setdefault(name, {})
        match = re.search(
            r"(\d+(?:\.\d+)?)B(?:\b|[-_])", config["model"], re.IGNORECASE
        )
        if match and not info.get("parameters"):
            info["parameters"] = float(match[1]) * 1e9
    problem.profile["models"] = problem.catalog
    types = {
        field_key(field): field["type"]
        for node in problem.logical.values()
        for field in node.get("output_schema", [])
    }
    for oid, op in problem.all_ops.items():
        template = op.get("sampling_sql")
        if not template and not op.get("sampling_cardinality_only"):
            raise ValueError(
                f"Online sampling stage {op['node_id']} has no bound SQL template"
            )
        if template:
            op["sql"] = template
        op.setdefault("kind", op.get("type"))
        if op.get("pair_targets") and template and "$INPUT" in template:
            predicate = template.split(" WHERE ", 1)[1]
            op["candidate_sql"] = (
                "SELECT l.__sample_id __left_sample_id,r.__sample_id __right_sample_id,l.* EXCLUDE (__sample_id),r.* EXCLUDE (__sample_id) FROM $LEFT l JOIN $RIGHT r ON "
                + predicate
            )
        if oid in problem.ops:
            for field in op["input_field_schema"]:
                field.setdefault("value_key", f"__semantic_input:{oid}:{field['name']}")
                if field["value_key"].startswith("__semantic_input:"):
                    types[field["value_key"]] = field["type"]
            problem.ops[oid].update(copy.deepcopy(op))
    prepare_population_samples(problem)
    sampling = sampling or {}
    for op in problem.all_ops.values():
        if "population_sample" in op:
            spec = op["population_sample"]
            spec["probe_sql"] = population_probe_sql(
                spec, sampling.get("relational_cardinality_sample_rows", 2000),
                sampling.get("seed", 42) + op["node_id"] * 104729,
            )
    return types
