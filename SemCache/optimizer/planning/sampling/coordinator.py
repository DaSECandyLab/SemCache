"""Independent base-sample selectivities and projection-only input preparation."""

import math
import random
from collections import Counter, OrderedDict, defaultdict
from dataclasses import replace
from typing import Any

from planning.execution.timing import PhaseTimings
from planning.observations.evidence import freeze_evidence, is_frozen_evidence

from ..kernels import observations as native_observations
from ..state import action_key, canonical, copy_statistics, digest, task_key
from .rows import CartesianRows, ProductFieldValues, SampleRow, lineages


def _publish_observation(previous, evidence):
    # A fresh observation replaces derived estimates, including a DuckDB-only
    # prior published before any model answer arrived. Never carry that prior
    # into a measured cohort just because its provenance keys are absent there.
    derived = {
        "conditional_observation",
        "estimated_selectivity",
        "selectivity_fusion",
        "survival_estimated",
        "survival_estimation_basis",
    }
    result = copy_statistics(
        {
            "node": {
                key: value
                for key, value in previous.items()
                if key not in evidence and key not in derived
            }
        }
    )["node"]
    # New observation tables are already cloned by mutable_statistics; copying
    # them a second time adds no isolation. Only old branch annotations need a copy.
    result.update(native_observations.mutable_statistics(evidence))
    return result


class SamplingCoordinator:
    def __init__(
        self,
        sources,
        backend,
        *,
        initial_size=16,
        batch_size=16,
        budget=128,
        min_survivors=8,
        accuracy_size=8,
        seed=42,
        max_strata=4,
        depth_targets=4,
        min_depth_samples=6,
        depth_error=0.15,
        max_model_calls=512,
        max_accuracy_batches_per_operator=1,
        population_counts=None,
        adaptive_sketch=None,
        selectivity_estimator="mask",
        operator_sample_rows=20,
        max_operator_sample_rows=50,
        selectivity_parallel_models=True,
        relational_sample_rows=200,
        sample_depth_selectivity=True,
        ensure_prefix_depth_rates=True,
        sample_accuracy=True,
        join_pair_budget=None,
        direct_sources=None,
    ):
        if selectivity_estimator not in {"legacy", "mask"}:
            raise ValueError(
                "Ancestor condition estimation requires the native sampler"
            )
        self.selectivity_estimator = selectivity_estimator
        if (
            type(sample_depth_selectivity) is not bool
            or type(ensure_prefix_depth_rates) is not bool
            or type(sample_accuracy) is not bool
        ):
            raise ValueError("Sampling mode flags must be booleans")
        self.sample_depth_selectivity = sample_depth_selectivity
        self.ensure_prefix_depth_rates = ensure_prefix_depth_rates
        self.sample_accuracy = sample_accuracy
        self.operator_sample_rows = operator_sample_rows
        self.max_operator_sample_rows = max_operator_sample_rows
        self.selectivity_parallel_models = selectivity_parallel_models
        if (
            type(max_operator_sample_rows) is not int
            or max_operator_sample_rows < operator_sample_rows
        ):
            raise ValueError("max_operator_sample_rows must be >= operator_sample_rows")
        self.relational_sample_rows = relational_sample_rows
        self.seed = seed
        if type(seed) is not int:
            raise ValueError("Sampling seed must be an integer")
        if any(
            type(v) is not int or v < 1
            for v in (
                initial_size,
                operator_sample_rows,
                relational_sample_rows,
                batch_size,
                budget,
                min_survivors,
                accuracy_size,
                max_strata,
                depth_targets,
                min_depth_samples,
                max_model_calls,
                max_accuracy_batches_per_operator,
            )
        ):
            raise ValueError("Sampling limits must be positive integers")
        self.max_accuracy_batches_per_operator = max_accuracy_batches_per_operator
        self.accuracy_batches_used = Counter()
        self.accuracy_sampled_inputs = defaultdict(set)
        self.accuracy_request_used = Counter()
        if (
            type(depth_error) not in (int, float)
            or not math.isfinite(depth_error)
            or not 0 < depth_error < 1
        ):
            raise ValueError("Depth error must be finite and in (0,1)")
        self.max_strata = max_strata
        self.min_depth_samples, self.depth_error, self.max_model_calls = (
            min_depth_samples,
            depth_error,
            max_model_calls,
        )
        self.strata, self.row_strata, self.length_counts = {}, {}, {}
        self.backend = backend
        self.direct_sources = direct_sources
        self.timings = PhaseTimings()
        self.sketch_config = dict(adaptive_sketch or {})
        self.population_sketch_plans = {}
        self._semantic_sources = {}
        # The source pool serves both classes of consumer. Each operator draws
        # its own bounded rows/pairs from this outcome-independent pool.
        self.initial_size = self.batch_size = self.budget = max(
            operator_sample_rows, relational_sample_rows
        )
        self.minimum, self.accuracy_size = min_survivors, accuracy_size
        (
            self.sources,
            self.admitted,
            self.population_summaries,
            self.population_counts,
        ) = (
            {},
            {},
            {},
            {},
        )
        from planning.observations.cache import PrefixSummary
        from planning.observations.statistics import serialize_value

        for relation, records in sources.items():
            rows = list(records)
            if any(not isinstance(r, SampleRow) or not r.identity for r in rows) or len(
                {r.identity for r in rows}
            ) != len(rows):
                raise ValueError("Base sample IDs must be nonempty and unique")
            self.population_counts[relation] = (population_counts or {}).get(
                relation, len(rows)
            )
            population = self.population_counts[relation]
            if (
                type(population) not in (int, float)
                or not math.isfinite(population)
                or population < len(rows)
            ):
                raise ValueError(
                    "Base population must be finite and no smaller than its sample"
                )
            fields = {
                name: [serialize_value(r.values.get(name)) for r in rows]
                for name in {k for r in rows for k in r.values}
            }
            if fields:
                built = self._field_summaries(fields)
                self.population_sketch_plans[relation] = built["plan"]
                for name, raw in built["summaries"].items():
                    self.population_summaries[(relation, name)] = PrefixSummary.load(
                        raw
                    )
            self._prepare(relation, rows, seed, serialize_value)
            if rows:
                scale = self.population_counts[relation] / len(rows)
                for stratum in self.strata[relation]:
                    stratum["population"] *= scale
                for key in list(self.length_counts):
                    if key[0] == relation:
                        self.length_counts[key] = Counter(
                            {
                                length: count * scale
                                for length, count in self.length_counts[key].items()
                            }
                        )
        self.selection_cache, self.quality_cache, self.request_results = {}, set(), {}
        self._signatures = OrderedDict()
        self._signature_rows = 0
        self._weight_cohort = None
        self._row_weights: Any = None
        self._active_weights = None
        self._prefix_common = {}
        self.offline_statistics = {}
        self.task_selectivity = {}
        self._boundaries = OrderedDict()
        self._boundary_rows = 0
        self._boundary_weights = None
        self.calls = {
            "selectivity": 0,
            "accuracy": 0,
            "judge": 0,
            "model_requests": 0,
            "model_http_requests": 0,
            "accuracy_model_requests": 0,
            "selection_model_requests": 0,
        }

    def sample_limit(self, problem, op):
        return (
            self.operator_sample_rows
            if op["operator_id"] in problem.ops
            else self.relational_sample_rows
        )

    def semantic_source_rows(self, relation):
        from .independent import bounded

        if self.direct_sources is not None:
            return tuple(
                self.direct_sources[str(relation)][str(self.operator_sample_rows)]
            )
        if relation not in self._semantic_sources:
            self._semantic_sources[relation] = bounded(
                self,
                self.sources[relation],
                ["semantic-source", relation],
                limit=self.operator_sample_rows,
            )
        return self._semantic_sources[relation]

    def _selection_remaining(self):
        used = self.calls["model_requests"] - self.calls["accuracy_model_requests"]
        return self.max_model_calls - used

    def _field_summaries(self, fields):
        """Build an independent bank for the current field values."""
        from planning.observations.statistics import FIELD_SERIALIZER

        from ..kernels.sketches import adaptive_build

        built = adaptive_build(fields, self.sketch_config)
        for raw in built["summaries"].values():
            raw["metadata"]["serializer"] = FIELD_SERIALIZER
        return built

    def _boundary_statistics(self, rows):
        """Share published distributions only within the exact weight cohort."""
        from .distribution import summarize_rows

        weights = self._active_weights
        if weights is not self._boundary_weights:
            self._boundaries.clear()
            self._boundary_rows = 0
            self._boundary_weights = weights
        key = (id(rows), id(weights))
        if weights is not None and key in self._boundaries:
            self._boundaries.move_to_end(key)
            return self._boundaries[key][2]
        result = summarize_rows(rows, self._weight)
        if (
            weights is not None
            and type(rows) is tuple
            and len(rows) <= 65536
            and all(is_frozen_evidence(row.values) for row in rows)
        ):
            result = result[0], freeze_evidence(result[1])
            self._boundaries[key] = rows, weights, result
            self._boundary_rows += len(rows)
            while len(self._boundaries) > 128 or self._boundary_rows > 65536:
                _, old = self._boundaries.popitem(last=False)
                self._boundary_rows -= len(old[0])
        return result

    def _prepare(self, relation, rows, seed, serialize):
        # Fixed query-local length strata and outcome-independent sample order.
        lengths = {
            r.identity: {k: len(serialize(v)) for k, v in r.values.items()}
            for r in rows
        }
        direct = self.direct_sources is not None
        groups = [sorted(rows, key=lambda r: r.identity)] if rows else []
        names = sorted({name for row in rows for name in row.values})
        while not direct and len(groups) < min(self.max_strata, self.budget, len(rows)):
            choices = []
            for index, members in enumerate(groups):
                for name in names:
                    values = [lengths[r.identity].get(name, 4) for r in members]
                    bins = sorted({value.bit_length() for value in values})
                    if len(bins) > 1:
                        boundary = bins[(len(bins) - 1) // 2]
                        importance = (max(values) - min(values)) * math.sqrt(
                            len(members)
                        )
                        choices.append((importance, index, name, boundary))
            if not choices:
                break
            _, index, name, boundary = max(choices)
            members = groups[index]
            low = [
                r
                for r in members
                if lengths[r.identity].get(name, 4).bit_length() <= boundary
            ]
            high = [
                r
                for r in members
                if lengths[r.identity].get(name, 4).bit_length() > boundary
            ]
            groups[index : index + 1] = [low, high]
        self.strata[relation], self.row_strata[relation] = [], {}
        for index, members in enumerate(groups):
            members = [replace(r, lineage=((relation, r.identity),)) for r in members]
            if not direct:
                random.Random(int(digest([seed, relation, index])[:16], 16)).shuffle(
                    members
                )
            pool = tuple(members if direct else members[: self.budget])
            self.strata[relation].append(
                {
                    "population": len(members),
                    "rows": pool,
                    "selected": len(pool) if direct else 0,
                }
            )
            self.row_strata[relation].update({r.identity: index for r in pool})
            for name in {k for r in members for k in r.values}:
                self.length_counts[(relation, index, name)] = Counter(
                    lengths[r.identity].get(name, 4) for r in members
                )
        if direct:
            self.sources[relation], self.admitted[relation] = tuple(rows), len(rows)
        else:
            self.sources[relation], self.admitted[relation] = (), 0
            self._admit(relation, self.initial_size)

    def _admit(self, relation, count, priorities=None):
        selected = list(self.sources[relation][: self.admitted[relation]])
        for _ in range(min(count, self.budget - len(selected))):
            available = [
                (i, h)
                for i, h in enumerate(self.strata[relation])
                if h["selected"] < len(h["rows"])
            ]
            if not available:
                break

            def score(item):
                index, h = item
                if h["selected"] == 0:
                    return (1, h["population"])
                value = (priorities or {}).get(index, 0)
                return (
                    0,
                    (
                        value / (h["selected"] * (h["selected"] + 1))
                        if priorities
                        else math.sqrt(h["population"]) / (h["selected"] + 1)
                    ),
                )

            _, chosen = max(available, key=score)
            selected.append(chosen["rows"][chosen["selected"]])
            chosen["selected"] += 1
        changed = len(selected) > self.admitted[relation]
        # 已准入前缀保持不变，尚未准入的候选只占有界的分层样本池。
        self.admitted[relation] = len(selected)
        if changed:
            self.sources[relation] = tuple(selected)
        return changed

    def _weight(self, row, counts=None):
        # Only incremental replay activates this memo. Explicit accuracy counts
        # and direct observations always read their current population/quota.
        memo = self._active_weights if counts is None else None
        if memo is not None:
            return memo.weight(row)
        value = math.prod(
            self.strata[rel][self.row_strata[rel][identity]]["population"]
            / (
                counts[(rel, self.row_strata[rel][identity])]
                if counts is not None
                else self.strata[rel][self.row_strata[rel][identity]]["selected"]
            )
            for rel, identity in set(row.lineage)
        )
        return value

    def _row_effective(self, rows):
        if isinstance(rows, CartesianRows):
            return rows.effective(self._weight)
        if self._active_weights is not None:
            return self._active_weights.effective(rows)
        return self._effective([self._weight(row) for row in rows])

    _effective = staticmethod(native_observations.effective)
    _interval = staticmethod(native_observations.interval)

    def _population_index(self, relation, name):
        result = []
        for index in range(len(self.strata[relation])):
            counts = sorted(self.length_counts.get((relation, index, name), {}).items())
            suffix = [0] * (len(counts) + 1)
            for position in range(len(counts) - 1, -1, -1):
                suffix[position] = suffix[position + 1] + counts[position][1]
            result.append((tuple(length for length, _ in counts), tuple(suffix)))
        return tuple(result)

    def _field_rates(self, relation, name, reached, eligible=None, complete=True):
        from planning.observations.statistics import serialize_value

        summary = self.population_summaries[(relation, name)]
        samples = self.sources[relation][: self.admitted[relation]]
        # Each observation builds its own length index. Eligibility and pass
        # membership are independent of the anchor and need checking only once.
        members = [[] for _ in self.strata[relation]]
        for row in samples:
            length = len(serialize_value(row.values.get(name)))
            if eligible is None or row.identity in eligible:
                members[self.row_strata[relation][row.identity]].append(
                    (length, int(row.identity in reached))
                )
        cohorts = []
        for group in members:
            group.sort()
            suffix = [0] * (len(group) + 1)
            for index in range(len(group) - 1, -1, -1):
                suffix[index] = suffix[index + 1] + group[index][1]
            cohorts.append(([length for length, _ in group], suffix))
        population_cohorts = self._population_index(relation, name)
        return native_observations.field_rates(
            cohorts,
            population_cohorts,
            [(h["population"], h["selected"]) for h in self.strata[relation]],
            [0, *summary.sketches] if self.sample_depth_selectivity else [0],
            eligible is None,
            complete,
            self.min_depth_samples,
            self.depth_error,
        )

    def _models(self, problem, op):
        supported = [
            m
            for m in op["models"]
            if not (
                m not in problem.catalog
                and m in problem.profile.get("teacher_models", {})
            )
            and op.get("kind")
            in problem.catalog.get(m, {}).get("supported_kinds", [op.get("kind")])
        ]
        if not supported:
            raise ValueError("No model supports this sampling task")
        if len(supported) == 1:
            return supported
        # Size must be known to implement the specified smallest/strongest choice.
        for model in supported:
            size = problem.catalog.get(model, {}).get("parameters")
            if not problem._number(size) or size <= 0:
                raise ValueError("Sampling requires model parameter counts")
        return sorted(supported, key=lambda m: (problem.catalog[m]["parameters"], m))

    def _signature(self, op, inputs, *, readonly=False):
        metadata = [
            op.get("kind", op.get("type")),
            op.get("task", op.get("expressions")),
            op.get("required_columns"),
            op.get("produced_fields"),
            op.get("output_schema"),
            op.get("sql"),
            op.get("candidate_sql"),
            op.get("pair_columns"),
        ]
        key = None
        leaves, size = [], 0
        if readonly:

            def source_key(rows):
                if isinstance(rows, CartesianRows):
                    left, right = source_key(rows.left), source_key(rows.right)
                    return (
                        ("product", left, right)
                        if left is not None and right is not None
                        else None
                    )
                if type(rows) is tuple:
                    leaves.append(rows)
                    return ("rows", id(rows))
                return None

            sources = tuple(source_key(group) for group in inputs)
            size = sum(len(leaf) for leaf in leaves)
            if all(source is not None for source in sources) and size <= 65536:
                key = canonical(metadata), sources
                cached = self._signatures.get(key)
                if cached is not None:
                    self._signatures.move_to_end(key)
                    return cached[1]
                # A miss proves the complete row bank immutable. Hits retain
                # those exact tuples and only revisit the product's structure.
                if not all(
                    type(row) is SampleRow
                    and is_frozen_evidence((row.identity, row.values, row.lineage))
                    for leaf in leaves
                    for row in leaf
                ):
                    key = None
        if any(isinstance(group, CartesianRows) for group in inputs):
            from ..backends.factorized import fingerprint

            signature = digest([*metadata, [fingerprint(group) for group in inputs]])
        else:
            signature = digest(
                [
                    *metadata,
                    [
                        [(r.identity, r.values, r.lineage) for r in rows]
                        for rows in inputs
                    ],
                ]
            )
        if key is not None:
            # Strong leaf references prevent recycled IDs, without retaining
            # recreated product wrappers or expanding their logical population.
            self._signatures[key] = (
                tuple(leaves),
                signature,
                size,
            )
            self._signature_rows += size
            while len(self._signatures) > 1024 or self._signature_rows > 65536:
                _, evicted = self._signatures.popitem(last=False)
                self._signature_rows -= evicted[2]
        return signature

    def _sampling_model(self, problem, op):
        return op.get("_sample_model") or self._models(problem, op)[0]

    def _call_backend(self, op, inputs, model, layout, phase, *, pairs=None):
        physical = getattr(self.backend, "physical_request_telemetry", False)
        start = len(getattr(self.backend, "events", ()))
        try:
            with self.timings.measure(phase):
                if pairs is not None:
                    return tuple(self.backend.join_pairs(op, pairs, model, layout))
                return tuple(self.backend(op, inputs, model, layout))
        finally:
            if model:
                # A submit event is emitted at the HTTP boundary, before waiting
                # for a response. Failed attempts count; cache hits emit none. A
                # retransmission repeats one logical request but is billed again,
                # so it counts as an HTTP request too.
                requests = (
                    sum(
                        event.get("event")
                        in ("sampling_llm_submit", "sampling_llm_retransmit")
                        for event in self.backend.events[start:]
                    )
                    if physical
                    else (
                        len(pairs)
                        if pairs is not None
                        else math.prod(len(group) for group in inputs)
                    )
                )
                accuracy = phase in {"accuracy_model", "accuracy_judge"}
                self.calls["model_requests"] += requests
                self.calls["model_http_requests"] = self.calls["model_requests"]
                self.calls[
                    (
                        "accuracy_model_requests"
                        if accuracy
                        else "selection_model_requests"
                    )
                ] += requests
                if accuracy:
                    self.accuracy_request_used[op["operator_id"]] += requests
                    self.calls[
                        "judge" if phase == "accuracy_judge" else "accuracy"
                    ] += requests

    def request_latency(self):
        """Actual request telemetry; unavailable reports remain unknown."""
        values = [
            event["latency_ms"] / 1000
            for event in getattr(self.backend, "events", [])
            if event.get("event") == "semantic_row"
            and type(event.get("latency_ms")) in (int, float)
            and math.isfinite(event["latency_ms"])
            and event["latency_ms"] >= 0
        ]
        return {
            "reported_requests": len(values),
            "seconds": sum(values) if values else None,
            "source": "semantic_row.latency_ms; excludes SQL and history processing",
        }

    def _execute(self, op, inputs, model=None, *, pairs=None):
        self.calls["selectivity"] += 1
        layout = op.get("default_layout") if model else None
        rows = self._call_backend(
            op,
            inputs,
            model,
            layout,
            "selection_semantic" if model else "selection_relational",
            pairs=pairs,
        )
        if any(not isinstance(r, SampleRow) for r in rows) or len(
            {r.identity for r in rows}
        ) != len(rows):
            raise ValueError(
                "Sampling backend must return uniquely identified SampleRows"
            )
        possible = set().union(*(lineages(group) for group in inputs))
        aggregate = "AGGREGATE" in op.get("kind", op.get("type", "")).upper()
        if any(
            (possible and not r.lineage and not aggregate)
            or not set(r.lineage) <= possible
            for r in rows
        ):
            raise ValueError("Sampling backend lost or invented base-row lineage")
        if (
            len(inputs) == 1
            and not aggregate
            and not (
                all(inputs[0].contains(row) for row in rows)
                if isinstance(inputs[0], CartesianRows)
                else {r.identity for r in rows} <= {r.identity for r in inputs[0]}
            )
        ):
            raise ValueError("Unary sampling must preserve row IDs")
        if len(inputs) == 2 and any(
            isinstance(group, CartesianRows) for group in inputs
        ):
            indexes = [
                (
                    None
                    if isinstance(group, CartesianRows)
                    else {
                        tuple(sorted(row.lineage, key=repr)): row.identity
                        for row in group
                    }
                )
                for group in inputs
            ]
            domains = [
                {relation for relation, _ in lineages(group)} for group in inputs
            ]
            for row in rows:
                identities = []
                for group, index, domain in zip(inputs, indexes, domains):
                    origin = tuple(pair for pair in row.lineage if pair[0] in domain)
                    identities.append(
                        group.identity_for(origin)
                        if index is None
                        else index.get(origin)
                    )
                if None in identities or row.identity != digest(identities):
                    raise ValueError(
                        "Join samples must use stable candidate-pair identities"
                    )
        elif len(inputs) == 2:
            candidates = (
                pairs
                if pairs is not None
                else ((left, right) for left in inputs[0] for right in inputs[1])
            )
            expected = {
                digest([left.identity, right.identity]) for left, right in candidates
            }
            if not {r.identity for r in rows} <= expected:
                raise ValueError(
                    "Join samples must use stable candidate-pair identities"
                )
        return rows

    def _selection(self, problem, op, inputs, *, readonly=False):
        from .independent import bounded
        from .independent import units as sample_units

        limit = self.sample_limit(problem, op)
        if len(inputs) == 1:
            inputs = (bounded(self, inputs[0], op["operator_id"], limit=limit),)
        elif math.prod(map(len, inputs)) > limit:
            # Bound the candidate domain before executing SQL or issuing model
            # requests; bounded sides alone would still admit limit**2 pairs.
            output = []
            for unit in sample_units(self, inputs, op["operator_id"], limit=limit):
                emitted = self._selection(problem, op, unit, readonly=readonly)
                if emitted is None:
                    return None
                output.extend(emitted)
            return tuple(output)
        key = self._signature(op, inputs, readonly=readonly)
        semantic = op["operator_id"] in problem.ops
        model = self._sampling_model(problem, op) if semantic else None
        key = digest([key, model, op.get("default_layout")])
        # Relational SQL can process the complete boundary in one vectorized
        # stage. Only semantic requests need per-unit caching and call budgets.
        if not semantic:
            if key not in self.selection_cache:
                factory = getattr(self.backend, "cartesian", None)
                if (
                    factory is not None
                    and op.get("kind", op.get("type", "")).upper() == "CROSS_PRODUCT"
                ):
                    self.calls["selectivity"] += 1
                    with self.timings.measure("selection_relational"):
                        self.selection_cache[key] = factory(op, inputs, limit)
                else:
                    self.selection_cache[key] = self._execute(op, inputs)
            return self.selection_cache[key]
        if callable(getattr(self.backend, "sample_request_key", None)):
            if key not in self.selection_cache:
                from .semantic import select

                observed = select(self, problem, op, inputs)
                if observed is None:
                    return None
                self.selection_cache[key] = observed
            return self.selection_cache[key]
        if key not in self.selection_cache:
            # Cache individual unary rows/pairs as well as the whole boundary;
            # replenishment executes only newly admitted units.
            if len(inputs) == 1:
                units = (([r],) for r in inputs[0])
            else:
                units = (([left], [right]) for left in inputs[0] for right in inputs[1])
            missing, outputs = [], []
            for unit in units:
                unit_key = digest(
                    [
                        self._signature(op, unit, readonly=readonly),
                        model,
                        op.get("default_layout"),
                    ]
                )
                if unit_key in self.selection_cache:
                    outputs.extend(self.selection_cache[unit_key])
                else:
                    missing.append((unit_key, unit))
                    if len(missing) > self._selection_remaining():
                        return None
            if len(inputs) == 1 and missing:
                fresh = self._execute(op, ([u[0][0] for _, u in missing],), model)
                by_id = {r.identity: r for r in fresh}
                for unit_key, unit in missing:
                    row = by_id.get(unit[0][0].identity)
                    self.selection_cache[unit_key] = (row,) if row else ()
                outputs.extend(fresh)
            elif missing and callable(getattr(self.backend, "join_pairs", None)):
                pairs = [(unit[0][0], unit[1][0]) for _, unit in missing]
                fresh = self._execute(op, inputs, model, pairs=pairs)
                by_id = {row.identity: row for row in fresh}
                for unit_key, unit in missing:
                    row = by_id.get(digest([unit[0][0].identity, unit[1][0].identity]))
                    self.selection_cache[unit_key] = (row,) if row else ()
                    if row is not None:
                        outputs.append(row)
            else:
                for unit_key, unit in missing:
                    fresh = self._execute(op, unit, model)
                    self.selection_cache[unit_key] = fresh
                    outputs.extend(fresh)
            self.selection_cache[key] = tuple(outputs)
        return self.selection_cache[key]

    def _sampled_statistics(
        self, node, op, inputs, emitted, facts, *, input_boundaries=None
    ):
        """Fresh state-local sampling facts; layouts and quality stay outside."""
        from .distribution import request_characters

        reached = facts["reached"]
        selectivity = facts["selectivity"]
        relational = facts["relational"]
        semantic = facts["semantic"]

        if input_boundaries is None:
            input_boundaries = [self._boundary_statistics(group) for group in inputs]
        widths = [width for width, _ in input_boundaries]
        output_width, output_fields = self._boundary_statistics(emitted)
        field_info = {}
        for _, fields_info in input_boundaries:
            field_info.update(fields_info)
        info: dict[str, Any] = {
            "input_width": (
                sum(w for w in widths if w is not None)
                if all(w is not None for w in widths)
                else None
            ),
            "output_width": output_width,
            "width_basis": "serialized scalar bytes",
            "input_rows": facts["input_rows"],
            "output_rows": facts["output_rows"],
            "sample_input_rows": reached,
            "sample_output_rows": len(emitted),
            "selectivity": selectivity,
            "effective_samples": (
                self._row_effective(inputs[0])
                if len(inputs) == 1
                else min(
                    (self._row_effective(group) for group in inputs),
                    default=0,
                )
            ),
            "uncertain": reached == 0 or not facts["output_complete"],
            "sampling_complete": facts["output_complete"],
            "model_budget_exhausted": not facts["output_complete"],
            "sample_model": facts["sample_model"],
            "input_side_rows": facts["side_rows"],
            "input_side_widths": widths,
            "input_field_statistics": field_info,
            "output_field_statistics": (
                output_fields if facts["output_complete"] else {}
            ),
        }
        if not semantic:
            info.update(
                {
                    k: v
                    for k, v in relational.items()
                    if k not in ("input_width", "output_width")
                }
            )
        # Prefix sketches belong to semantic request inputs, not arbitrary
        # operator outputs. _cache_inputs reuses base-field sketches and builds
        # conditional sketches when a consumer actually reads derived values.
        if semantic:
            chars, left_chars, right_chars = request_characters(op, inputs, field_info)
            info.update(
                prompt_input_characters=chars,
                left_data_characters=left_chars,
                right_data_characters=right_chars,
            )
        input_depths = {}
        output_depths = facts["independent_depths"]
        info["depth_survival_statistics"] = input_depths
        info["depth_selectivity_statistics"] = output_depths
        census = all(
            h["selected"] == h["population"]
            for rel in node.relations
            for h in self.strata[rel]
        )
        effective = info["effective_samples"]
        bounds = (
            self._interval(
                selectivity * effective,
                effective,
                effective if census else math.inf,
            )
            if selectivity is not None
            else (0.0, 1.0)
        )
        info["intervals"] = {"selectivity": list(bounds)} if semantic else {}
        info["selectivity_precision"] = (
            semantic
            and selectivity is not None
            and (
                census
                or effective >= self.min_depth_samples
                and (bounds[1] - bounds[0]) / 2 <= self.depth_error
            )
        )
        info["input_survival_precision"] = bool(input_depths) and all(
            e["0"]["reliable"] for e in input_depths.values()
        )
        if len(inputs) == 2:
            for side, relation in zip(("left", "right"), facts["child_relations"]):
                values = next(
                    (
                        depths["0"]
                        for key, depths in output_depths.items()
                        if key.startswith(str(relation) + ":")
                    ),
                    None,
                )
                info[side + "_selectivity"] = values["rate"] if values else None
            info["depth_basis"] = (
                "distinct-row match rate against sampled opposite side"
            )
        return info, (output_width, output_fields)

    def _replay(self, problem, state):
        """Observe operators independently using one fixed base-weight cohort."""
        cohorts = {
            relation: (
                id(rows),
                self.admitted[relation],
                self.population_counts[relation],
                tuple((h["population"], h["selected"]) for h in self.strata[relation]),
            )
            for relation, rows in self.sources.items()
        }
        token = tuple(cohorts.items())
        if token != self._weight_cohort:
            self._weight_cohort = token
            self._row_weights = native_observations.Weights(
                self.strata, self.row_strata
            )
        previous = self._active_weights
        self._active_weights = self._row_weights
        try:
            from .independent import observe

            return observe(self, problem, state)
        finally:
            # Fresh nested or failed observations cannot leak a weight context.
            self._active_weights = previous

    def _cache_inputs(
        self,
        problem,
        op,
        action,
        inputs,
        complete=True,
        *,
        estimated_survivals=None,
    ):
        from planning.observations.statistics import prefix, serialize_value

        layout = action["layout"]
        first = layout.get("field_order", layout.get("left_field_order", []))
        if not first:
            return {}
        names = [first[0]]
        if "right_field_order" in layout:
            names.append(layout["right_field_order"][0])
        descriptors = {c["name"]: c for c in op.get("input_field_schema", [])}
        summaries, survivals = {}, {}
        estimated_survival = False
        first_origin = None
        conditional = {}
        for name in names:
            descriptor = descriptors.get(name, {"name": name})
            key = descriptor.get("value_key", descriptor.get("name"))
            source = next(
                (rows for rows in inputs if rows and key in rows[0].values), ()
            )
            origins = {
                relation
                for relation, _ in lineages(source)
                if (relation, key) in self.population_summaries
            }
            if not origins:
                origins = {
                    rel
                    for rel, field in self.population_summaries
                    if field == key and (not op.get("scope") or rel in op["scope"])
                }
            if len(origins) == 1:
                relation = next(iter(origins))
                if name == names[0]:
                    first_origin = [str(relation), key]
                summary = self.population_summaries[(relation, key)]
                reached = {
                    identity for rel, identity in lineages(source) if rel == relation
                }
                evidence = self._field_rates(
                    relation,
                    key,
                    reached,
                    complete=complete,
                )
                survivals[name] = {
                    "total": evidence["0"]["rate"],
                    "depths": {d: e["rate"] for d, e in evidence.items() if d != "0"},
                }
                estimate = (estimated_survivals or {}).get(f"{relation}:{key}")
                if estimate is not None:
                    estimated_survival = True
                    survivals[name] = {
                        "total": estimate.get("0"),
                        "depths": {d: rate for d, rate in estimate.items() if d != "0"},
                    }

            elif source:
                conditional[name] = (
                    ProductFieldValues(source, key, serialize_value)
                    if isinstance(source, CartesianRows)
                    else [serialize_value(r.values[key]) for r in source]
                )
                estimate = (estimated_survivals or {}).get("generated:" + key)
                survivals[name] = (
                    {
                        "total": estimate.get("0"),
                        "depths": {
                            d: value for d, value in estimate.items() if d != "0"
                        },
                    }
                    if estimate is not None
                    else 1.0
                )
                continue
            else:
                continue
            summaries[name] = summary.json()
        conditional_plan = None
        if conditional:
            built = self._field_summaries(conditional)
            summaries.update(built["summaries"])
            conditional_plan = built["plan"]
        if first[0] not in summaries:
            return {}
        system = getattr(self.backend, "system_prefix", None) or problem.profile.get(
            "system_prefix", "semantic-global-v1"
        )
        specification = getattr(self.backend, "operator_specifications", {}).get(
            op["kind"]
        ) or problem.profile.get("operator_specifications", {}).get(
            op["kind"], op["kind"]
        )
        problem.profile["system_prefix"] = system
        problem.profile.setdefault("operator_specifications", {})[op["kind"]] = (
            specification
        )
        result: dict[str, Any] = {
            "summary": summaries[first[0]],
            "alpha": survivals[first[0]],
            "pre": prefix(op, action, system, specification),
            "basis": "independent base-field summary with preceding depth selectivity product",
            "conditional_sketch_plan": conditional_plan,
            "input_survival_estimated": estimated_survival,
        }
        if estimated_survival:
            result["basis"] = (
                "independent base-field summary with preceding depth selectivity product"
            )
            result["reliability"] = dict.fromkeys(
                ("alpha", "right_alpha", "l_self", "l_existing", "l_previous"), 0.1
            )
        if first_origin is not None:
            result["cache_field_origin"] = first_origin
        if "right_field_order" in layout:
            right = layout["right_field_order"][0]
            if right not in summaries:
                return {}
            result["right_summary"] = summaries[right]
            result["right_alpha"] = survivals[right]
            # Whole left D segment is repeated for every right candidate row.
            left_names = op["left_input_fields"]
            result["left_total_length"] = (
                self._left_input_length(inputs[0], left_names, descriptors)
                * result["summary"]["count"]
                if inputs[0]
                else 0
            )
        return result

    def _left_input_length(self, rows, names, descriptors):
        """Keep the original weighted sum order for one read-only boundary."""
        from planning.observations.statistics import serialize_value

        fields = tuple(
            descriptors.get(name, {}).get("value_key", name) for name in names
        )
        if isinstance(rows, CartesianRows):
            summaries = self._boundary_statistics(rows)[1]
            return sum(
                summaries.get(name, {}).get("average_length", 0.0) for name in fields
            )
        lengths = [
            sum(len(serialize_value(row.values.get(name))) for name in fields)
            for row in rows
        ]
        average = (
            sum(length * self._weight(row) for length, row in zip(lengths, rows))
            / sum(self._weight(row) for row in rows)
            if lengths
            else 0
        )
        return average

    def selectivity(self, problem, state):
        """Observe each operator on its fixed independent base input sample."""
        problem._prefix_coupling = self._prefix_coupling
        updated = self._replay(problem, state)
        for action in problem.semantic_actions(updated):
            key = updated._action_keys_by_operator[action["operator_id"]]
            updated.statistics[key].update(
                problem.estimate(
                    updated, action, updated.partial_plans, copy_observation=False
                )
            )
        return updated

    def _prefix_coupling(self, source, target):
        """Read offline conditional samples only; never sample input rows here."""
        from planning.observations.offline_statistics import prefix_common

        f, g = source.get("cache_field_origin"), target.get("cache_field_origin")
        if not f or not g or f[0] != g[0]:
            return None
        if f[1] == g[1]:
            return {"kind": "same_field"}
        key = tuple(f) + tuple(g)
        if key in self._prefix_common:
            return self._prefix_common[key]
        offline = self.offline_statistics.get(str(f[0]))
        if (
            not offline
            or f[1] not in offline["columns"]
            or g[1] not in offline["columns"]
        ):
            return None
        result = {
            "kind": "same_relation",
            "common": prefix_common(
                offline["cooccurrence"],
                source["summary"],
                target["summary"],
                offline["columns"][f[1]],
                offline["columns"][g[1]],
            ),
        }
        self._prefix_common[key] = result
        return result

    def accuracy_inputs(self, problem, state, oid):
        from .independent import Inputs, ProjectionBudgetExhausted, units

        node = state._nodes_by_operator[oid]
        op = problem.execution_operator(state, oid)
        try:
            inputs = Inputs(self, problem, state).operator(node, op)
        except ProjectionBudgetExhausted:
            return (), ()
        return inputs, units(self, inputs, oid, limit=self.sample_limit(problem, op))

    def accuracy(self, problem, state):
        if not self.sample_accuracy:
            return state
        # Beam 中较早保留的状态必须使用自身的准入配额，不能套用后继的补样权重。
        counts = Counter(
            (
                next(iter(node.relations)),
                self.row_strata[next(iter(node.relations))][row.identity],
            )
            for node in state.partial_plans.nodes
            if node.operator_id is None
            for row in state.sample.get(node.node_id, ())
        )
        if callable(getattr(self.backend, "sample_request_key", None)):
            from .semantic import quality

            quality(self, problem, state, counts)
        else:
            self._legacy_quality(problem, state, counts)
        from planning.observations.accuracy import evidence as accuracy_evidence

        stats = copy_statistics(state.statistics)
        for action in problem.semantic_actions(state):
            op = problem.ops[action["operator_id"]]
            estimate: dict[str, Any] | None = accuracy_evidence(
                problem.profile,
                action["model_id"],
                op["kind"],
                action["layout"],
            )
            if isinstance(estimate, dict):
                stats.setdefault(action_key(action), {})["accuracy"] = estimate["value"]
                task_evidence = (
                    problem.profile.get("task_accuracy", {})
                    .get(action["model_id"], {})
                    .get(task_key(op), estimate)
                )
                effective = task_evidence.get("effective_samples")
                if effective:
                    stats[action_key(action)].setdefault("intervals", {})[
                        "accuracy"
                    ] = list(
                        self._interval(
                            task_evidence["value"] * effective, effective, math.inf
                        )
                    )
            elif estimate is not None:
                stats.setdefault(action_key(action), {})["accuracy"] = estimate
        return replace(state, statistics=stats)

    def _legacy_quality(self, problem, state, counts):
        from planning.observations.accuracy import sampling_layouts

        from .accuracy_budget import legacy_input_key, select_accuracy_groups
        from .quality import record_quality
        from .semantic import InputGroup

        for action in problem.semantic_actions(state):
            oid = action["operator_id"]
            if (
                self.accuracy_batches_used[oid]
                >= self.max_accuracy_batches_per_operator
            ):
                continue
            op = problem.execution_operator(state, oid)
            inputs, selected_units = self.accuracy_inputs(problem, state, oid)
            if not inputs or any(not rows for rows in inputs):
                continue
            models = self._models(problem, op)
            teacher = problem.profile.get("judge_model", models[-1])
            from planning.settings.teacher import teacher_supports

            if not teacher_supports(problem.profile, op, teacher):
                raise ValueError("Judge model must support the semantic task")
            units = selected_units
            observed = {}
            for unit in units:
                identity = legacy_input_key(op, unit, models[0])
                if identity not in observed:
                    observed[identity] = InputGroup(identity, unit)
                observed[identity].weight += math.prod(
                    self._weight(row, counts) for side in unit for row in side
                )
            selected = select_accuracy_groups(self, op, observed.values())
            produced = [
                column.get("name") if isinstance(column, dict) else column
                for column in op.get("produced_fields", op.get("output_schema", []))
            ]

            def result(rows):
                return sorted(
                    canonical(
                        [row.identity, {key: row.values.get(key) for key in produced}]
                    )
                    for row in rows
                )

            for group in selected:
                teacher_layout = op["default_layout"]
                reference_key = digest(
                    [self._signature(op, group.unit), teacher, teacher_layout]
                )
                if reference_key not in self.selection_cache:
                    self.selection_cache[reference_key] = self._call_backend(
                        op, group.unit, teacher, teacher_layout, "accuracy_judge"
                    )
                reference_rows = self.selection_cache[reference_key]
                reference = (bool(reference_rows), result(reference_rows))
                for layout in sampling_layouts(op):
                    for model in models:
                        actual_key = digest(
                            [self._signature(op, group.unit), model, layout]
                        )
                        if actual_key not in self.selection_cache:
                            self.selection_cache[actual_key] = self._call_backend(
                                op, group.unit, model, layout, "accuracy_model"
                            )
                        key = (actual_key, teacher)
                        if key in self.quality_cache:
                            continue
                        rows = self.selection_cache[actual_key]
                        actual = (bool(rows), result(rows))
                        record_quality(
                            problem.profile,
                            op,
                            model,
                            layout,
                            actual,
                            reference,
                            group.weight,
                            teacher,
                            actual_key,
                            reference_key,
                        )
                        self.quality_cache.add(key)
