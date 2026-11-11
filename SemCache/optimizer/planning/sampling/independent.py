"""Operator-local base samples, with only required projection dependencies.

Plan children supply cardinalities and survival products. Their predicate
outputs never supply another operator's selection or accuracy inputs.
"""

import math
from dataclasses import replace

from ..state import copy_statistics
from .evidence import (
    depth_rates,
    generated_field_identities,
    generated_survivals,
    input_evidence,
    projection_columns,
    value_producer_inputs,
)
from .prefix_rates import (
    complete_prefix_selectivity,
    local_prefix_selectivity,
)
from .units import (
    bounded,
    combine,
    evaluate,
    field_tokens,
    sampling_operator,
    units,
)

ASSUMPTION = "independent_base_v1"


class Inputs:
    def __init__(self, coordinator, problem, state, *, projections=True):
        self.coordinator, self.problem, self.state = coordinator, problem, state
        self.memo = {}
        self.role = None
        self.limit = max(
            coordinator.operator_sample_rows, coordinator.relational_sample_rows
        )
        self.projections = projections

    def operator(self, node, op):
        self.limit = self.coordinator.sample_limit(self.problem, op)
        needed = field_tokens(op.get("required_columns", []))
        inputs = []
        for side, child in enumerate(node.children):
            self.role = (op["operator_id"], side)
            inputs.append(self.materialize(child, needed))
        inputs = tuple(inputs)
        if node.operator_id not in self.problem.ops and len(inputs) == 2:
            inputs = self.relational_pair_domain(op, inputs)
        prepare = getattr(self.coordinator.backend, "prepare_inputs", None)
        if (
            prepare is not None
            and node.operator_id in self.problem.ops
            and self.projections
        ):
            inputs = prepare(op, inputs)
        return inputs

    def relational_pair_domain(self, op, inputs):
        """Keep both typed SQL sides while bounding their full pair domain."""
        if self.coordinator.direct_sources is not None:
            return inputs
        if not all(inputs) or math.prod(map(len, inputs)) <= self.limit:
            return inputs
        left = min(len(inputs[0]), max(1, math.isqrt(self.limit)))
        right = min(len(inputs[1]), max(1, self.limit // left))
        left = min(len(inputs[0]), max(1, self.limit // right))
        return tuple(
            bounded(
                self.coordinator,
                rows,
                [op["operator_id"], "pair-domain", side],
                limit=count,
            )
            for side, (rows, count) in enumerate(zip(inputs, (left, right)))
        )

    def materialize(self, nid, needed, *, limit=None):
        if limit is None:
            limit = self.limit
        key = (self.role, nid, tuple(sorted(needed)), limit)
        if key in self.memo:
            return self.memo[key]
        # Collapse skipped predicates and joins before drawing any coordinates.
        # Canonical relation order makes equivalent reordered trees share a pool.
        atoms = sorted(
            self.atoms(nid, needed, limit=limit),
            key=lambda atom: tuple(sorted(atom[0])),
        )
        if not atoms:
            # Constant expressions still need a row context on this input side.
            relation = min(self.state._nodes_by_id[nid].relations)
            atoms = [(frozenset({relation}), self.source_rows(relation, limit))]
        salt = [
            self.role,
            [sorted(relations) for relations, _ in atoms],
        ]
        result = combine(
            self.coordinator, [rows for _, rows in atoms], salt, limit=limit
        )
        self.memo[key] = result
        return result

    def source_rows(self, relation, limit):
        if self.coordinator.direct_sources is not None:
            return tuple(self.coordinator.direct_sources[str(relation)][str(limit)])
        # All semantic consumers and required producers start from the same
        # base cohort, including when another generated column is requested.
        if limit <= self.coordinator.operator_sample_rows:
            return self.coordinator.semantic_source_rows(relation)
        return self.coordinator.sources[relation]

    def atoms(self, nid, needed, *, limit):
        node = self.state._nodes_by_id[nid]
        if node.operator_id is None:
            if needed and not needed & field_tokens(node.fields):
                return []
            relation = next(iter(node.relations))
            return [(node.relations, self.source_rows(relation, limit))]
        op = sampling_operator(
            self.coordinator, self.problem, self.state, node.operator_id
        )
        kind = op.get("kind", op.get("type", "")).upper()
        if "sampling_passthrough_input" in op:
            return self.atoms(
                node.children[op["sampling_passthrough_input"]], needed, limit=limit
            )
        child_fields = field_tokens(
            [
                field
                for child in node.children
                for field in self.state._nodes_by_id[child].fields
            ]
        )
        produced = field_tokens(op.get("produced_fields", []))
        produced |= field_tokens(node.fields) - child_fields
        if (
            not self.projections
            or ("PROJECTION" not in kind and kind != "CTE_SCAN")
            or not needed & produced
        ):
            return [
                atom
                for child in node.children
                for atom in self.atoms(child, needed, limit=limit)
            ]
        if node.operator_id not in self.problem.ops:
            op = projection_columns(op, node.fields, needed, child_fields)
        required = (needed - produced) | field_tokens(op.get("required_columns", []))
        semantic = node.operator_id in self.problem.ops
        producer_limit = self.coordinator.sample_limit(self.problem, op)
        if not semantic:
            producer_limit = min(limit, producer_limit)
        # Required semantic projections own one query-local input cohort. A new
        # consumer/field request must not redraw a further model sample.
        previous_role = self.role
        children = []
        try:
            for side, child in enumerate(node.children):
                if semantic:
                    self.role = (node.operator_id, side)
                children.append(self.materialize(child, required, limit=producer_limit))
        finally:
            self.role = previous_role
        children = tuple(children)
        prepare = getattr(self.coordinator.backend, "prepare_inputs", None)
        if prepare is not None and node.operator_id in self.problem.ops:
            children = prepare(op, children)
        result, _, _, complete = evaluate(self.coordinator, self.problem, op, children)
        if not complete:
            raise ProjectionBudgetExhausted()
        relations = frozenset(
            rel for rows in children for row in rows for rel, _ in row.lineage
        )
        return [(relations or node.relations, result)]


class ProjectionBudgetExhausted(Exception):
    """A required generated input is unknown; do not substitute empty evidence."""


def observe(coordinator, problem, state):
    from ..kernels.observations import observation
    from .coordinator import _publish_observation
    from .survival import estimate

    problem.profile["sampling_strategy"] = "independent_base"
    bank = problem.profile.setdefault("operator_statistics", {})
    builder = Inputs(coordinator, problem, state)
    outputs, stats, observations, cardinalities = {}, {}, {}, {}
    for node in state.partial_plans.nodes:
        oid = node.operator_id
        if oid is None:
            relation = next(iter(node.relations))
            rows = coordinator.sources[relation]
            boundary = coordinator._boundary_statistics(rows)
            info = {
                "input_rows": coordinator.population_counts[relation],
                "output_rows": coordinator.population_counts[relation],
                "selectivity": 1.0,
                "input_width": boundary[0],
                "output_width": boundary[0],
                "input_field_statistics": boundary[1],
                "output_field_statistics": boundary[1],
                "sampling_strategy": "independent_base",
                "estimated_output_survival": {
                    f"{relation}:{name}": {
                        str(depth): 1.0
                        for depth in (
                            [0, *summary.sketches]
                            if coordinator.sample_depth_selectivity
                            or getattr(coordinator, "ensure_prefix_depth_rates", True)
                            else [0]
                        )
                    }
                    for (rel, name), summary in coordinator.population_summaries.items()
                    if rel == relation
                },
            }
            key, complete = f"scan:{node.node_id}", True
            from ..observations.evidence import freeze_evidence

            problem.profile.setdefault("source_statistics", {})[str(node.node_id)] = (
                freeze_evidence(info)
            )
        else:
            key = state._action_keys_by_operator[oid]
            action = state._action_by_operator[oid]
            op = sampling_operator(coordinator, problem, state, oid)
            semantic = oid in problem.ops
            limit = coordinator.sample_limit(problem, op)
            builder.limit = limit
            builder.role = (oid, 0)
            algebraic_projection = semantic and "PROJECTION" in op["kind"].upper()
            passthrough = op.get("sampling_passthrough_input")
            cardinality_boundary = (
                not coordinator.sample_depth_selectivity
                and not semantic
                and (
                    op["kind"].upper() in ("PROJECTION", "CTE_SCAN")
                    or passthrough is not None
                )
            )
            estimate_only_relation = (
                not coordinator.sample_depth_selectivity and not semantic
            )
            try:
                if cardinality_boundary:
                    # The mapping itself has selectivity one. Predicate/join
                    # consumers materialize their own required producer chain;
                    # output aliases alone must not trigger semantic requests.
                    child = node.children[passthrough if passthrough is not None else 0]
                    required = (
                        field_tokens(op.get("required_columns", []))
                        if getattr(coordinator, "ensure_prefix_depth_rates", True)
                        else set()
                    )
                    inputs = (builder.materialize(child, required, limit=limit),)
                    rows = tuple(inputs[0])
                    candidates = units(coordinator, inputs, oid, limit=limit)
                    successes, complete = (True,) * len(candidates), True
                elif estimate_only_relation:
                    inputs = builder.operator(node, op)
                    rows = combine(
                        coordinator, inputs, [oid, "relational-estimate"], limit=limit
                    )
                    candidates = units(coordinator, inputs, oid, limit=limit)
                    successes, complete = (True,) * len(candidates), True
                elif algebraic_projection:
                    inputs = Inputs(
                        coordinator,
                        problem,
                        state,
                        projections=coordinator.sample_depth_selectivity
                        or getattr(coordinator, "ensure_prefix_depth_rates", True),
                    ).operator(node, op)
                    rows = tuple(inputs[0]) if inputs else ()
                    candidates = units(coordinator, inputs, oid, limit=limit)
                    successes, complete = (True,) * len(candidates), True
                elif passthrough is not None:
                    inputs = (
                        builder.materialize(
                            node.children[passthrough],
                            field_tokens(op.get("required_columns", [])),
                            limit=limit,
                        ),
                    )
                    rows = tuple(inputs[0])
                    candidates = units(coordinator, inputs, oid, limit=limit)
                    successes, complete = (True,) * len(candidates), True
                else:
                    inputs = builder.operator(node, op)
                    rows, candidates, successes, complete = evaluate(
                        coordinator, problem, op, inputs
                    )
            except ProjectionBudgetExhausted:
                inputs, rows, candidates, successes, complete = (), (), (), (), False
            weights = [
                math.prod(coordinator._weight(row) for side in unit for row in side)
                for unit in candidates
            ]
            total = sum(weights)
            rate = (
                1.0
                if algebraic_projection or cardinality_boundary
                else (
                    sum(w for w, passed in zip(weights, successes) if passed) / total
                    if total and complete
                    else None
                )
            )
            side_rows = [cardinalities[child] for child in node.children]
            expected = (
                math.prod(side_rows)
                if all(value is not None for value in side_rows)
                else None
            )
            if passthrough is not None:
                rate = side_rows[passthrough] / expected if expected else None
            if (
                rate is None
                and complete
                and inputs
                and any(not group for group in inputs)
            ):
                # An evaluated empty input proves no match; predecessor
                # cardinality estimates alone cannot establish this fact.
                rate = 0.0
            relational = {}
            empirical_rate = rate
            if not semantic:
                from planning.cost.estimates import propagate

                prior = dict(
                    problem.profile.get("node_statistics", {}).get(
                        str(node.node_id), {}
                    )
                )
                kind = op["kind"].upper()
                algebraic = (
                    cardinality_boundary
                    or kind in ("PROJECTION", "CTE_SCAN", "CROSS_PRODUCT", "ORDER_BY")
                    or op.get("scalar_aggregate")
                    or kind == "LIMIT"
                    and op.get("limit") is not None
                )
                estimator = getattr(coordinator.backend, "estimate_relation", None)
                if estimator is not None and complete and not algebraic:
                    prior.update(estimator(op, inputs))
                relational = propagate(op, prior, side_rows)
                rate = relational.get("selectivity")
            facts = {
                "reached": len(candidates),
                "selectivity": rate,
                "relational": relational,
                "semantic": semantic,
                "input_rows": expected,
                "output_rows": (
                    side_rows[passthrough]
                    if passthrough is not None
                    else (
                        0.0
                        if expected == 0
                        else (
                            expected * rate
                            if expected is not None and rate is not None
                            else None
                        )
                    )
                ),
                "side_rows": side_rows,
                "input_complete": complete,
                "output_complete": complete,
                "child_relations": [
                    min(state._nodes_by_id[c].relations) for c in node.children
                ],
                "sample_model": (
                    coordinator._sampling_model(problem, op) if semantic else None
                ),
                "independent_depths": {}
                if estimate_only_relation
                else depth_rates(coordinator, node, candidates, successes, complete),
            }
            info, boundary = coordinator._sampled_statistics(
                node, op, inputs, rows, facts
            )
            info.update(
                sampling_strategy="independent_base",
                refill_rounds=0,
                population_census=False,
                finite_zero=False,
                depth_basis="independent base samples by serialized field prefix character depth",
            )
            if algebraic_projection:
                info["projection_values_materialized"] = False
            if cardinality_boundary:
                info["cardinality_boundary_algebraic"] = True
            if estimate_only_relation:
                info["relational_estimate_only"] = True
                info["selectivity_basis"] = info.get("cardinality_basis")
            info["generated_field_identities"] = generated_field_identities(op)
            if semantic or getattr(coordinator, "ensure_prefix_depth_rates", True):
                fallback_inputs = value_producer_inputs(
                    problem, state, node, outputs, observations
                )
                info["input_field_summaries"] = input_evidence(
                    coordinator, op, inputs, fallback_inputs=fallback_inputs
                )
            if not semantic and "input_field_summaries" in info:
                required = {
                    item["survival_field"]
                    for item in info["input_field_summaries"].values()
                }
                info["depth_selectivity_statistics"] = {
                    field: table
                    for field, table in info["depth_selectivity_statistics"].items()
                    if field in required
                }
            if getattr(coordinator, "ensure_prefix_depth_rates", True):
                complete_prefix_selectivity(info)
            info["generated_input_survival"] = generated_survivals(
                state, observations, node, op, local_prefix_selectivity(info)
            )
            incoming = estimate(coordinator, problem, node, op, observations, info)
            if semantic:
                info.update(
                    coordinator._cache_inputs(
                        problem,
                        op,
                        action,
                        inputs,
                        complete,
                        estimated_survivals=incoming,
                    )
                )
            if semantic:
                info["sampled_selectivity"] = rate
                info["sampled_selectivity_complete"] = complete and rate is not None
            else:
                info.setdefault("sampled_selectivity", empirical_rate)
                info.setdefault(
                    "sampled_selectivity_complete",
                    (
                        not estimate_only_relation
                        and complete
                        and empirical_rate is not None
                    ),
                )
            bank[str(oid)] = {
                "selectivity": rate,
                "depth_selectivity_statistics": info["depth_selectivity_statistics"],
                "sampled_selectivity_complete": info["sampled_selectivity_complete"],
                "sample_input_rows": len(candidates),
                "sampling_strategy": "independent_base",
                "selectivity_scope": "operator",
                "sample_model_id": facts["sample_model"],
                "sample_layout": op.get("default_layout") if semantic else None,
                "generated_field_identities": info["generated_field_identities"],
            }
            for name in (
                "prefix_selectivity_statistics",
                "input_field_summaries",
                "input_width",
                "output_width",
                "input_field_statistics",
                "output_field_statistics",
                "prompt_input_characters",
                "left_data_characters",
                "right_data_characters",
            ):
                if name in info:
                    bank[str(oid)][name] = freeze_evidence(info[name])
        outputs[node.node_id] = rows
        cardinalities[node.node_id] = info["output_rows"]
        stats[key] = _publish_observation(state.statistics.get(key, {}), info)
        observations[node.node_id] = observation(
            (ASSUMPTION, oid), (coordinator, problem), rows, info, complete, boundary
        )
    stats.update(
        copy_statistics(
            {key: value for key, value in state.statistics.items() if key not in stats}
        )
    )
    return replace(
        state, statistics=stats, sample=outputs, sample_observations=observations
    )
