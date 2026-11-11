"""Explicit reference Beam loop for offline differential verification."""

import math
import random
import time

from planning.state import ActionBudgetExceeded

from .heuristics import action_priorities, fixed_model
from .pareto import pareto_fronts, trim


def run_reference(search, problem):
    """Explicit Python controller; production search remains native."""
    search.require_model(problem)
    if problem.sampler is not None:
        from planning.kernels import _search_core

        if not isinstance(
            getattr(problem.sampler, "native_sampler", None),
            _search_core.ProfileEnvironment,
        ):
            raise ValueError(
                "Reference Beam search accepts frozen statistics only; "
                "collect independent operator statistics before searching"
            )
    search._fixed_model_id = (
        fixed_model(problem, search.config.fixed_model_id)
        if search.config.request_guided
        else None
    )
    search._resident_scores = None
    search._resident_choices = {}
    search._completion_values = {}
    start, beam, completed, original_candidate_error = search._start(problem)
    search.remember_completions(problem, completed)
    expansions, depth = 0, 0
    rng = random.Random(search.config.seed)
    exhausted = action_budget_exhausted = False
    priorities = action_priorities(problem, search.config.priority_mode)
    while beam:
        candidates = {}
        new_completions = []
        for parent in beam:
            if (
                expansions >= search.config.max_expansions
                or time.monotonic() - start >= search.config.max_seconds
            ):
                exhausted = True
                break
            try:
                actions = problem.actions(
                    parent,
                    fixed_layout=search.config.request_guided,
                    highest_priorities=priorities,
                    fixed_model_id=search._fixed_model_id,
                )
            except ActionBudgetExceeded:
                exhausted = action_budget_exhausted = True
                break
            for action in actions:
                if (
                    expansions >= search.config.max_expansions
                    or time.monotonic() - start >= search.config.max_seconds
                ):
                    exhausted = True
                    break
                child = problem.transition(parent, action)
                expansions += 1
                if problem.complete(child):
                    completed.append(child)
                    new_completions.append(child)
                else:
                    candidates.setdefault(child.history, child)
            if exhausted:
                break
        states = list(candidates.values())
        search.remember_completions(problem, new_completions)
        if not states:
            break
        search.refresh_context(problem, states)
        prediction = search.predictions(problem, states)
        scores, excluded = prediction["scores"], []
        order = []
        cost_pruned = 0
        ranking_diagnostics = {"accuracy_threshold": None, "accuracy_relaxations": 0}
        if not search.config.training and search.uses_value_model:
            from .ranking import rank_layer

            scores, excluded = rank_layer(
                prediction,
                search.config.preference,
                problem.constraints.get("a_min"),
                diagnostics=ranking_diagnostics,
            )
            if search.config.layer_pruning == "accuracy_time":
                from .ranking import accuracy_time_order

                order = accuracy_time_order(prediction)
            elif search.config.layer_pruning == "grouped":
                from .grouped import grouped_order

                order = grouped_order(problem, states, prediction, search.config)
        if search.config.uses_join_filter_cost_pruning:
            from planning.kernels import _search_core

            statistics = [
                {
                    key: value
                    for key, value in state.statistics.get(
                        state._action_keys[-1], {}
                    ).items()
                    if key in ("input_rows", "output_rows", "input_side_rows")
                }
                for state in states
            ]
            pruned = _search_core.prune_join_filter_cost(
                problem.native_problem,
                [state.native_handle for state in states],
                statistics,
                problem.profile.get("operator_time_calibration", {}),
            )
            excluded = pruned["excluded"]
            cost_pruned = len(excluded)
        if len(scores) != len(states):
            raise ValueError("PlanValueModel returned the wrong number of scores")
        eligible = [i for i in range(len(states)) if i not in set(excluded)]
        eligible_set = set(eligible)
        fronts = (
            [[i for i in order if i in eligible_set]]
            if order
            else [
                [eligible[i] for i in front]
                for front in pareto_fronts([scores[i] for i in eligible])
            ]
        )
        slots = (
            math.ceil(search.config.beam_width * search.config.exploration_fraction)
            if search.config.training and len(states) > search.config.beam_width
            else 0
        )
        keep = (
            [i for i in order if i in eligible_set][: search.config.beam_width - slots]
            if order
            else trim(scores, search.config.beam_width - slots, fronts=fronts, rng=rng)
        )
        remaining = [i for i in eligible if i not in set(keep)]
        keep.extend(rng.sample(remaining, min(slots, len(remaining))))
        beam = [
            problem.update_accuracy(states[i]) if search.uses_value_model else states[i]
            for i in keep
        ]
        search.trace.append(
            {
                "depth": depth + 1,
                "enumerated": len(states),
                "retained": len(keep),
                "complete": len(completed),
                "fronts": [len(f) for f in fronts],
                "quality_pruned": len(excluded) - cost_pruned,
                "cost_pruned": cost_pruned,
                "heuristic_guard": 0,
                "random_exploration": slots,
                **ranking_diagnostics,
            }
        )
        depth += 1
        if exhausted:
            break
    search._guard_result = {"status": "disabled", "plan": None}
    return search._finish(
        problem,
        completed,
        exhausted,
        action_budget_exhausted,
        expansions,
        original_candidate_error,
    )
