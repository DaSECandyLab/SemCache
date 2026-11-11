"""Expand and rank legal plans using shared operator statistics."""

import math
import time
from typing import Any

from planning.cost.row_time import estimated_plan_time, time_predictions
from planning.state import digest

from .config import SearchConfig as SearchConfig
from .pareto import pareto_fronts as pareto_fronts


class ParetoBeamSearch:
    def __init__(self, ranker=None, config=None):
        self.ranker, self.config = ranker, config or SearchConfig()
        self.trace = []
        self._resident_scores = None
        self._resident_choices = {}
        self._completion_values = {}
        self._guard_result = {"status": "disabled", "plan": None}

    @property
    def uses_value_model(self):
        return bool(
            not self.config.request_guided
            and self.ranker
            and self.ranker.value_available
        )

    def require_model(self, problem=None):
        if (
            self.config.request_guided
            and problem is not None
            and problem.constraints.get("a_min") is not None
        ):
            raise ValueError(
                "An accuracy constraint requires heuristic_value; heuristic_requests does not predict accuracy"
            )
        if not self.config.request_guided and not self.uses_value_model:
            raise ValueError(
                "heuristic_value requires a trained, available value model"
            )

    def scores(self, problem, states):
        return self.predictions(problem, states)["scores"]

    def predictions(self, problem, states):
        if self.uses_value_model:
            assert self.ranker is not None
            if hasattr(self.ranker, "completion_prediction"):
                from planning.learning.features.graph import encode

                return self.ranker.completion_prediction(
                    [encode(problem, state) for state in states],
                )
            scores = self.ranker.scores(problem, states)
            return {
                "scores": scores,
                "times": [-row[0] for row in scores],
                "accuracies": [row[1] for row in scores],
            }
        return self._heuristic_predictions(problem, states)

    @staticmethod
    def _heuristic_predictions(problem, states):
        from planning.cost.requests import estimated_request_count

        return time_predictions(
            states,
            lambda state: estimated_plan_time(problem, state),
            request_fallback=(
                (lambda state: estimated_request_count(problem, state))
                if problem.profile.get("cardinality_only_sources")
                else None
            ),
        )

    @staticmethod
    def _heuristic_scores(problem, states):
        prediction = ParetoBeamSearch._heuristic_predictions(problem, states)
        return prediction["scores"], prediction["times"], prediction["time_reasons"]

    @staticmethod
    def refresh_context(problem, states):
        provider = getattr(problem, "history_provider", None)
        if provider is not None:
            problem.cache_history = tuple(provider())
        for state in states:
            for action in problem.semantic_actions(state):
                state.statistics[digest(action)].update(
                    problem.estimate(
                        state, action, state.partial_plans, copy_observation=False
                    )
                )

    @staticmethod
    def complete_key(problem, state):
        schedule = (
            problem.semantic_actions(state)
            if problem.native and not problem.tree_regions
            else None
        )
        return digest([sorted(state.history), state.partial_plans.roots, schedule])

    def remember_completions(self, problem, states):
        """Capture admission values, retained for value-model final selection."""
        fresh = {}
        for state in states:
            key = self.complete_key(problem, state)
            if key not in self._completion_values:
                fresh.setdefault(key, state)
        if not fresh:
            return
        states = list(fresh.values())
        if self.uses_value_model:
            states = [problem.update_accuracy(state) for state in states]
        self.refresh_context(problem, states)
        learned = self.uses_value_model
        if learned:
            prediction = self.predictions(problem, states)
            scores = prediction["scores"]
        else:
            prediction = self._heuristic_predictions(problem, states)
            scores = prediction["scores"]
            times, reasons = prediction["times"], prediction["time_reasons"]
        if len(scores) != len(states) or any(
            len(row) != 2 or not all(math.isfinite(value) for value in row)
            for row in scores
        ):
            raise ValueError("PlanValueModel must return two finite scores per state")
        if learned:
            times = prediction["times"]
            accuracies = prediction["accuracies"]
            reasons = prediction.get("time_reasons", [None] * len(states))
        else:
            accuracies = prediction["accuracies"]
        for key, score, seconds, accuracy, reason in zip(
            fresh, scores, times, accuracies, reasons, strict=True
        ):
            self._completion_values[key] = (tuple(score), seconds, accuracy, reason)

    def _start(self, problem):
        start = time.monotonic()
        beam, completed = [problem.initial], []
        original_candidate_error = None
        if problem.complete(problem.initial):
            completed.append(problem.initial)
            beam = []
        self.trace = []
        return start, beam, completed, original_candidate_error

    def run(self, problem) -> dict[str, Any]:
        self.require_model(problem)
        from planning.kernels.search import run_native

        from .heuristics import fixed_model

        self._fixed_model_id = (
            fixed_model(problem, self.config.fixed_model_id)
            if self.config.request_guided
            else None
        )
        self._resident_scores = None
        self._resident_choices = {}
        self._completion_values = {}
        self._guard_result = {"status": "disabled", "plan": None}
        return run_native(self, problem)

    def run_python(self, problem) -> dict[str, Any]:
        from .reference import run_reference

        return run_reference(self, problem)

    def _finish(
        self,
        problem,
        completed,
        exhausted,
        action_budget_exhausted,
        expansions,
        original_candidate_error,
        *,
        deduplicated=False,
    ):
        if not deduplicated:
            unique = {}
            for state in completed:
                unique.setdefault(self.complete_key(problem, state), state)
            completed = list(unique.values())
        if self.config.request_guided and completed:
            # Match resident final selection: refresh existing estimates only,
            # then evaluate semantic work against one common final context.
            self.refresh_context(problem, completed)
            prediction = self._heuristic_predictions(problem, completed)
            complete_scores = prediction["scores"]
            times = prediction["times"]
            accuracies = prediction["accuracies"]
            time_reasons = prediction["time_reasons"]
        else:
            values = [
                self._completion_values[self.complete_key(problem, state)]
                for state in completed
            ]
            complete_scores = [value[0] for value in values]
            times = [value[1] for value in values]
            accuracies = [value[2] for value in values]
            time_reasons = [value[3] for value in values]
        minimum = (
            None if self.config.request_guided else problem.constraints.get("a_min")
        )
        from planning.kernels import _search_core

        decision = _search_core.select_completions(
            complete_scores,
            times,
            [0.0] * len(completed) if self.config.request_guided else accuracies,
            minimum,
        )
        from .results import finish_search

        return finish_search(
            self,
            problem,
            completed,
            scores=complete_scores,
            times=times,
            accuracies=accuracies,
            time_reasons=time_reasons,
            decision=decision,
            exhausted=exhausted,
            action_budget_exhausted=action_budget_exhausted,
            expansions=expansions,
            original_error=original_candidate_error,
            heuristic_guard=self._guard_result,
        )

    def choose(self, problem, result, preference="fastest"):
        return problem.spec(self.choose_state(problem, result, preference))

    def choose_state(self, problem, result, preference="fastest"):
        """Return the selected frozen state for execution and runtime feedback."""
        if preference not in ("fastest", "accurate"):
            raise ValueError("preference must be fastest or accurate")
        if (
            not self.config.training
            and self.uses_value_model
            and problem.constraints.get("a_min") is not None
        ):
            preference = "fastest"
        states = result["states"]
        if not states:
            raise ValueError(result["reason"])
        resident_scores = getattr(self, "_resident_scores", None)
        if resident_scores is not None and all(
            s.history in resident_scores for s in states
        ):
            history = self._resident_choices[preference]
            state = next(state for state in states if state.history == history)
            return state
        if preference == "fastest" and not self.uses_value_model:
            scores, times, _ = self._heuristic_scores(problem, states)
            index = min(
                range(len(states)),
                key=lambda i: (times[i] is None, times[i] or 0.0, -scores[i][0], i),
            )
            return states[index]
        scores = self.scores(problem, states)
        axis = 0 if preference == "fastest" else 1
        return max(
            enumerate(states), key=lambda pair: (scores[pair[0]][axis], -pair[0])
        )[1]
