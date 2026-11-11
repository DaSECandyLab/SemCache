"""Native-v1 socket adapter for the rebuilt §4.1–4.3 planner."""

import copy
import json
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

from planning.learning.devices import load_inference_model
from planning.search.beam import ParetoBeamSearch, SearchConfig
from planning.state import SearchProblem, query_key


class PlanningPolicy:
    def __init__(
        self,
        checkpoint=None,
        profiles=None,
        config=None,
        preference="fastest",
        log=None,
        sampler_factory=None,
        inference_settings=None,
        runtime_adaptation=None,
    ):
        config = config or (
            SearchConfig.configured(inference_settings)
            if inference_settings
            else SearchConfig()
        )
        config = replace(config, preference=preference)
        self.search = ParetoBeamSearch(
            None
            if config.request_guided
            else load_inference_model(checkpoint, inference_settings),
            config,
        )
        self.profiles = copy.deepcopy(profiles or {})
        self.preference, self.log = preference, Path(log) if log else None
        self.last_result = None
        self.query = None
        self.sampler_factory = sampler_factory
        self.runtime_sessions = {}
        self.runtime_history = None
        from planning.execution.adaptation_monitor import AdaptationSettings

        self.adaptation_settings = runtime_adaptation or AdaptationSettings()
        self.runtime_adapters = {}
        self.runtime_ranker = self.search.ranker
        if self.adaptation_settings.enabled:
            if self.runtime_ranker is None:
                self.runtime_ranker = load_inference_model(
                    checkpoint, inference_settings
                )
            if not self.runtime_ranker or not self.runtime_ranker.value_available:
                raise ValueError(
                    "Runtime adaptation requires a trained value-model checkpoint"
                )

    def optimize(self, query):
        if query.get("kind") == "RuntimeRequest":
            return self.reoptimize(query)
        self.query = query
        from planning.settings.modes import profile_for_mode

        profile = profile_for_mode(
            self.profiles.get(query_key(query), {}),
            query,
            self.search.config.plan_generation,
            fixed_model_id=self.search.config.fixed_model_id or "gemma4-e2b",
        )
        problem = SearchProblem(query, profile)
        session = (
            self.sampler_factory(problem) if self.sampler_factory else nullcontext(None)
        )
        with session as sampler:
            problem.sampler = sampler
            result = self.search.run(problem)
            if sampler is not None:
                result["sampling"] = dict(sampler.calls)
            selected_state = (
                self.search.choose_state(problem, result, self.preference)
                if result["states"]
                else None
            )
            if (
                self.search.uses_value_model or self.adaptation_settings.enabled
            ) and selected_state is not None:
                from planning.execution.feedback import freeze_problem

                frozen, _ = freeze_problem(problem, [selected_state])
                self.runtime_sessions[query["query_id"]] = (
                    copy.deepcopy(query),
                    frozen.profile,
                )
                if len(self.runtime_sessions) > 32:
                    oldest = next(iter(self.runtime_sessions))
                    del self.runtime_sessions[oldest]
                    adapter = self.runtime_adapters.pop(oldest, None)
                    if adapter is not None:
                        adapter.close()
        self.last_result = {k: v for k, v in result.items() if k != "states"}
        if result["constraint_satisfied"] is False and not result.get(
            "accuracy_fallback"
        ):
            print(result["reason"], file=sys.stderr, flush=True)
        if self.log:
            with self.log.open("a") as output:
                output.write(
                    json.dumps(
                        dict(query_key=problem.key, **self.last_result),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
        if selected_state is None:
            raise ValueError(result["reason"])
        return problem.spec(selected_state)

    def reoptimize(self, request):
        from planning.execution.runtime import optimize_runtime
        from planning.observations.cache import CacheHistory

        session = self.runtime_sessions.get(request.get("query_id"))
        if session is None:
            raise ValueError("Runtime query has no initial value-model session")
        query, profile = session
        if query != request.get("query_ir"):
            raise ValueError("Runtime request changed its bound query")
        if request.get("chunk_policy"):
            if not self.adaptation_settings.enabled:
                return {
                    "version": 1,
                    "kind": "RuntimeUpdate",
                    "query_id": request["query_id"],
                    "updates": [],
                    "diagnostics": {"status": "runtime_adaptation_disabled"},
                }
            if query["query_id"] not in self.runtime_adapters:
                from planning.execution.adaptation import ChunkAdaptationSession

                self.runtime_adapters[query["query_id"]] = ChunkAdaptationSession(
                    query,
                    profile,
                    self.runtime_ranker,
                    self.search.config,
                    self.adaptation_settings,
                )
            updates, diagnostics = self.runtime_adapters[query["query_id"]].check(
                request
            )
            return {
                "version": 1,
                "kind": "RuntimeUpdate",
                "query_id": request["query_id"],
                "updates": updates,
                "diagnostics": diagnostics,
            }
        if self.runtime_history is None:
            self.runtime_history = getattr(
                self.sampler_factory, "history", None
            ) or CacheHistory(":memory:")
        updates = optimize_runtime(
            self.search.ranker,
            query,
            profile,
            request,
            self.runtime_history,
            self.search.config,
        )
        return {
            "version": 1,
            "kind": "RuntimeUpdate",
            "query_id": request["query_id"],
            "updates": updates,
        }

    def close(self):
        for adapter in self.runtime_adapters.values():
            adapter.close()
        self.runtime_adapters.clear()
        shared_history = getattr(self.sampler_factory, "history", None)
        if (
            self.runtime_history is not None
            and self.runtime_history is not shared_history
        ):
            self.runtime_history.close()
        close = getattr(self.sampler_factory, "close", None)
        if close is not None:
            close()
        self.runtime_sessions.clear()
