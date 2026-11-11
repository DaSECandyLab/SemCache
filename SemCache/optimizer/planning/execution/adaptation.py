"""Query-scoped, fixed-topology runtime adaptation in the optimizer process."""

import copy
import math
import time
from dataclasses import replace

import torch

from planning.execution.adaptation_history import RuntimeHistoryReader
from planning.execution.adaptation_monitor import drift_checks, plan_triggers
from planning.execution.cache_prediction import CacheTokenMLP, reuse_features
from planning.execution.runtime import optimize_runtime, runtime_snapshot
from planning.observations.cache import CacheHistory


class ChunkAdaptationSession:
    def __init__(
        self,
        query,
        profile,
        model,
        config,
        settings,
        initial_references=None,
    ):
        if initial_references is not None:
            from planning.execution.adaptation_profile import compact_profile

            profile = compact_profile(profile)
        self.query, self.profile, self.model = query, profile, model
        self.config = replace(
            config,
            preference="fastest",
            beam_width=settings.beam_width,
            workers=1,
            layer_pruning="pareto",
            max_seconds=4.0,
        )
        self.settings = settings
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu")
            if settings.device == "auto"
            else settings.device
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "Runtime adaptation requires CUDA, but CUDA is unavailable"
            )
        self.history = CacheHistory(":memory:")
        self.reader = RuntimeHistoryReader()
        self.cache_model = (
            CacheTokenMLP.load(settings.cache_predictor_checkpoint, device=self.device)
            if settings.cache_predictor_checkpoint
            else None
        )
        self.references = {}
        self.last_chunks = 0
        self.last_checkpoint = 0
        self.initial_selectivities = {
            oid: reference["selectivity"]
            for oid, reference in (initial_references or {}).items()
        }
        for oid, reference in (initial_references or {}).items():
            features = reference["prefix_features"]
            self.references[oid, 0] = {
                **reference,
                "cached_tokens": self.cache_model.predict([features])[0]
                if self.cache_model
                and features
                and all(v is not None for v in features.values())
                else None,
            }

    def close(self):
        self.reader.close()
        self.history.close()

    def _prediction(self, features):
        result = self.model.completion_prediction([features])
        pair = result["times"][0], result["accuracies"][0]
        if (
            not all(type(v) in (int, float) and math.isfinite(v) for v in pair)
            or pair[0] < 0
            or not 0 <= pair[1] <= 1
        ):
            raise ValueError("Invalid runtime value prediction")
        return pair

    def _references(self, request, entries):
        missing = [
            oid
            for oid, e in entries.items()
            if (oid, e["current"]["config_generation"]) not in self.references
            or self.cache_model is not None
            and self.references.get((oid, e["current"]["config_generation"]), {}).get(
                "cached_tokens"
            )
            is None
        ]
        if not missing:
            return
        # The expected value is frozen per implementation generation. Actual
        # cache observations must not continually move their own drift baseline.
        problem, state, graph = runtime_snapshot(
            self.query,
            self.profile,
            request,
            None,
            observed=bool(self.initial_selectivities)
            and any(
                entries[oid]["current"]["config_generation"] > 0 for oid in missing
            ),
        )
        for oid in missing:
            key = oid, entries[oid]["current"]["config_generation"]
            op = problem.ops[oid]
            node = graph["nodes"][op["node_id"]]
            action = state._action_by_operator[oid]
            features = reuse_features(
                node,
                op,
                action,
                problem.profile,
                include_tokens=self.cache_model is not None
                and self.cache_model.feature_mode == "operator_tokens",
            )
            prediction = (
                self.cache_model.predict([features])[0]
                if self.cache_model and all(v is not None for v in features.values())
                else None
            )
            action_key = state._action_keys_by_operator[oid]
            stat = state.statistics[action_key]
            selection = (
                stat.get("estimated_selectivity")
                if stat.get("shared_selectivity")
                else stat.get("selectivity")
            )
            self.references[key] = {
                "selectivity": self.initial_selectivities.get(oid, selection),
                "cached_tokens": prediction,
                "prefix_features": features,
            }

    def check(self, request):
        if (
            request.get("chunk_policy") is not True
            or request.get("query_ir") != self.query
        ):
            raise ValueError("Chunk adaptation request identity mismatch")
        completed = request["completed_chunks"]
        if type(completed) is not int or completed < self.last_chunks:
            raise ValueError("Runtime chunk sequence moved backwards")
        diagnostics = {
            "completed_chunks": completed,
            "triggers": [],
            "operators": {},
            "statistics_source": "actual processed execution chunks",
            "sampling_requests": 0,
            "beam_width": self.config.beam_width,
            "inference_device": str(self.device),
            "topology_search": False,
        }
        checkpoint = request["progress_checkpoint"]
        index = checkpoint["index"]
        limit = self.settings.max_switches
        if (
            request["max_switches"] != limit
            or checkpoint["count"] != limit
            or type(index) is not int
            or not 1 <= index <= limit
            or not math.isclose(checkpoint["target"], index / (limit + 1))
            or not 0 <= checkpoint["observed"] <= 1
        ):
            raise ValueError("Invalid runtime progress checkpoint")
        diagnostics["progress_checkpoint"] = checkpoint
        if index <= self.last_checkpoint:
            diagnostics["status"] = "checkpoint_already_checked"
            return [], diagnostics
        if checkpoint["observed"] + 1e-12 < checkpoint["target"]:
            diagnostics["status"] = "before_progress_checkpoint"
            return [], diagnostics
        self.last_checkpoint = index
        self.last_chunks = completed
        entries = {e["operator_id"]: e for e in request["state"]}
        switches = request["applied_switches"]
        diagnostics["applied_switches"] = switches
        diagnostics["switch_limit"] = limit
        if switches >= limit:
            diagnostics["status"] = "switch_limit_reached"
            return [], diagnostics
        timings = diagnostics["timings_seconds"] = {}
        measured = time.monotonic()
        prefixes = self.reader.consume(request, self.query, self.history)
        timings["read_execution_evidence"] = time.monotonic() - measured
        diagnostics["history_requests"] = prefixes
        history = tuple(self.history.snapshot())
        measured = time.monotonic()
        request = copy.deepcopy(request)
        request["runtime_input_statistics"] = self.reader.statistics(entries)
        diagnostics["operators_with_actual_input_statistics"] = sorted(
            request["runtime_input_statistics"]
        )
        for observation in request["observations"]:
            actual_inputs = request["runtime_input_statistics"].get(
                str(observation["operator_id"])
            )
            if (
                actual_inputs
                and observation["config_generation"]
                == actual_inputs["config_generation"]
            ):
                observation["input_statistics"] = actual_inputs
            statistics = request["operator_statistics"].get(
                str(observation["operator_id"])
            )
            if (
                statistics
                and statistics["config_generation"] == observation["config_generation"]
            ):
                observation["selectivity"] = (
                    statistics["emitted"] / statistics["completed"]
                )
                observation["effective_samples"] = statistics["completed"]
        self._references(request, entries)
        problem, state, features = runtime_snapshot(
            self.query, self.profile, request, history
        )
        value_before = self._prediction(features)
        timings["current_snapshot_and_prediction"] = time.monotonic() - measured
        minimum = problem.constraints.get("a_min")
        diagnostics.update(
            predicted_accuracy=value_before[1],
            accuracy_scope="predicted final query F1",
            a_min=minimum,
        )
        for oid, entry in entries.items():
            reference = self.references[oid, entry["current"]["config_generation"]]
            check = drift_checks(
                request["operator_statistics"].get(str(oid)),
                reference["selectivity"],
                reference["cached_tokens"],
                problem.ops[oid]["kind"],
                self.settings,
                generation=entry["current"]["config_generation"],
            )
            diagnostics["operators"][str(oid)] = check
        diagnostics.update(
            plan_triggers(
                diagnostics["operators"],
                value_before[1],
                minimum,
                self.settings,
            )
        )
        if not diagnostics["triggers"] or all(
            e["status"] == "DONE" for e in entries.values()
        ):
            diagnostics["status"] = (
                "stable" if not diagnostics["triggers"] else "no_unfinished_operators"
            )
            return [], diagnostics
        measured = time.monotonic()
        updates = optimize_runtime(
            self.model, self.query, self.profile, request, history, self.config
        )
        timings["beam_search"] = time.monotonic() - measured
        if not updates:
            diagnostics["status"] = "no_implementation_change"
            return [], diagnostics
        if len({u["operator_id"] for u in updates}) != len(updates):
            raise ValueError("Duplicate runtime update target")
        for update in updates:
            entry = entries[update["operator_id"]]
            if (
                entry["status"] == "DONE"
                or update["expected_generation"]
                != entry["current"]["config_generation"]
            ):
                raise ValueError(
                    "Runtime search changed a completed or stale implementation"
                )
        # Publication still belongs to DuckDB, which independently checks the
        # generation and chunk boundary after this asynchronous RPC returns.
        diagnostics["status"] = "implementation_update_proposed"
        diagnostics["proposed_operators"] = [u["operator_id"] for u in updates]
        return updates, diagnostics
