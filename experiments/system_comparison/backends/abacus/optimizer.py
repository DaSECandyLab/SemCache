"""Native Abacus MAB sampling, Validator, sampled cost model and Pareto selection."""

from __future__ import annotations

import logging
import math
import json
import signal
import threading
import time
from contextlib import contextmanager
from copy import copy
from pathlib import Path
from unittest.mock import patch

from experiments.core.io import save
from experiments.core.paths import ROOT
from experiments.system_comparison.backends.abacus.compat import native_join_sampling
from experiments.system_comparison.reporting.timing import QueryTimer, stage

logger = logging.getLogger(__name__)


def quality_evidence(job, policy, plan):
    """Record the native estimate and reject its unconstrained fallback when requested."""
    if job.get("quality_threshold") is None:
        return {}
    quality = float(plan.plan_cost.quality)
    return {
        "quality_threshold": float(job["quality_threshold"]),
        "estimated_quality": quality,
        "quality_constraint_satisfied": math.isfinite(quality) and bool(policy.constraint(plan.plan_cost)),
        "quality_metric": "native plan estimate, not measured reference F1",
    }


def fresh_dataset(dataset):
    """Copy native operators while preserving indexable MemoryDataset roots.

    Dataset.copy() lowers even MemoryDataset to Dataset, which has no len().
    A fresh shallow root keeps frozen values shared and leaves schemas and
    operator state private to the new processor.
    """
    result = copy(dataset)
    result._operator = dataset._operator.copy()
    result._sources = [fresh_dataset(source) for source in dataset._sources]
    return result


class AbacusSearchTimeout(TimeoutError):
    """External experiment watchdog for the native Abacus planning phase."""


def stop_on_search_timeout(job):
    """Read the active campaign's policy without altering other campaigns."""
    for parent in Path(job["output"]).resolve().parents:
        control = parent / "control.json"
        if (parent / "campaign.json").is_file() and control.is_file():
            return bool(json.loads(control.read_text()).get("abacus_stop_on_search_timeout", False))
    return False


@contextmanager
def external_search_deadline(seconds):
    """Interrupt only the experiment worker's main-thread optimizer call."""
    if seconds is None:
        yield
        return
    limit = float(seconds)
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("Abacus search limit must be a finite positive number")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Abacus search watchdog requires the worker main thread")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[0]:
        raise RuntimeError("Abacus watchdog cannot replace an active real-time timer")

    def alarm(_signum, _frame):
        raise AbacusSearchTimeout(f"external Abacus planning limit {limit:g}s")

    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, limit)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


@contextmanager
def judge_transport(model):
    import litellm

    original = litellm.completion

    def completion(*args, **kwargs):
        if kwargs.get("model") in {model.value, "openai/o4-mini"}:
            kwargs["model"] = model.value
            kwargs.setdefault("api_base", model.api_base)
            kwargs.setdefault("api_key", "EMPTY")
            for name, value in model.vllm_kwargs.items():
                kwargs.setdefault(name, value)
        return original(*args, **kwargs)

    with patch.object(litellm, "completion", completion):
        yield


def run_without_sampling(dataset, job, *, config=None):
    """Use native Dataset.run with prior costs, without training or Validator."""
    from palimpzest.query.optimizer.optimizer import Optimizer
    from palimpzest.query.processor.query_processor import QueryProcessor

    from experiments.system_comparison.execution.worker import load_module

    formal = load_module(
        "comparison_abacus_unsampled_plan_tools", ROOT / "systems/abacus/formal.py"
    )
    config = (config or formal.native_config(job)).copy()
    config.sentinel_execution_strategy = None
    if config.optimizer_strategy != "pareto":
        raise ValueError("Abacus requires native Pareto search")
    folder = Path(job["output"])
    evidence = {
        "system": "abacus",
        "policy": type(config.policy).__name__,
        "entrypoint": "Dataset.run",
        "sampling_enabled": False,
        "sentinel": None,
        "validator": None,
        "train_dataset": None,
        "optimizer": "pareto",
        "sampled_cost_models": 0,
        "physical_selections": 0,
        "fallback_used": False,
        "available_models": [model.value for model in config.available_models],
        "search_timeout_seconds": None,
        "whole_query_timeout_seconds": job.get("timeout", 1800),
        "optimizer_settings": {
            name: getattr(config, name)
            for name in (
                "allow_bonded_query",
                "allow_rag_reduction",
                "allow_mixtures",
                "allow_critic",
                "allow_split_merge",
                "optimizer_strategy",
                "sentinel_execution_strategy",
            )
        },
        "searches": [],
        "completed": False,
    }
    original_optimize = Optimizer.optimize
    original_execute = QueryProcessor._execute_best_plan

    def optimize(self, *args, **kwargs):
        started = time.monotonic()
        try:
            with stage(
                "planning", "native Abacus optimization without MAB or sampling"
            ):
                plans = original_optimize(self, *args, **kwargs)
            if plans and hasattr(plans[0], "operator"):
                evidence["physical_selections"] += 1
                save(
                    folder / "plan.json",
                    {
                        "phase": "native_no_mab_no_sampling",
                        "policy": type(config.policy).__name__,
                        **quality_evidence(job, config.policy, plans[0]),
                        "plan": formal.describe_plan(plans[0]),
                        "native_plan_text": str(plans[0]),
                    },
                )
                # Keep the native estimate as evidence, but do not turn an
                # infeasible quality target into an execution error. Abacus's
                # policy still guides selection when feasible plans exist; if
                # none satisfy it, the native Pareto choice is executable and
                # its unmet estimate is recorded in plan.json.
            return plans
        finally:
            evidence["searches"].append(
                {
                    "mode": "native_no_mab_no_sampling",
                    "seconds": time.monotonic() - started,
                    "timeout_seconds": None,
                }
            )
            save(folder / "mechanism.json", evidence)

    def execute(self, *args, **kwargs):
        with stage("execution", "native Abacus final plan without sampling"):
            return original_execute(self, *args, **kwargs)

    save(folder / "mechanism.json", evidence)
    try:
        with (
            QueryTimer(folder),
            patch.object(Optimizer, "optimize", optimize),
            patch.object(QueryProcessor, "_execute_best_plan", execute),
        ):
            result = fresh_dataset(dataset).run(config=config)
        save(
            folder / "native-stats.json", result.execution_stats.model_dump(mode="json")
        )
        evidence["completed"] = True
        return result
    finally:
        save(folder / "mechanism.json", evidence)


def optimize_and_run(dataset, job, *, config=None, train_dataset=None):
    if job.get("abacus_optimizer", {}).get("sampling_enabled", True) is False:
        return run_without_sampling(dataset, job, config=config)
    import palimpzest as pz
    from palimpzest.query.execution.mab_execution_strategy import MABExecutionStrategy
    from palimpzest.query.optimizer.cost_model import SampleBasedCostModel
    from palimpzest.query.optimizer.optimizer import Optimizer
    from palimpzest.query.processor.query_processor import QueryProcessor

    from experiments.system_comparison.execution.worker import load_module

    formal = load_module(
        "comparison_abacus_plan_tools", ROOT / "systems/abacus/formal.py"
    )
    config = config or formal.native_config(job)
    # Keep a pristine config/tree for a fresh native optimization after a
    # timeout. The factory relaxes schemas in place during the primary run.
    fallback_config = config.copy()
    options = job.get("abacus_optimizer", {})
    search_limit = options.get("max_search_seconds", 100)
    config.sentinel_execution_strategy = "mab"
    for name in ("sample_budget", "k", "j", "seed"):
        if name in options:
            setattr(config, name, int(options[name]))
    if config.optimizer_strategy != "pareto":
        raise ValueError("Abacus requires native Pareto search")
    pool = formal.model_pool(job)
    alias = options.get("judge_model", "semantic-large")
    if alias not in pool:
        raise ValueError("Internal judge must be in the configured local pool")
    judge = pool[alias]
    validator = pz.Validator(model=judge)
    folder = Path(job["output"])
    evidence = {
        "system": "abacus",
        "policy": type(config.policy).__name__,
        "entrypoint": "Dataset.optimize_and_run",
        "sentinel": "mab",
        "optimizer": "pareto",
        "sample_budget": config.sample_budget,
        "k": config.k,
        "j": config.j,
        "judge_model": alias,
        "sampled_cost_models": 0,
        "physical_selections": 0,
        "static_fallback_allowed": False,
        "native_no_sampling_fallback_allowed": True,
        "judge_role": "internal quality estimates, not final reference labels",
        "search_timeout_seconds": search_limit,
        "timeout_controller": "experiments.system_comparison.backends.abacus.optimizer",
        "timeout_scope": "each native optimizer search; sampling and final execution timed separately",
        "fallback_used": False,
        "searches": [],
    }
    evidence["optimizer_settings"] = {
        name: getattr(config, name)
        for name in (
            "allow_bonded_query",
            "allow_rag_reduction",
            "allow_mixtures",
            "allow_critic",
            "allow_split_merge",
            "optimizer_strategy",
            "sentinel_execution_strategy",
        )
    }
    evidence["relational_join_compatibility"] = {
        "sentinel_retain_inputs": False,
        "deterministic_quality": 1.0,
        "semantic_quality": "unmodified native Validator",
    }
    save(folder / "mechanism.json", evidence)
    original_cost = SampleBasedCostModel.__init__
    original_optimize = Optimizer.optimize
    original_sentinel = MABExecutionStrategy.execute_sentinel_plan
    original_execute = QueryProcessor._execute_best_plan

    def cost_init(self, *args, **kwargs):
        original_cost(self, *args, **kwargs)
        stats = args[0] if args else kwargs.get("sentinel_plan_stats")
        if stats is not None:
            evidence["sampled_cost_models"] += 1
            save(folder / "sentinel-stats.json", stats.model_dump(mode="json"))
            save(folder / "mechanism.json", evidence)

    def record_plan(plans, phase):
        if plans and hasattr(plans[0], "operator"):
            evidence["physical_selections"] += 1
            save(
                folder / "plan.json",
                {
                    "phase": phase,
                    "policy": type(config.policy).__name__,
                    **quality_evidence(job, config.policy, plans[0]),
                    "plan": formal.describe_plan(plans[0]),
                    "native_plan_text": str(plans[0]),
                },
            )
            # An unmet native quality estimate is diagnostic metadata only.
            # Execute the selected Pareto plan instead of rejecting it here.

    def optimize(self, *args, **kwargs):
        started = time.monotonic()
        event = {"mode": "mab", "timeout_seconds": search_limit, "timed_out": False}
        try:
            with (
                stage("planning", "native Abacus memo and Pareto search"),
                external_search_deadline(search_limit),
            ):
                plans = original_optimize(self, *args, **kwargs)
        except AbacusSearchTimeout:
            event["timed_out"] = True
            raise
        finally:
            event["seconds"] = time.monotonic() - started
            evidence["searches"].append(event)
            save(folder / "mechanism.json", evidence)
        record_plan(plans, "sampled_final_selection")
        return plans

    def sentinel(self, *args, **kwargs):
        with stage("sampling", "native Abacus MAB and Validator"):
            return original_sentinel(self, *args, **kwargs)

    def execute(self, *args, **kwargs):
        with stage("execution", "native Abacus final plan"):
            return original_execute(self, *args, **kwargs)

    timed_out = False
    try:
        with QueryTimer(folder):
            try:
                with (
                    judge_transport(judge),
                    native_join_sampling(),
                    patch.object(SampleBasedCostModel, "__init__", cost_init),
                    patch.object(Optimizer, "optimize", optimize),
                    patch.object(
                        MABExecutionStrategy, "execute_sentinel_plan", sentinel
                    ),
                    patch.object(QueryProcessor, "_execute_best_plan", execute),
                ):
                    result = fresh_dataset(dataset).optimize_and_run(
                        config=config, validator=validator, train_dataset=train_dataset
                    )
            except AbacusSearchTimeout as error:
                timed_out = True
                if stop_on_search_timeout(job):
                    evidence["search_timeout"] = {
                        "seconds": float(search_limit),
                        "error": str(error),
                        "fallback": None,
                    }
                    evidence["fallback_used"] = False
                    evidence["terminated_on_search_timeout"] = True
                    save(folder / "mechanism.json", evidence)
                    logger.warning("%s; ending query without fallback", error)
                    raise
                evidence["search_timeout"] = {
                    "seconds": float(search_limit),
                    "error": str(error),
                    "fallback": "native_no_mab_no_sampling",
                }
                evidence["fallback_used"] = True
                logger.warning(
                    "%s; reoptimizing from scratch with Dataset.run, no MAB or sampling",
                    error,
                )
                save(folder / "mechanism.json", evidence)

            if timed_out:
                fallback_config.sentinel_execution_strategy = None
                # Dataset.run constructs a new native processor and a prior
                # cost model. No validator, training set, sampled memo or
                # manually assembled plan is passed to this call.
                evidence["fallback_config"] = {
                    "entrypoint": "Dataset.run",
                    "sentinel_execution_strategy": None,
                    "optimizer_strategy": fallback_config.optimizer_strategy,
                    "policy": type(fallback_config.policy).__name__,
                    "available_models": [
                        model.value for model in fallback_config.available_models
                    ],
                    "validator": None,
                    "train_dataset": None,
                    "sampled_cost_model_reused": False,
                }
                save(folder / "mechanism.json", evidence)

                def fallback_optimize(self, *args, **kwargs):
                    started = time.monotonic()
                    with stage(
                        "planning", "native Abacus no-MAB/no-sampling optimization"
                    ):
                        plans = original_optimize(self, *args, **kwargs)
                    evidence["searches"].append(
                        {
                            "mode": "native_no_mab_no_sampling",
                            "seconds": time.monotonic() - started,
                            "timed_out": False,
                            "timeout_seconds": None,
                        }
                    )
                    record_plan(plans, "native_no_mab_no_sampling_fallback")
                    save(folder / "mechanism.json", evidence)
                    return plans

                with (
                    native_join_sampling(),
                    patch.object(Optimizer, "optimize", fallback_optimize),
                    patch.object(QueryProcessor, "_execute_best_plan", execute),
                ):
                    result = fresh_dataset(dataset).run(config=fallback_config)
                evidence["fallback_completed"] = True

        if not timed_out and (
            not evidence["sampled_cost_models"] or not evidence["physical_selections"]
        ):
            raise RuntimeError(
                "Native sampled optimization did not finish; no static fallback"
            )
        save(
            folder / "native-stats.json", result.execution_stats.model_dump(mode="json")
        )
        evidence["completed"] = True
        return result
    finally:
        save(folder / "mechanism.json", evidence)
