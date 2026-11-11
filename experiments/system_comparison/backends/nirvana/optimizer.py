"""Observe native Nirvana decisions without replacing its selection algorithm."""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar
from pathlib import Path
from unittest.mock import patch

from experiments.core.io import save
from experiments.system_comparison.reporting.timing import QueryTimer, stage


def plan(node):
    op = node.operator
    return {
        "operator": node.op_name,
        "model": getattr(op, "model", None),
        "instruction": getattr(op, "user_instruction", None),
        "input_columns": getattr(op, "input_columns", None),
        "output_columns": getattr(op, "output_columns", None),
        "tool": str(getattr(op, "tool", None)),
        "fields": vars(node.node_fields),
        "children": [
            plan(child)
            for child in (node.left_child, node.right_child)
            if child is not None
        ],
    }


def optimize_and_execute(dataframe, job, options=None):
    from nirvana.optim import OptimizeConfig
    from nirvana.optim.logical import LogicalOptimizer
    from nirvana.optim.optimizer import PlanOptimizer
    from nirvana.optim.physical import PhysicalOptimizer

    config = OptimizeConfig(**(options or job.get("optimizer", {})))
    if job.get("native_mechanisms") and not (
        config.do_logical_optimization and config.do_physical_optimization
    ):
        raise ValueError("Formal Nirvana requires both native optimizers")
    folder = Path(job["output"])
    evidence = {
        "logical_optimizer": config.do_logical_optimization,
        "physical_optimizer": config.do_physical_optimization,
        "available_models": config.available_models,
        "sample_size": config.sample_size,
        "max_rounds": config.max_rounds,
        "candidate_evaluations": 0,
        "physical_decisions": [],
    }
    logical_original = PlanOptimizer.optimize_logical_plan
    estimate_original = LogicalOptimizer.estimate_plan_cost
    physical_original = PhysicalOptimizer.optimize_exec_model
    marker = ContextVar("native_physical_boundary", default=None)

    class Boundary(logging.Handler):
        def emit(self, record):
            context = marker.get()
            if (
                context is not None
                and "Physical Plan Optimization Time:" in record.getMessage()
            ):
                context["rest_start"] = time.monotonic()

    def logical(self, *args, **kwargs):
        with stage("planning", "native Nirvana logical search and rewrites"):
            selected = logical_original(self, *args, **kwargs)
            from experiments.system_comparison.backends.nirvana.sql_pushdown import (
                push_sql_filters,
            )

            save(folder / "plan-logical-native.json", plan(selected))
            selected, evidence["sql_filter_pushdown"] = push_sql_filters(selected)
        save(folder / "plan-logical.json", plan(selected))
        return selected

    async def estimate(self, *args, **kwargs):
        with stage(
            "sampling",
            "native Nirvana candidate sample execution and quality assessment",
        ):
            result = await estimate_original(self, *args, **kwargs)
        evidence["candidate_evaluations"] += 1
        return result

    async def physical(self, node, input_data):
        context = {"start": time.monotonic(), "rest_start": None}
        token = marker.set(context)
        try:
            return await physical_original(self, node, input_data)
        finally:
            finished = time.monotonic()
            marker.reset(token)
            split = context["rest_start"]
            if split is not None:
                timer.intervals.append(
                    {
                        "phase": "sampling",
                        "label": "native Nirvana model-comparison sample",
                        "start": context["start"] - timer.start,
                        "end": split - timer.start,
                    }
                )
            timer.intervals.append(
                {
                    "phase": "execution",
                    "label": "native Nirvana selected-model execution",
                    "start": (split or context["start"]) - timer.start,
                    "end": finished - timer.start,
                }
            )
            evidence["physical_decisions"].append(
                {
                    "operator": node.op_name,
                    "model": getattr(node.operator, "model", None),
                    "sampled": split is not None,
                }
            )

    logger = logging.getLogger("nirvana.optim.physical")
    handler = Boundary(level=logging.INFO)
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        with (
            QueryTimer(folder) as timer,
            patch.object(PlanOptimizer, "optimize_logical_plan", logical),
            patch.object(LogicalOptimizer, "estimate_plan_cost", estimate),
            patch.object(PhysicalOptimizer, "optimize_exec_model", physical),
        ):
            output, cost, native_runtime = dataframe.optimize_and_execute(
                optim_config=config
            )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
    save(folder / "plan.json", plan(dataframe.leaf_node))
    evidence["completed"] = True
    evidence["native_reported_runtime"] = native_runtime
    save(folder / "nirvana-mechanism.json", evidence)
    return output, cost, timer.result["query_seconds"], timer.result
