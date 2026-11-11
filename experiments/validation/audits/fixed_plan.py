"""Fixed legal SemCache plans for model-free native EXPLAIN checks."""

import copy
import sys

from experiments.core.paths import ROOT

sys.path.insert(0, str(ROOT / "SemCache/optimizer"))
from optimizer_server import validate_spec


class FixedPlan:
    def optimize(self, query):
        operators = {op["operator_id"]: op for op in query["semantic_operators"]}
        spec = {
            "version": 1,
            "kind": "PlanSpec",
            "query_id": query["query_id"],
            "actions": [
                {
                    "operator_id": key,
                    "model_id": "validation",
                    "layout": copy.deepcopy(operators[key]["default_layout"]),
                }
                for group in query["order_groups"]
                for key in group
            ],
        }
        validate_spec(query, spec)
        return spec
