"""Bind a single semantic sampling stage to its explicit action."""

from typing import Any

from planning.search.protocol import validate_spec


class StageActionPolicy:
    """Apply the candidate's model/layout to one bound semantic stage."""

    def __init__(self):
        self.action: dict[str, Any] | None = None
        self.pairwise = False
        self.candidate_pairs = False

    def optimize(self, query):
        if self.action is None or len(query["semantic_operators"]) != 1:
            raise ValueError(
                "Execution requires one explicitly configured semantic action per stage"
            )
        operator = query["semantic_operators"][0]
        if (operator["kind"] == "SEMANTIC_JOIN") != self.pairwise:
            raise ValueError("Semantic join requires the candidate-pair path")
        action = {**self.action, "operator_id": operator["operator_id"]}
        logical = next(n for n in query["operators"] if n["node_id"] == operator["node_id"])
        if self.candidate_pairs and len(logical["children"]) == 2:
            # Lower an already selected flat pair stage. The identity equality
            # must precede inference, or a two-table stage would call the LLM N² times.
            from ...state import SearchProblem

            problem = SearchProblem(
                query, {"layouts": {str(operator["operator_id"]): [action["layout"]]}}
            )
            state = problem.initial
            while not problem.complete(state):
                choices = problem.actions(state)
                if not choices:
                    raise ValueError("Cannot lower the selected candidate-pair stage")
                choice = min(
                    choices,
                    key=lambda a: ("pair_for" not in a, a["operator_id"], a["inputs"]),
                )
                if choice["operator_id"] in problem.ops:
                    choice = dict(
                        choice, model_id=action["model_id"], layout=action["layout"]
                    )
                state = problem.transition(state, choice, observe=False)
            spec = problem.spec(state)
            if not any(
                "pair_for" in a
                for r in spec.get("tree_choices", [])
                for a in r["actions"]
            ):
                raise ValueError(
                    "Candidate identity filter was not applied before inference"
                )
            validate_spec(query, spec)
            return spec
        spec = {
            "version": 1,
            "kind": "PlanSpec",
            "query_id": query["query_id"],
            "actions": [action],
        }
        validate_spec(query, spec)
        return spec
