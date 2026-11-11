"""Keep runtime priors small; fresh execution sketches replace sampling banks."""

from planning.execution.cache_prediction import reuse_features
from planning.learning.features.graph import encode


def initial_references(problem, state):
    graph = encode(problem, state)
    result = {}
    for oid, op in problem.ops.items():
        action = state._action_by_operator[oid]
        stat = state.statistics[state._action_keys_by_operator[oid]]
        result[oid] = {
            "estimated_input_rows": stat["input_rows"],
            "selectivity": stat.get("estimated_selectivity", stat.get("selectivity")),
            "prefix_features": reuse_features(
                graph["nodes"][op["node_id"]],
                op,
                action,
                problem.profile,
                include_tokens=True,
            ),
        }
    return result


def compact_profile(profile):
    keys = (
        "models",
        "constraints",
        "system_prefix",
        "operator_specifications",
        "accuracy",
        "task_accuracy",
        "request_profiles",
        "task_request_profiles",
        "relational_profiles",
        "operator_time_calibration",
        "sampling_strategy",
        "history",
        "prefix_cooccurrences",
    )
    result = {k: profile[k] for k in keys if k in profile}
    discarded = {
        "summary",
        "right_summary",
        "input_field_summaries",
        "output_field_summaries",
        "depth_survival_statistics",
        "depth_selectivity_statistics",
        "prefix_selectivity_statistics",
        "conditional_sketch_plan",
        "adaptive",
        "estimated_output_survival",
        "implementations",
    }
    for bank in (
        "node_statistics",
        "source_statistics",
        "operator_statistics",
        "statistics",
    ):
        result[bank] = {
            key: {k: v for k, v in row.items() if k not in discarded}
            for key, row in profile.get(bank, {}).items()
            if bank != "statistics" or key.startswith("scan:")
        }
    return result
