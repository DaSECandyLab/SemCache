"""One result contract for native and explicit reference Beam controllers."""

import copy

from planning.state import digest

from .heuristics import PRIORITY_POLICY


def finish_search(
    search,
    problem,
    completed,
    *,
    scores,
    times,
    accuracies,
    time_reasons,
    decision,
    exhausted,
    action_budget_exhausted,
    expansions,
    original_error,
    heuristic_guard,
):
    if any(
        len(values) != len(completed)
        for values in (scores, times, accuracies, time_reasons, decision["statuses"])
    ):
        raise ValueError("Prediction count differs from completed states")
    basis = (
        "learned completion value"
        if search.uses_value_model
        else "calibrated input rows × seconds per input"
    )
    assessments = []
    for state, seconds, quality, reason, status in zip(
        completed, times, accuracies, time_reasons, decision["statuses"], strict=True
    ):
        row = {
            "plan_key": digest(state.history),
            "accuracy": quality,
            "status": status,
            "basis": basis,
            "time_prediction": seconds,
        }
        if search.config.request_guided:
            from planning.cost.objectives import MissingEstimate
            from planning.cost.requests import estimated_request_count

            try:
                row["estimated_requests"] = estimated_request_count(problem, state)
            except MissingEstimate:
                row["estimated_requests"] = None
            if seconds is None:
                row.update(time_status="unknown", time_basis=reason)
                if reason and "ranking by estimated semantic request count" in reason:
                    row["basis"] = "estimated semantic request count; full time unknown"
        elif seconds is None:
            row.update(time_status="unknown", time_basis=reason)
        assessments.append(row)
    selected = [completed[i] for i in decision["selected_indices"]]
    accuracy_fallback = decision["constraint_satisfied"] is False and bool(selected)
    search._resident_scores = {
        state.history: tuple(score)
        for state, score in zip(completed, scores, strict=True)
    }
    search._resident_choices = {
        "fastest": selected[0].history if selected else None,
        "accurate": completed[decision["accurate_index"]].history
        if decision["accurate_index"] is not None
        else None,
    }
    from planning.state.trees import cached_validation

    with cached_validation():
        complete_specs = [problem.spec(state) for state in completed]
    frontier_specs = [
        copy.deepcopy(complete_specs[i]) for i in decision["selected_indices"]
    ]
    guard = dict(heuristic_guard)
    if "complete_index" in guard:
        index = guard.pop("complete_index")
        guard["plan"] = complete_specs[index] if index is not None else None
    reason = None
    budget_reason = (
        (
            "max_actions"
            if action_budget_exhausted
            else "max_expansions"
            if expansions >= search.config.max_expansions
            else "max_seconds"
        )
        if exhausted
        else None
    )
    if decision["constraint_satisfied"] is False and (
        completed or any(row.get("quality_pruned", 0) for row in search.trace)
    ):
        reason = (
            "No candidate meets A_min; selected the highest predicted accuracy completion"
            if accuracy_fallback
            else "No candidate satisfies A_min with the available completion quality estimates"
        )
    elif not completed:
        reason = (
            "ParetoBeamSearch budget exhausted before completing a plan: "
            f"{budget_reason}; expansions={expansions}/{search.config.max_expansions}; "
            f"layers={len(search.trace)}; beam_width={search.config.beam_width}"
            if exhausted
            else "No legal complete plan"
        )
    return {
        "objective_units": "seconds"
        if search.config.request_guided
        else "seconds_and_accuracy",
        "plan_generation": search.config.plan_generation,
        "duplicate_implementation_policy": "same_task_source_model_layout_v1",
        "duplicate_operator_groups": [list(group) for group in
                                      sorted(set(problem.duplicate_groups.values()))],
        "operator_priority_policy": PRIORITY_POLICY
        if search.config.priority_mode != "value"
        else "none",
        "preference": search.config.preference,
        "layer_ranking": search.config.layer_ranking(problem.constraints.get("a_min")),
        "time_calibration": copy.deepcopy(
            problem.profile.get("operator_time_calibration", {})
        ),
        "frontier": frontier_specs,
        "complete": complete_specs,
        "states": selected,
        "assessments": assessments,
        "trace": search.trace,
        "constraint_satisfied": decision["constraint_satisfied"],
        "accuracy_fallback": accuracy_fallback,
        "selection_basis": "highest_predicted_accuracy_fallback"
        if accuracy_fallback
        else "requested_objective",
        "reason": reason,
        "budget_exhausted": exhausted,
        "budget_reason": budget_reason,
        "action_budget_exhausted": action_budget_exhausted,
        "expansions": expansions,
        "value_available": search.uses_value_model,
        "selected": copy.deepcopy(
            complete_specs[decision["accurate_index"]]
            if not search.config.training
            and search.config.preference == "accurate"
            and problem.constraints.get("a_min") is None
            and decision["accurate_index"] is not None
            else frontier_specs[0]
        )
        if frontier_specs
        else None,
        "original_candidate_error": original_error,
        "heuristic_guard": guard,
    }
