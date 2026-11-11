"""One search's rollout, paired with its own pre-execution cache snapshot."""

import copy

from planning.learning.training.data import dataset
from planning.learning.training.trajectory import experience, state_key
from planning.observations.request_timing import record_phase
from planning.observations.statistics import ingest_history
from planning.state import digest
from reference_answers.sql.aggregate import capture_plan

from training.artifacts import save
from training.calibration.global_profile import shared_profile
from training.collection.features import frozen_features
from training.collection.history import save_history_context
from training.collection.policy import (
    PLAN_GENERATION_POLICIES,
    appstore_filter_before_join,
    plan_signature,
)
from training.references.scoring import RESULT_ACCURACY_METRIC


def expected_requests(problem, state):
    return {
        str(a["operator_id"]): {
            "input_rows": state.statistics.get(digest(a), {}).get("input_rows"),
            "seconds_per_request": problem.profile.get("request_profiles", {})
            .get(a["model_id"], {})
            .get(problem.ops[a["operator_id"]]["kind"], {})
            .get("seconds_per_request"),
        }
        for a in problem.semantic_actions(state)
    }


def measured_record(
    evaluator,
    query,
    batch,
    *,
    workload,
    history,
    output,
    policy,
    plan_generation,
    metadata,
    online_timing,
    descriptive_profile,
    initial_calibration,
    progress,
):
    """Execute the selected completion; leave quality pending comparison."""
    frozen, states, selected_spec = batch.frozen, batch.states, batch.selected_spec
    measured_search_policy = PLAN_GENERATION_POLICIES[plan_generation]
    selected_state = next(s for s in states if frozen.spec(s) == selected_spec)
    save(
        output / "selected-plan.json",
        {
            "plan": selected_spec,
            "plan_key": state_key(selected_state),
            "plan_signature": plan_signature(selected_spec),
            "minimum_accuracy": metadata.get("minimum_accuracy"),
            "selection_basis": batch.result["rollout_selection"]["policy"],
            "rollout_selection": batch.result["rollout_selection"],
            "search_policy": measured_search_policy,
            "plan_generation": plan_generation,
            "candidate_count": len(states),
        },
    )
    before = history.snapshot()
    before_ref = save_history_context(output, before)
    current, rebuilt, features, audit = frozen_features(
        query, frozen.profile, selected_spec, before
    )
    context = "stream-" + digest(
        [
            workload["id"],
            before_ref["sha256"],
            state_key(selected_state),
            metadata["rollout_index"],
        ]
    )
    limit = policy.threshold([])
    structural_timeout = workload["dataset"] == "appstore" and appstore_filter_before_join(
        query, selected_spec
    )
    if structural_timeout:
        artifact = output / "sql/timeout"
        artifact.mkdir(parents=True, exist_ok=True)
        actual = {
            "status": "timeout", "time_lower_bound": limit,
            "events": [], "artifact": str(artifact.resolve()),
            "execution_skipped": True,
        }
        save(artifact / "timeout.json", {
            "status": "timeout", "execution_skipped": True,
            "accuracy": 0.0, "time_lower_bound": limit,
        })
    else:
        actual = evaluator.run(
            current.spec(rebuilt),
            source_query=query,
            allow_timeout=True,
            include_rows=False,
            execution_timeout=limit,
            capture_aggregate_input=capture_plan(workload["sql"]) is not None,
            expected_progress=expected_requests(current, rebuilt),
        )
    ingest_history(
        actual["events"], history, adaptive=frozen.profile.get("adaptive_sketch")
    )
    # Sampling already pooled its own physical responses natively. Start from
    # that updated bank, then add only this execution's response observations.
    descriptive_profile["request_profiles"] = copy.deepcopy(
        frozen.profile.get("request_profiles", {})
    )
    online_timing["measured"] = record_phase(
        descriptive_profile,
        actual["events"],
        source="measured",
        scope=str(output.resolve()) + "/measured",
        wall_seconds=actual.get("seconds"),
        wall_basis="SQL execution wall latency; unknown for timed out execution",
        merge_costs=True,
    )
    save(output / "online-timing.json", online_timing)
    # The final query commit includes actual execution responses;
    # the frozen feature/label profile excludes descriptive timing.
    save(
        output / "calibration-update.json",
        {
            "before": initial_calibration,
            "after": shared_profile({**frozen.profile, **descriptive_profile}),
        },
    )
    record = experience(
        current,
        rebuilt,
        source="measured",
        trajectory_features=features,
        times=[actual["seconds"]] if actual["status"] == "completed" else [],
        time_lower_bound=actual.get("time_lower_bound"),
        accuracy=0.0 if actual["status"] == "timeout" else None,
        context=context,
        metadata=dict(
            metadata,
            feature_audit=audit,
            cache_history_before=before_ref,
            single_search_candidate=True,
            accuracy_status="pending_comparison",
            quality={
                "status": "pending_comparison",
                "metric": RESULT_ACCURACY_METRIC,
            },
            timing_artifacts=[actual["artifact"]],
            progress=actual.get("progress"),
            timeout_seconds=limit,
            execution_skipped=actual.get("execution_skipped", False),
        ),
    )
    # Persist executed evidence before callbacks or later orchestration can fail.
    save(output / "experience.json", dataset([record]))
    progress(
        f"{workload['id']}: search {metadata['rollout_index']}/{metadata['searches_per_query']} "
        f"feedback recorded; status={actual['status']}",
        flush=True,
    )
    return record
