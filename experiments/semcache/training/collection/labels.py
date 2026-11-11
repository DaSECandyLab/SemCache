"""Compatible collection metadata and estimated labels in one frozen context."""

from planning.cost.objectives import (
    MissingEstimate,
    estimated_objectives,
    plan_accuracy,
)
from planning.learning.training.trajectory import experience
from planning.state import SearchProblem, digest

from training.collection.features import replay_features
from training.collection.history import save_history_context
from training.collection.policy import PLAN_GENERATION_POLICIES

# Measured episodes require the feedback protocol marker; estimated episodes use the bootstrap feature contract.
COLLECTION_PROTOCOL = 28


def record_metadata(
    workload, *, phase, plan_generation, uses_value_model, cache_segment=None
):
    """Shared metadata; per-record feature and cache audits are added later."""
    measured_search_policy = PLAN_GENERATION_POLICIES[plan_generation]
    return {
        "collection_protocol": COLLECTION_PROTOCOL,
        "execution_mode": "data_chunk_pipeline",
        "semantic_join_mode": "stream_first_left_then_replay",
        "workload": workload["id"],
        "dataset": workload["dataset"],
        "workload_sql_digest": digest(workload["sql"]),
        "base_rows": workload["rows"],
        "quality_basis": "frozen configured teacher answer agreement; not ground truth",
        **({"cache_segment": cache_segment} if cache_segment else {}),
        **(
            {
                "feedback_search_policy": measured_search_policy,
                "plan_generation": plan_generation,
            }
            if phase == "measured"
            else {}
        ),
        "search_mode": plan_generation,
    }


def estimated_records(query, frozen, states, *, history, output, timings, metadata):
    """Label all Beam completions without running any full candidate query."""
    records = []
    before = history.snapshot()
    before_ref = save_history_context(output, before)
    context = "pre-" + before_ref["sha256"]
    with timings.measure("context_setup"):
        current = SearchProblem(query, {**frozen.profile, "history": before})
    for state in states:
        with timings.measure("features_and_audit"):
            current, rebuilt, features, audit = replay_features(
                current, frozen.spec(state)
            )
        missing = None
        cost_audit = {}
        with timings.measure("objective_estimation"):
            try:
                seconds, accuracy = estimated_objectives(
                    current, rebuilt, audit=cost_audit
                )
            except MissingEstimate as error:
                seconds, accuracy = None, plan_accuracy(current, rebuilt)
                missing = str(error)
        record = experience(
            current,
            rebuilt,
            source="estimated",
            trajectory_features=features,
            times=[seconds] if seconds is not None else [],
            accuracy=accuracy,
            context=context,
            metadata=dict(
                metadata,
                feature_audit=audit,
                cache_history_before=before_ref,
                estimated_label_audit={
                    "time": "unknown" if seconds is None else "estimated",
                    "accuracy": "unknown" if accuracy is None else "calibrated product",
                    "reason": missing,
                    "cost_prior": cost_audit,
                },
            ),
        )
        records.append(record)
    return records
