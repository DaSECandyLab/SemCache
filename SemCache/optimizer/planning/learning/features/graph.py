"""Build one plan's GAT features from logical structure and frozen observations."""

from planning.observations.accuracy import evidence as accuracy_evidence

from .cache import (
    bind_prompt_features,
    cache_relations,
    history_features,
    static_prompt_features,
)
from .remaining import remaining_numeric
from .schema import (
    KINDS,
    description,
    field_identity,
    fields,
    first_fields,
    implementation,
    kind,
    known,
)
from .statistics import add_input_time, estimated_feature, statistics


def encode(problem, state):
    from planning.observations.statistics import prefix_length

    logical = problem.logical
    semantic = {op["node_id"]: op for op in problem.ops.values()}
    chosen = state._action_by_operator
    keys = state._action_keys_by_operator
    work = {}
    for nid in state.partial_plans.remaining:
        node, op = logical[nid], semantic.get(nid)
        work[nid] = {
            "fields": fields(node, op),
            "description": description(node, op),
            "numeric": remaining_numeric(op, node, problem.profile),
        }
    rows, by_operator, estimates = {}, {}, {}
    for plan_node in state.partial_plans.nodes:
        nid = plan_node.node_id
        node, op = logical[nid], semantic.get(nid)
        action = chosen.get(plan_node.operator_id)
        stat = (
            dict(state.statistics.get(keys[plan_node.operator_id], {}))
            if action
            else {
                "input_rows": node.get("estimated_rows"),
                "output_rows": node.get("estimated_rows"),
                **problem.profile.get("node_statistics", {}).get(str(nid), {}),
                **state.statistics.get(f"scan:{nid}", {}),
            }
        )
        if not action and stat.get("input_width") is None:
            stat["input_width"] = stat.get("output_width")
        if op and action:
            static_pre, static_length = static_prompt_features(
                op, action, problem.profile
            )
            bind_prompt_features(stat, static_pre, static_length)
            quality_evidence = accuracy_evidence(
                problem.profile, action["model_id"], op["kind"], action["layout"]
            )
            quality = (
                quality_evidence.get("value")
                if isinstance(quality_evidence, dict)
                else quality_evidence
            )
            if known(quality):
                stat["accuracy"] = quality
            else:
                stat.pop("accuracy", None)
            stat["reliability"] = dict(stat.get("reliability", {}))
            count = (
                quality_evidence.get("effective_samples")
                if isinstance(quality_evidence, dict)
                else None
            )
            if known(count) and count > 0:
                stat["reliability"]["accuracy"] = count / (count + 1)
            by_operator[plan_node.operator_id] = nid
            estimates[plan_node.operator_id] = stat
        add_input_time(
            stat,
            op or node,
            problem.profile,
            action.get("model_id") if action else None,
        )
        label = kind(op or node)
        rows[nid] = {
            "fields": (
                fields(node, op)[0],
                sorted({field_identity(f) for f in plan_node.fields}),
            ),
            "description": description(node, op),
            "first_fields": first_fields(op, action) if op else [[], []],
            "numeric": [float(label == name) for name in KINDS]
            + implementation(
                action,
                stat.get("pre_length")
                if stat.get("pre_length") is not None
                else prefix_length(stat.get("pre")),
            )
            + statistics(stat, bool(op))
            + estimated_feature(stat, "l_self", applicable=bool(op)),
            "l_existing": history_features(problem, action, stat)
            if op and action
            else None,
            "children": list(plan_node.children),
            "l_previous": {},
        }
    cache_relations(rows, by_operator, chosen, estimates, state._descendants)
    return {
        "nodes": rows,
        "work": work,
    }
