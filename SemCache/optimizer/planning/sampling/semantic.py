"""Distinct actual semantic inputs, with exact population restoration."""

import math
from dataclasses import dataclass, field

from planning.observations.evidence import freeze_evidence

from ..state import digest, task_key
from .rows import CartesianRows, SampleRow


@dataclass
class InputGroup:
    key: str
    unit: tuple
    members: list = field(default_factory=list)
    weight: float = 0.0


def input_keys(op):
    descriptors = {f["name"]: f for f in op.get("input_field_schema", [])}
    names = op.get("input_fields") or op.get("left_input_fields", []) + op.get(
        "right_input_fields", []
    )
    return [descriptors.get(name, {}).get("value_key", name) for name in names]


def groups(
    coordinator,
    op,
    inputs,
    model,
    counts=None,
    *,
    max_missing=None,
    selected_units=None,
):
    product = (
        inputs[0] if len(inputs) == 1 and isinstance(inputs[0], CartesianRows) else None
    )
    if any(not rows for rows in inputs):
        return [], product, None
    leaf = product.leaf_for(input_keys(op)) if product is not None else None
    units = (
        selected_units
        if selected_units is not None
        else (
            (([row],) for row in (leaf if leaf is not None else inputs[0]))
            if len(inputs) == 1
            else (([left], [right]) for left in inputs[0] for right in inputs[1])
        )
    )
    result = {}
    missing_count = 0
    multiplier = 1.0
    if product is not None and leaf is not None:
        multiplier = product.mass(lambda row: coordinator._weight(row, counts)) / sum(
            coordinator._weight(row, counts) for row in leaf
        )
    for unit in units:
        actual = (
            ([product.representative(leaf, unit[0][0])],)
            if product is not None and leaf is not None
            else unit
        )
        key = coordinator.backend.sample_request_key(
            op, actual, model, op["default_layout"]
        )
        if key not in result:
            if key not in coordinator.request_results:
                missing_count += 1
                if max_missing is not None and missing_count > max_missing:
                    return None, product, leaf
            result[key] = InputGroup(key, actual)
        group = result[key]
        group.members.append(unit)
        group.weight += (
            math.prod(coordinator._weight(row, counts) for side in unit for row in side)
            * multiplier
        )
    return list(result.values()), product, leaf


def payload(coordinator, op, unit, output):
    if not output:
        return False, None
    column = coordinator.backend.sample_output_column(op)
    return True, output[0].values[column] if column else None


def restore(coordinator, op, unit, result):
    success, value = result
    if not success:
        return None
    row = (
        unit[0][0] if len(unit) == 1 else CartesianRows.combine(unit[0][0], unit[1][0])
    )
    values = {
        key: val
        for key, val in row.values.items()
        if not key.startswith("__semantic_input:")
    }
    column = coordinator.backend.sample_output_column(op)
    if column:
        values[column] = value
    return SampleRow(row.identity, freeze_evidence(values), row.lineage)


def select(coordinator, problem, op, inputs):
    model = coordinator._sampling_model(problem, op)
    observed, product, leaf = groups(
        coordinator,
        op,
        inputs,
        model,
        max_missing=coordinator._selection_remaining(),
    )
    if observed is None:
        return None
    missing = [
        group for group in observed if group.key not in coordinator.request_results
    ]
    if len(missing) > coordinator._selection_remaining():
        return None
    if missing:
        if len(inputs) == 1:
            representatives = tuple(group.unit[0][0] for group in missing)
            output = coordinator._execute(op, (representatives,), model)
        else:
            pairs = [(group.unit[0][0], group.unit[1][0]) for group in missing]
            output = coordinator._execute(op, inputs, model, pairs=pairs)
        by_id = {row.identity: row for row in output}
        for group in missing:
            identity = (
                group.unit[0][0].identity
                if len(inputs) == 1
                else digest([group.unit[0][0].identity, group.unit[1][0].identity])
            )
            row = by_id.get(identity)
            coordinator.request_results[group.key] = payload(
                coordinator, op, group.unit, (row,) if row else ()
            )
    prior = coordinator.task_selectivity.setdefault(task_key(op), {})
    for group in observed:
        prior.setdefault(group.key, coordinator.request_results[group.key][0])
    if product is not None and leaf is not None:
        by_id = {
            unit[0][0].identity: coordinator.request_results[group.key]
            for group in observed
            for unit in group.members
        }
        replacement = tuple(
            row
            for source in leaf
            if (row := restore(coordinator, op, ([source],), by_id[source.identity]))
            is not None
        )
        return product.replace_leaf(leaf, replacement)
    output = []
    # Restore input order and every duplicate, not just representative context.
    units = (
        (([row],) for row in inputs[0])
        if len(inputs) == 1
        else (([left], [right]) for left in inputs[0] for right in inputs[1])
    )
    for unit in units:
        key = coordinator.backend.sample_request_key(
            op, unit, model, op["default_layout"]
        )
        row = restore(coordinator, op, unit, coordinator.request_results[key])
        if row is not None:
            output.append(row)
    return tuple(output)


def evaluate(coordinator, op, unit, model, phase, layout=None):
    layout = op["default_layout"] if layout is None else layout
    key = coordinator.backend.sample_request_key(op, unit, model, layout)
    if key not in coordinator.request_results:
        output = coordinator._call_backend(op, unit, model, layout, phase)
        coordinator.request_results[key] = payload(coordinator, op, unit, output)
    return coordinator.request_results[key]


def quality(coordinator, problem, state, counts):
    from planning.observations.accuracy import sampling_layouts

    from .accuracy_budget import select_accuracy_groups
    from .quality import record_quality

    for action in problem.semantic_actions(state):
        oid = action["operator_id"]
        if (
            coordinator.accuracy_batches_used[oid]
            >= coordinator.max_accuracy_batches_per_operator
        ):
            continue
        op = problem.execution_operator(state, oid)
        inputs, selected_units = coordinator.accuracy_inputs(problem, state, oid)
        if not inputs or any(not rows for rows in inputs):
            continue
        models = coordinator._models(problem, op)
        teacher = problem.profile.get("judge_model", models[-1])
        from planning.settings.teacher import teacher_supports

        if not teacher_supports(problem.profile, op, teacher):
            raise ValueError("Judge model must support the semantic task")
        # One canonical model/layout fixes parameter identities across all
        # candidate actions; every model receives this round's same groups.
        observed, _, _ = groups(
            coordinator, op, inputs, models[0], counts, selected_units=selected_units
        )
        if observed is None:
            raise RuntimeError("Accuracy grouping requires complete population weights")
        for group in select_accuracy_groups(coordinator, op, observed):
            teacher_layout = op["default_layout"]
            reference_key = coordinator.backend.sample_request_key(
                op, group.unit, teacher, teacher_layout
            )
            reference = evaluate(
                coordinator, op, group.unit, teacher, "accuracy_judge", teacher_layout
            )
            for layout in sampling_layouts(op):
                for model in models:
                    actual_key = coordinator.backend.sample_request_key(
                        op, group.unit, model, layout
                    )
                    key = (actual_key, teacher)
                    if key in coordinator.quality_cache:
                        continue
                    actual = evaluate(
                        coordinator, op, group.unit, model, "accuracy_model", layout
                    )
                    record_quality(
                        problem.profile,
                        op,
                        model,
                        layout,
                        actual,
                        reference,
                        group.weight,
                        teacher,
                        actual_key,
                        reference_key,
                    )
                    coordinator.quality_cache.add(key)
