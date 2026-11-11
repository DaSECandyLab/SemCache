"""Field projection and survival evidence helpers for independent sampling."""

import math
import re

from ..observations.statistics import serialize_value
from ..state import canonical, field_key
from .prefix_rates import prefix_rate
from .rows import SampleRow
from .units import field_tokens


def projection_columns(op, fields, needed, child_fields):
    """Restrict exported ordinary projection expressions to needed bindings.

    This parser only separates the extension's SELECT list. Quoted strings,
    escaped quotes and nested calls keep their original SQL byte for byte.
    """
    sql = op.get("sql", op.get("sampling_sql", ""))
    if not sql.startswith("SELECT __sample_id,") or not sql.endswith(" FROM $INPUT"):
        return op
    body = sql[len("SELECT __sample_id,") : -len(" FROM $INPUT")]
    expressions, start, nesting, quote, index = [], 0, 0, None, 0
    while index < len(body):
        char = body[index]
        if quote:
            if char == quote:
                if index + 1 < len(body) and body[index + 1] == quote:
                    index += 1
                else:
                    quote = None
        elif char in ("'", '"'):
            quote = char
        elif char in "([":
            nesting += 1
        elif char in ")]":
            nesting -= 1
        elif char == "," and nesting == 0:
            expressions.append(body[start:index])
            start = index + 1
        index += 1
    expressions.append(body[start:])
    if len(expressions) != len(fields):
        raise ValueError("Exported projection expressions do not match output bindings")
    chosen = [
        expression
        for expression, field in zip(expressions, fields)
        if field_tokens([field]) & needed
    ]
    dependencies = set()
    for expression in chosen:
        value = expression.rsplit(" AS ", 1)[0]
        value = re.sub(r"'(?:[^']|'')*'", "", value)
        dependencies.update(
            name.replace('""', '"')
            for name in re.findall(r'"((?:[^"]|"")*)"', value)
            if name.replace('""', '"') in child_fields
        )
    result = dict(op)
    result["required_columns"] = sorted(dependencies)
    result["sql"] = "SELECT __sample_id," + ",".join(chosen) + " FROM $INPUT"
    result["sampling_sql"] = result["sql"]
    return result


def weighted_depths(coordinator, records, depths, complete):
    table = {}
    for depth in depths:
        selected = [
            (weight, passed) for length, weight, passed in records if length >= depth
        ]
        mass = sum(weight for weight, _ in selected)
        squares = sum(weight * weight for weight, _ in selected)
        effective = mass * mass / squares if squares else 0.0
        rate = (
            sum(weight for weight, passed in selected if passed) / mass
            if mass and complete
            else None
        )
        interval = (
            coordinator._interval(rate * effective, effective, math.inf)
            if rate is not None
            else (0.0, 1.0)
        )
        table[str(depth)] = {
            "rate": rate,
            "observed": len(selected),
            "passed": sum(passed for _, passed in selected),
            "effective_samples": effective,
            "population": mass,
            "interval": list(interval),
            "reliable": rate is not None
            and effective >= coordinator.min_depth_samples
            and (interval[1] - interval[0]) / 2 <= coordinator.depth_error,
            "strata": [],
            "interval_basis": "independent weighted Wilson diagnostic",
        }
    return table


def depth_rates(coordinator, node, candidates, successes, complete):
    fields = {}
    include_depths = coordinator.sample_depth_selectivity or getattr(
        coordinator, "ensure_prefix_depth_rates", True
    )
    for (relation, name), summary in coordinator.population_summaries.items():
        if relation not in node.relations:
            continue
        depths = [0, *summary.sketches] if include_depths else [0]
        members = {}
        for index, unit in enumerate(candidates):
            row = next(
                (
                    row
                    for side in unit
                    for row in side
                    if any(rel == relation for rel, _ in row.lineage)
                ),
                None,
            )
            if row is None or name not in row.values:
                continue
            origins = tuple(origin for origin in row.lineage if origin[0] == relation)
            source = SampleRow(row.identity, row.values, origins)
            passed = (successes[index] if complete else False) or members.get(
                origins, (0, 0, False)
            )[2]
            members[origins] = (
                len(serialize_value(row.values[name])),
                coordinator._weight(source),
                passed,
            )
        records = list(members.values())
        if records:
            fields[f"{relation}:{name}"] = weighted_depths(
                coordinator, records, depths, complete
            )
    base_names = {name for (_, name) in coordinator.population_summaries}
    generated = {
        name
        for unit in candidates
        for side in unit
        for row in side
        for name in row.values
        if name not in base_names
    }
    for name in sorted(generated):
        members = {}
        for index, unit in enumerate(candidates):
            row = next(
                (row for side in unit for row in side if name in row.values), None
            )
            if row is None:
                continue
            value = serialize_value(row.values[name])
            origin = tuple(sorted(row.lineage, key=repr))
            passed = (successes[index] if complete else False) or members.get(
                origin, ("", 0, False)
            )[2]
            members[origin] = (value, coordinator._weight(row), passed)
        values = [value for value, _, _ in members.values()]
        records = [
            (len(value), weight, passed) for value, weight, passed in members.values()
        ]
        summary = coordinator._field_summaries({name: values})["summaries"][name]
        depths = [0, *map(int, summary["sketches"])] if include_depths else [0]
        fields["generated:" + name] = weighted_depths(
            coordinator, records, depths, complete
        )
    return fields


def generated_field_identities(op):
    return {
        "generated:" + descriptor.get("value_key", descriptor["name"]): canonical(
            [
                descriptor.get(
                    "sample_expression",
                    descriptor.get(
                        "binding", descriptor.get("value_key", descriptor["name"])
                    ),
                ),
                descriptor.get("type"),
            ]
        )
        for descriptor in op.get("input_field_schema", [])
    }


def generated_survivals(state, observations, node, op, local):
    """Start generated populations at their producer, then multiply ancestors.

    A preceding predicate need not read this generated value. Its independent
    scalar selectivity is the fallback when it has no coordinate for the field.
    """
    result = {}
    dependencies = field_tokens(op.get("required_columns", []))
    identities = generated_field_identities(op)
    for field, table in local.items():
        if not field.startswith("generated:"):
            continue
        name = field[len("generated:") :]

        def at(nid, depth):
            current = state._nodes_by_id[nid]
            if current.operator_id is None:
                return 1.0
            child_fields = field_tokens(
                [
                    column
                    for child in current.children
                    for column in state._nodes_by_id[child].fields
                ]
            )
            produced = field_tokens(current.fields) - child_fields
            if (
                name in produced
                or name.startswith("__semantic_input:")
                and dependencies & produced
            ):
                return 1.0
            children = [
                child
                for child in current.children
                if name in field_tokens(state._nodes_by_id[child].fields)
                or dependencies & field_tokens(state._nodes_by_id[child].fields)
            ]
            if not children:
                children = list(current.children)
            values = [at(child, depth) for child in children]
            info = observations[nid].statistics
            previous_field = field
            if field in identities:
                previous_field = next(
                    (
                        name
                        for name, identity in info.get(
                            "generated_field_identities", {}
                        ).items()
                        if identity == identities[field]
                    ),
                    field,
                )
            factor = prefix_rate(info, previous_field, depth)
            return (
                0.0
                if factor == 0 or 0 in values
                else math.prod(values) * factor
                if factor is not None and all(value is not None for value in values)
                else None
            )

        children = [
            child
            for child in node.children
            if name in field_tokens(state._nodes_by_id[child].fields)
            or dependencies & field_tokens(state._nodes_by_id[child].fields)
        ]
        if not children:
            children = list(node.children)
        result[field] = {}
        for depth in table:
            values = [at(child, depth) for child in children]
            result[field][depth] = (
                0.0
                if 0 in values
                else math.prod(values)
                if all(value is not None for value in values)
                else None
            )
    return result


def value_producer_inputs(problem, state, node, outputs, observations):
    """Read only completed aggregate/window values already produced by children."""
    result = []
    for child in node.children:
        producer = state._nodes_by_id[child]
        if producer.operator_id is None:
            continue
        op = problem.all_ops[producer.operator_id]
        kind = op.get("kind", op.get("type", "")).upper()
        if not ("AGGREGATE" in kind or "WINDOW" in kind):
            continue
        if observations[child].statistics.get("sampling_complete", False):
            result.append(outputs[child])
    return tuple(result)


def input_evidence(coordinator, op, inputs, *, fallback_inputs=()):
    """Freeze field populations from the independent input view, without rows."""
    from ..observations.evidence import freeze_evidence

    result = {}
    descriptors = op.get("input_field_schema", []) or op.get("required_columns", [])
    for descriptor in descriptors:
        if isinstance(descriptor, str):
            key = descriptor
        elif isinstance(descriptor, dict):
            key = descriptor.get("value_key")
            if key is None:
                key = (
                    field_key(descriptor)
                    if "binding" in descriptor or "identity" in descriptor
                    else descriptor.get("name", field_key(descriptor))
                )
        else:
            key = field_key(descriptor)
        if key in result:
            continue
        source = next((rows for rows in inputs if rows and key in rows[0].values), ())
        if not source:
            source = next(
                (rows for rows in fallback_inputs if rows and key in rows[0].values),
                (),
            )
        origins = {
            relation
            for row in source
            for relation, _ in row.lineage
            if (relation, key) in coordinator.population_summaries
        }
        if not origins:
            origins = {
                relation
                for relation, name in coordinator.population_summaries
                if name == key
            }
        if len(origins) == 1:
            relation = next(iter(origins))
            summary = coordinator.population_summaries[(relation, key)].json()
            evidence = {
                "summary": summary,
                "survival_field": f"{relation}:{key}",
                "origin": [str(relation), key],
            }
        elif source:
            values = [serialize_value(row.values[key]) for row in source]
            summary = coordinator._field_summaries({key: values})["summaries"][key]
            evidence = {"summary": summary, "survival_field": "generated:" + key}
        elif getattr(coordinator, "ensure_prefix_depth_rates", True):
            if not inputs or not any(not rows for rows in inputs):
                raise ValueError(
                    f"Cannot estimate prefix rates for required field {key}: "
                    "missing input field summary or materialized values"
                )
            summary = coordinator._field_summaries({key: []})["summaries"][key]
            evidence = {"summary": summary, "survival_field": "generated:" + key}
        else:
            continue
        result[key] = freeze_evidence(evidence)
    return result
