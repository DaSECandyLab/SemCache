"""Query-local implementation sharing for repeated semantic tasks."""

import re
from collections import defaultdict

from .identity import canonical, field_key

_BOUND = re.compile(r'"(\[\d+,\d+\])"')
_ALIAS = re.compile(r'"(\[\d+,\d+\])"\s+AS\s+"(\[\d+,\d+\])"', re.I)


def duplicate_groups(problem):
    """Match tasks by ordered source expressions, ignoring table-instance IDs."""
    origins = {}
    for relation in problem.query.get("relations", []):
        for column in relation.get("columns", []):
            origins[field_key(column)] = canonical([
                "column", relation.get("catalog"), relation.get("schema"),
                relation.get("table"), column.get("name"), column.get("type"),
            ])
    # Resolve direct bound aliases without equating unrelated computed columns.
    aliases = [pair for node in problem.logical.values()
               for pair in _ALIAS.findall(node.get("sampling_sql", ""))]
    for _ in range(len(aliases) + 1):
        changed = False
        for source, target in aliases:
            if source in origins and target not in origins:
                origins[target] = origins[source]
                changed = True
        if not changed:
            break

    def expression(field):
        if not isinstance(field, dict):
            return origins.get(field_key(field), field_key(field))
        if "binding" in field:
            return origins.get(field_key(field), field_key(field))
        text = field.get("sample_expression")
        if text:
            return _BOUND.sub(lambda m: origins.get(m[1], m[0]), text)
        return field_key(field)

    groups = defaultdict(list)
    for oid, op in problem.ops.items():
        fields = op.get("input_field_schema") or op.get("required_columns", [])
        key = canonical([
            op["kind"], op["task"], op.get("operator_specification"),
            op.get("input_fields"), op.get("left_input_fields"),
            op.get("right_input_fields"),
            [(expression(f), f.get("type") if isinstance(f, dict) else None)
             for f in fields],
            [{k: v for k, v in field.items()
              if k not in ("binding", "value_key")}
             if isinstance(field, dict) else field
             for field in op.get("output_schema", [])],
        ])
        groups[key].append(oid)
    return {oid: tuple(sorted(ids)) for ids in groups.values() if len(ids) > 1
            for oid in ids}


def implementation(action):
    return canonical({k: action[k] for k in ("model_id", "layout")})


def matching_implementation(problem, state, action):
    group = problem.duplicate_groups.get(action["operator_id"], ())
    if not group:
        return True
    for previous in state.action_views:
        if previous["operator_id"] in group:
            return implementation(previous) == implementation(action)
    return True
