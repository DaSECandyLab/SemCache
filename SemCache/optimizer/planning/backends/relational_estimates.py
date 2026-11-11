"""Compile read-only population SQL for DuckDB relational cardinality priors.

These statements are only explained, never executed. Semantic predicates are
transparent boundaries: their measured survival is applied separately by the
planner. Real source columns retain their DuckDB statistics across the bound
relational stages, instead of borrowing the NDV of a small sample table.
"""

import json
import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

_TRANSPORT_ID = re.compile(
    r"row_number\(\)\s+OVER\s*\(\s*\)::VARCHAR\s+__sample_id", re.IGNORECASE
)
_SEMANTIC_CALL = re.compile(r"\bsemantic_[a-z0-9_]+\s*\(", re.IGNORECASE)


def _name(value):
    return '"' + value.replace('"', '""') + '"'


def _field(field):
    if "binding" in field:
        return json.dumps(field["binding"], separators=(",", ":"))
    return field["name"]


def _references(expression):
    # A quoted binding can also be the root of a struct-field access, which
    # sqlglot represents as the table part of a Column rather than its name.
    return {
        part.name
        for column in expression.find_all(exp.Column)
        for part in column.parts
    }


def _select(template):
    sql = template.replace("$INPUT", "__input").replace("$LEFT", "__left")
    sql = sql.replace("$RIGHT", "__right")
    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except SqlglotError as error:
        raise ValueError("population SQL dependency parsing failed") from error
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise ValueError("population SQL requires one parseable SELECT statement")
    return statements[0]


def _boolean_constant(expression):
    while isinstance(expression, exp.Paren):
        expression = expression.this
    if isinstance(expression, exp.Cast):
        if expression.args["to"].this != exp.DataType.Type.BOOLEAN:
            return None
        expression = expression.this
    while isinstance(expression, exp.Paren):
        expression = expression.this
    if isinstance(expression, exp.Boolean):
        return expression.this
    if isinstance(expression, exp.Literal) and expression.is_string:
        value = expression.this.lower()
        if value in ("t", "true", "f", "false"):
            return value in ("t", "true")
    return None


def _estimation_filter_sql(sql):
    """Expose Boolean CASE predicates to EXPLAIN only in filter contexts.

    CASE WHEN p THEN TRUE ELSE FALSE END passes exactly the same rows as p.
    Its scalar NULL behavior differs, so only positive WHERE/ON conditions
    and their AND/OR children may be simplified; projections and NOT stay put.
    """
    try:
        tree = sqlglot.parse_one(sql, read="duckdb")
    except SqlglotError:
        return sql
    changed = False

    def condition(expression):
        nonlocal changed
        if isinstance(expression, exp.Case) and expression.this is None:
            branches = expression.args.get("ifs", [])
            if (
                len(branches) == 1
                and _boolean_constant(branches[0].args.get("true")) is True
                and _boolean_constant(expression.args.get("default")) is False
            ):
                changed = True
                return condition(branches[0].this.copy())
        if isinstance(expression, (exp.Paren, exp.And, exp.Or)):
            expression.set("this", condition(expression.this))
            if isinstance(expression, (exp.And, exp.Or)):
                expression.set("expression", condition(expression.expression))
        return expression

    for clause in tree.find_all(exp.Where):
        clause.set("this", condition(clause.this))
    for join in tree.find_all(exp.Join):
        if join.args.get("on") is not None:
            join.set("on", condition(join.args["on"]))
    return tree.sql(dialect="duckdb") if changed else sql


def _projection_unknowns(select, inherited, fields):
    generated = set()
    for expression in select.expressions:
        star = expression if isinstance(expression, exp.Star) else expression.this
        if isinstance(star, exp.Star):
            carried = inherited - {
                column.name for column in star.args.get("except_") or []
            }
            for replacement in star.args.get("replace") or []:
                carried.discard(replacement.alias_or_name)
                if _references(replacement) & inherited:
                    carried.add(replacement.alias_or_name)
            for rename in star.args.get("rename") or []:
                if rename.this.name in carried:
                    carried.remove(rename.this.name)
                    carried.add(rename.alias_or_name)
            generated |= carried
        elif _references(expression) & inherited:
            output = expression.alias_or_name
            if not output or output not in fields:
                raise ValueError("unknown projection value has no bound output alias")
            generated.add(output)
    # Projection expressions may carry unknown values. Other clauses can
    # change cardinality based on those values and cannot use NULL stand-ins.
    # Include output aliases because DuckDB permits them in WHERE/ORDER BY.
    read_unknown = inherited | generated
    for key, value in select.args.items():
        if key == "expressions":
            continue
        clauses = value if isinstance(value, list) else [value]
        if (key == "distinct" and value and generated) or any(
            isinstance(clause, exp.Expression)
            and _references(clause) & read_unknown
            for clause in clauses
        ):
            raise ValueError(
                "relational expression depends on semantic generated columns"
            )
    return generated & fields


def prepare_population_estimates(problem):
    """Attach independent EXPLAIN statements to supported bound operators.

    ``population_estimate_sql`` estimates the output, and
    ``population_input_sql`` contains its population input statements. Their
    ratio is a local prior, to be rebased on each candidate's actual input
    cardinalities. Unsupported statements have neither key and carry an
    explicit ``population_estimate_unsupported`` reason instead.
    """
    if problem.query.get("kind") != "QueryIR":
        return
    logical = problem.logical
    sources = {r["node_id"]: r for r in problem.query.get("relations", [])}
    bodies, dependencies, unknowns, failures = {}, {}, {}, {}
    active = set()

    def reference(node_id):
        return _name(f"__population_{node_id}")

    def inputs(node):
        children = node.get("children", [])
        if "sampling_passthrough_input" in node:
            return [children[node["sampling_passthrough_input"]]]
        return children

    def compile_node(node_id):
        if node_id in bodies:
            return
        if node_id in failures:
            raise ValueError(failures[node_id])
        if node_id in active:
            raise ValueError("recursive population relation is unsupported")
        active.add(node_id)
        try:
            node = logical[node_id]
            children = inputs(node)
            for child in children:
                compile_node(child)
            kind = node.get("kind", node.get("type", "")).upper()
            schema = node.get("output_schema", [])
            fields = {_field(field) for field in schema}
            inherited = set().union(*(unknowns[c] for c in children))
            generated = inherited & fields
            template = node.get("sampling_sql", "")

            if not children:
                template = sources.get(node_id, {}).get("source_sql", "")
                if not template:
                    raise ValueError("population relation has no bound source_sql")
                sql = _TRANSPORT_ID.sub("NULL::VARCHAR __sample_id", template)
                generated = set()
            elif "sampling_passthrough_input" in node:
                sql = "SELECT * FROM " + reference(children[0])
            elif kind == "SEMANTIC_FILTER":
                sql = "SELECT * FROM " + reference(children[0])
            elif kind in ("SEMANTIC_JOIN", "SEMANTIC_NESTED_LOOP_JOIN"):
                if len(children) == 1:
                    sql = "SELECT * FROM " + reference(children[0])
                else:
                    sql = (
                        "SELECT l.* EXCLUDE (__sample_id),"
                        "r.* EXCLUDE (__sample_id) FROM "
                        + reference(children[0])
                        + " l CROSS JOIN "
                        + reference(children[1])
                        + " r"
                    )
            elif kind == "SEMANTIC_PROJECTION":
                source_fields = {
                    _field(f) for f in logical[children[0]].get("output_schema", [])
                }
                generated = (inherited | (fields - source_fields)) & fields
                projected = [
                    (
                        f"NULL::{field['type']} AS {_name(_field(field))}"
                        if _field(field) not in source_fields
                        else _name(_field(field))
                    )
                    for field in schema
                ]
                sql = (
                    "SELECT " + ",".join(projected) + " FROM " + reference(children[0])
                )
            else:
                required = {
                    json.dumps(field, separators=(",", ":"))
                    if isinstance(field, (list, tuple))
                    else _field(field)
                    for field in node.get("required_columns", [])
                }
                if kind == "DISTINCT":
                    required |= fields
                if not template or _SEMANTIC_CALL.search(template):
                    raise ValueError(
                        "population relation has no model-free bound SQL template"
                    )
                sql = _TRANSPORT_ID.sub("NULL::VARCHAR __sample_id", template)
                parsed = _select(sql)
                if kind == "PROJECTION":
                    generated = _projection_unknowns(parsed, inherited, fields)
                elif (
                    kind != "CTE_SCAN"
                    and (required | _references(parsed)) & inherited
                ):
                    raise ValueError(
                        "relational expression depends on semantic generated columns"
                    )
                if len(children) == 1:
                    if "$INPUT" not in sql:
                        raise ValueError("unary population SQL requires $INPUT")
                    sql = sql.replace("$INPUT", reference(children[0]))
                elif len(children) == 2:
                    if "$LEFT" not in sql or "$RIGHT" not in sql:
                        raise ValueError(
                            "binary population SQL requires $LEFT and $RIGHT"
                        )
                    sql = sql.replace("$LEFT", reference(children[0])).replace(
                        "$RIGHT", reference(children[1])
                    )
                else:
                    raise ValueError("population SQL supports at most two inputs")
                if kind == "CTE_SCAN":
                    producer = logical[children[0]].get("output_schema", [])
                    generated = {
                        _field(right)
                        for left, right in zip(producer, schema, strict=True)
                        if _field(left) in inherited
                    }
            if _SEMANTIC_CALL.search(sql):
                raise ValueError("population SQL contains a semantic function")
            # Every stage exposes only bound value columns. Transport IDs must
            # not add windows, change DISTINCT keys, or leak across join stages.
            projection = ["NULL::VARCHAR __sample_id"] + [
                _name(_field(field)) for field in schema
            ]
            bodies[node_id] = "SELECT " + ",".join(projection) + " FROM (" + sql + ")"
            dependencies[node_id] = children
            unknowns[node_id] = generated
        except (KeyError, IndexError, TypeError, ValueError) as error:
            failures[node_id] = str(error)
            raise ValueError(str(error)) from error
        finally:
            active.remove(node_id)

    def statement(node_id):
        compile_node(node_id)
        ordered, visited = [], set()

        def visit(current):
            if current in visited:
                return
            visited.add(current)
            for child in dependencies[current]:
                visit(child)
            ordered.append(current)

        visit(node_id)
        definitions = [
            reference(current) + " AS NOT MATERIALIZED (" + bodies[current] + ")"
            for current in ordered
        ]
        return _estimation_filter_sql(
            "WITH " + ",".join(definitions) + " SELECT * FROM " + reference(node_id)
        )

    for op in problem.all_ops.values():
        op.pop("population_estimate_sql", None)
        op.pop("population_input_sql", None)
        op.pop("population_estimate_unsupported", None)
        if op["operator_id"] in problem.ops:
            continue
        node_id = op["node_id"]
        try:
            output_sql = statement(node_id)
            input_sql = [statement(child) for child in inputs(logical[node_id])]
            op["population_estimate_sql"] = output_sql
            op["population_input_sql"] = input_sql
        except ValueError as error:
            op["population_estimate_unsupported"] = str(error)
