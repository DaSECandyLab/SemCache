"""Translate explicit SemCache tasks into native scalar semantic syntax."""

import re

import sqlglot
from sqlglot import exp

from experiments.validation.native_api import condition_instruction

_HIGHER_COMPARISON = re.compile(
    r"\AThe [^\n?]+ is higher than (?P<threshold>[+-]?\d+(?:\.\d+)?)\."
    r"\nInputs: [^:\n]+: (?P<placeholder>\{\d+\})\Z"
)


def _task(function, *, clarify_comparisons=False):
    arguments = function.expressions
    if not arguments or not isinstance(arguments[0], exp.Literal):
        raise ValueError("A semantic task must be a literal")
    values = (
        arguments[1:-1]
        if function.name.lower() == "semantic_project"
        else arguments[1:]
    )
    names = []
    for value in values:
        while isinstance(value, (exp.Cast, exp.TryCast)):
            value = value.this
        if not isinstance(value, exp.Column):
            raise ValueError("Native placeholder requires an explicit column")
        # Native template binders take identifier text, not SQL quoting.
        names.append(".".join(part.name for part in value.parts))

    def replace(match):
        index = int(match.group(1))
        if index >= len(names):
            raise ValueError("Task placeholder has no corresponding field")
        return "{" + names[index] + "}"

    task = arguments[0].this
    if clarify_comparisons and (match := _HIGHER_COMPARISON.fullmatch(task)):
        task += (
            "\n\nQuestion: Is the numerical comparison "
            + match["placeholder"]
            + " > "
            + match["threshold"]
            + " true?"
        )
    return re.sub(r"\{(\d+)\}", replace, task)


def _appstore_prefix(tree):
    """Expose id and review before semantic filtering to retain native bindings.

    A normal, inlinable CTE avoids Morrila's lost projection binding on a join.
    No materialization or optimizer setting is forced, and join bags are kept.
    """
    if {t.name.casefold() for t in tree.find_all(exp.Table)} != {
        "user_reviews",
        "playstore",
    }:
        return tree
    if (
        tree.args.get("with_")
        or len(tree.expressions) != 1
        or not tree.args.get("where")
    ):
        raise ValueError("Unrecognized Appstore relational contract")
    conditions = (
        list(tree.args["where"].this.flatten())
        if isinstance(tree.args["where"].this, exp.And)
        else [tree.args["where"].this]
    )
    semantic = [
        c
        for c in conditions
        if isinstance(c, exp.Anonymous) and c.name.casefold() == "semantic_filter"
    ]
    ordinary = [c for c in conditions if c not in semantic]
    if len(semantic) != 3 or len(ordinary) != 1:
        raise ValueError("Appstore needs exactly three conjunctive semantic predicates")
    base = tree.copy()
    base.set(
        "expressions",
        [tree.expressions[0].copy(), exp.column("Translated_Review", table="ur")],
    )
    base.set("where", exp.Where(this=ordinary[0].copy()))

    def unqualify(n):
        if isinstance(n, exp.Column) and n.name.casefold() == "translated_review":
            n.set("table", None)
        return n

    filters = [c.copy().transform(unqualify) for c in semantic]
    result = (
        exp.select(tree.expressions[0].alias_or_name)
        .from_("__appstore_candidates")
        .where(exp.and_(*filters))
    )
    result.set(
        "with_",
        exp.With(
            expressions=[
                exp.CTE(
                    this=base,
                    alias=exp.TableAlias(
                        this=exp.to_identifier("__appstore_candidates")
                    ),
                )
            ]
        ),
    )
    return result


def native_sql(
    sql,
    system,
    *,
    evaluate_conditions=False,
    clarify_comparisons=False,
    rowid_column=None,
):
    if system not in {"plop", "sema"}:
        raise ValueError(f"Unknown SQL system: {system}")
    tree = sqlglot.parse_one(sql, read="duckdb")
    if system == "sema":
        from experiments.validation.runtime.filter_groups import merge_filter_stages

        tree = merge_filter_stages(tree)
    if rowid_column is not None:
        for column in tree.find_all(exp.Column):
            if column.name.casefold() == "rowid" and column.table.casefold() == "ur":
                column.set("this", exp.to_identifier(rowid_column))
    if system == "plop":
        tree = _appstore_prefix(tree)
    replacements = {}

    def native_function(node):
        if not isinstance(node, exp.Anonymous) or node.name.lower() not in {
            "semantic_filter",
            "semantic_join",
            "semantic_project",
        }:
            return node
        task = _task(
            node,
            clarify_comparisons=clarify_comparisons
            and node.name.lower() != "semantic_project",
        )
        if evaluate_conditions and node.name.lower() != "semantic_project":
            task = condition_instruction(task)
        literal = exp.Literal.string(task).sql(dialect="duckdb")
        kind = "SEMANTIC"
        field = None
        if node.name.lower() == "semantic_project":
            schema = node.expressions[-1]
            if not isinstance(schema, exp.Cast) or not isinstance(
                schema.args.get("to"), exp.DataType
            ):
                raise ValueError("Projection needs an explicit STRUCT schema")
            fields = schema.args["to"].expressions
            if len(fields) != 1 or not isinstance(fields[0], exp.ColumnDef):
                raise ValueError("Native projection supports one output field")
            field = fields[0]
            kind = "SEMANTIC_STRING"
            task += (
                "\nOutput protocol: If the requested value is missing, return exactly __SQL_NULL__. "
                "Otherwise return only the requested value. Preserve an empty string as empty."
            )
            literal = exp.Literal.string(task).sql(dialect="duckdb")
        rendered = f"{kind}({literal})" if system == "plop" else "s" + literal
        marker = f"__NATIVE_SEMANTIC_{len(replacements)}__"
        replacements[marker] = rendered
        result = exp.column(marker)
        if field is not None:
            # Native scalar string APIs have no typed nullable field schema.
            # Use an explicit missing-value encoding without conflating it with ''.
            result = exp.Nullif(
                this=result, expression=exp.Literal.string("__SQL_NULL__")
            )
            if field.args["kind"].sql() not in {"TEXT", "VARCHAR"}:
                # TRY_CAST converts invalid numeric model outputs to SQL NULL so aggregates can consume remaining rows.
                result = exp.TryCast(this=result, to=field.args["kind"].copy())
            result = exp.Anonymous(
                this="STRUCT_PACK",
                expressions=[exp.PropertyEQ(this=field.this.copy(), expression=result)],
            )
        return result

    text = tree.transform(native_function).sql(dialect="duckdb")
    for marker, rendered in replacements.items():
        text = text.replace(marker, rendered)
    return text
