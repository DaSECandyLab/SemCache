"""Read and audit retained manually maintained additions; never emit queries."""

from __future__ import annotations

import ast
import re

import sqlglot
from sqlglot import exp

from experiments.core.paths import QUERY_ROOT as ROOT
from experiments.validation.audits import source as inspect

CONTRACT = "selected28-extension-v1"
BOX_OFFICE_TASK = (
    "Extract the movie box office gross as a single numeric amount in US dollars (USD). "
    "Expand explicitly stated thousand, million and billion multipliers. "
    "A dollar sign in this dataset means USD. Return null for missing values, N/A, "
    "ranges, ambiguous amounts, or other currencies; do not guess an exchange rate. "
    "Return no currency symbol, grouping separators, or unit suffix."
)
AGGREGATES = {
    "movies/Q5": {"function": "SUM", "column": "BoxOfficeUSD"},
    "movies/Q7": {"function": "MAX", "column": "IMDB_rating"},
    "movies/Q8": {"function": "COUNT", "column": None},
}


def task_key(task):
    task = task.split("\nOutput protocol:", 1)[0].split("\nInputs:", 1)[0]
    task = re.sub(r"\{([^{}]+)\}", lambda m: m[1].split(".")[-1], task)
    return " ".join(task.split())


def semantic_tree(text, system):
    if system == "sema":
        text = re.sub(
            r"\bs'((?:''|[^'])*)'",
            lambda m: (
                "__native("
                + exp.Literal.string(m[1].replace("''", "'")).sql(dialect="duckdb")
                + ")"
            ),
            text,
        )
    return sqlglot.parse_one(text, read="duckdb")


def sql_operations(text, system):
    tree = semantic_tree(text, system)
    names = {
        "semantic_filter",
        "semantic_project",
        "semantic_string",
        "semantic",
        "__native",
    }
    functions = [n for n in tree.find_all(exp.Anonymous) if n.name.lower() in names]

    # Inner query blocks precede outer blocks; WHERE precedes SELECT in one
    # block. Preserve sibling AND order rather than using raw AST depth.
    def order(fn):
        owner = next(n for n in ancestors(fn) if isinstance(n, exp.Select))
        before_select = []
        for parent in ancestors(fn):
            if parent is owner:
                break
            before_select.append(parent)
        return (
            -sum(1 for _ in ancestors(owner)),
            0
            if any(isinstance(n, (exp.Where, exp.Join)) for n in before_select)
            else 1,
            fn.expressions[0].meta.get("start", 0),
        )

    functions.sort(key=order)
    result = []
    for fn in functions:
        kind = fn.name.lower()
        task = fn.expressions[0].this
        if kind in {"semantic_filter", "semantic_project"}:
            projection = kind == "semantic_project"
            values = fn.expressions[1:-1] if projection else fn.expressions[1:]
            columns = [c.name for value in values for c in value.find_all(exp.Column)]
            outputs = (
                [f.name for f in fn.expressions[-1].args["to"].expressions]
                if projection
                else []
            )
            task = re.sub(r"\{(\d+)\}", lambda m: columns[int(m[1])], task)
        else:
            refs = re.findall(r"\{([^{}]+)\}", task.split("\nOutput protocol:", 1)[0])
            columns = list(dict.fromkeys(c.split(".")[-1] for c in refs))
            alias = next(
                (
                    p
                    for p in ancestors(fn)
                    if isinstance(p, (exp.Alias, exp.Select, exp.Where))
                ),
                None,
            )
            projection = (
                kind == "semantic_string"
                or kind == "__native"
                and isinstance(alias, exp.Alias)
            )
            outputs = [alias.alias] if projection else []
        result.append(
            {
                "method": "semantic_map" if projection else "semantic_filter",
                "task": task_key(task),
                "inputs": columns,
                "outputs": outputs,
            }
        )
    return result


def ancestors(node):
    while node.parent is not None:
        node = node.parent
        yield node


def expected_operations(record):
    path = ROOT / record["source"]
    if path.suffix == ".sql":
        return sql_operations(path.read_text(), "sema")
    ops = inspect.source_operations(path)
    if record["id"] in AGGREGATES:
        assert ops[-1]["method"] == "semantic_reduce"
        ops.pop()
    if record["id"] == "movies/Q5":
        ops.append(
            {
                "method": "semantic_map",
                "task": BOX_OFFICE_TASK,
                "inputs": ["BoxOffice"],
                "outputs": ["BoxOfficeUSD"],
            }
        )
    return [{**op, "task": task_key(op["task"])} for op in ops]


def same_operations(actual, expected):
    """Allow a safe schedule change for independent semantic stages.

    LOTUS keeps source map/filter operations in the workload, but may move a
    map after a selective filter when neither stage consumes the other's
    output.  The operation contract (task, inputs and outputs) remains exact;
    only this dependency-preserving order is optimized.
    """
    actual = list(actual)
    expected = list(expected)
    if actual == expected:
        return True
    if sorted(
        actual,
        key=lambda x: (
            x.get("method"),
            x.get("task"),
            tuple(x.get("inputs", [])),
            tuple(x.get("outputs", [])),
        ),
    ) != sorted(
        expected,
        key=lambda x: (
            x.get("method"),
            x.get("task"),
            tuple(x.get("inputs", [])),
            tuple(x.get("outputs", [])),
        ),
    ):
        return False
    # Bubble each expected operation into position, crossing only stages with
    # disjoint produced/consumed fields.
    for index, target in enumerate(expected):
        try:
            position = next(i for i in range(index, len(actual)) if actual[i] == target)
        except StopIteration:
            return False
        while position > index:
            left, right = actual[position - 1], actual[position]
            if set(left.get("outputs", [])) & set(right.get("inputs", [])):
                return False
            if set(right.get("outputs", [])) & set(left.get("inputs", [])):
                return False
            actual[position - 1], actual[position] = right, left
            position -= 1
    return actual == expected


def audit_records(records):
    result = []
    for record in records:
        errors = []
        path = ROOT / record["query_file"]
        try:
            for file_key, hash_key in (
                ("query_file", "sha256"),
                ("metadata_file", "metadata_sha256"),
                ("source", "source_sha256"),
            ):
                if inspect.digest(ROOT / record[file_key]) != record[hash_key]:
                    errors.append(file_key + " hash differs")
            expected = expected_operations(record)
            if path.suffix == ".sql":
                actual = sql_operations(path.read_text(), record["system"])
            else:
                actual = [
                    {**op, "task": task_key(op["task"])}
                    for op in inspect.python_operations(path, record["system"])
                ]
                tree = ast.parse(path.read_text())
                binds = {
                    n.targets[0].id: n.value
                    for n in tree.body
                    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                }
                aggregate = (
                    ast.literal_eval(binds["AGGREGATE"])
                    if "AGGREGATE" in binds
                    else None
                )
                if aggregate != AGGREGATES.get(record["id"]):
                    errors.append("Python aggregate differs")
            if not same_operations(actual, expected):
                errors.append(
                    "Semantic tasks, bindings, outputs or dependencies differ"
                )
            if record["aggregate"] != AGGREGATES.get(record["id"]):
                errors.append("Metadata aggregate differs")
            if len(actual) < 2:
                errors.append("Fewer than two logical semantic operations")
        except Exception as exc:
            actual = []
            errors.append(f"{type(exc).__name__}: {exc}")
        result.append(
            {
                "id": record["id"],
                "system": record["system"],
                "passed": not errors,
                "aggregate": record["aggregate"],
                "query_file": record["query_file"],
                "execution_mode": record["execution_mode"],
                "logical_operations": actual,
                "errors": errors,
            }
        )
    return result
