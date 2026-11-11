"""The explicitly rewritten, six-system numeric workload (not source equivalence)."""

from copy import deepcopy

from sqlglot import exp

CONTRACT = "selected19-common-numeric-v4"
AGGREGATES = {
    "estate/Q9": {"function": "AVG", "column": "Price"},
    "estate/Q10": {"function": "MIN", "column": "Price"},
    "estate/Q11": {"function": "MIN", "column": "Price"},
    "estate/Q12": {"function": "AVG", "column": "Price"},
    "movies/Q9": {"function": "MAX", "column": "IMDB_rating"},
    "movies/Q10": {"function": "COUNT", "column": None},
    "movies/Q11": {"function": "AVG", "column": "RuntimeMinutes"},
}
PRICE_TASK = (
    "Extract the house price from the detail about the estate. "
    "Return a single numeric amount in Nigerian naira (NGN), with no currency symbol or unit suffix. "
    "Expand thousand/k, million/m and billion/b to their numeric multipliers. "
    "This dataset uses NGN when currency is omitted. "
    "If no unique price is stated, the price is a range, or a different currency is stated, return null. "
    "Do not infer a price from bedroom counts or other unrelated numbers."
)
RUNTIME_TASK = (
    "Extract the movie runtime as a single numeric duration in minutes. "
    "Convert explicitly stated hours and minutes to total minutes; a bare numeric runtime means minutes. "
    "If the runtime is missing, ambiguous, or a range, return null. Return no unit suffix."
)


def rewrite(query_id, source_operations):
    """Keep every source map/filter in order; replace only the seven reducers."""
    ops = deepcopy(source_operations)
    aggregate = deepcopy(AGGREGATES.get(query_id))
    if aggregate is None:
        return ops, None, []
    if ops[-1]["method"] != "semantic_reduce":
        raise ValueError(f"{query_id}: expected terminal source Reduce")
    ops.pop()
    numeric_columns = []
    if query_id.startswith("estate/"):
        price = next(
            op
            for op in ops
            if op["method"] == "semantic_map"
            and op["kwargs"]["output_columns"] == ["Price"]
        )
        price["kwargs"]["user_instruction"] = PRICE_TASK
        numeric_columns.append("Price")
    elif query_id == "movies/Q11":
        ops.append(
            {
                "method": "semantic_map",
                "kwargs": {
                    "user_instruction": RUNTIME_TASK,
                    "input_columns": ["Runtime"],
                    "output_columns": ["RuntimeMinutes"],
                },
            }
        )
        numeric_columns.append("RuntimeMinutes")
    return ops, aggregate, numeric_columns


def stabilize_limits(tree):
    """Define a shared subset for the source's otherwise arbitrary LIMITs."""
    tree = tree.copy()
    for select in tree.find_all(exp.Select):
        if not select.args.get("limit") or select.args.get("order"):
            continue
        keys = select.expressions
        if len(keys) == 1 and isinstance(keys[0], exp.Star):
            relation = select.args.get("from_")
            if (
                not relation
                or not isinstance(relation.this, exp.Table)
                or relation.this.name.casefold() != "customer"
            ):
                raise ValueError("No frozen sampling key for this SELECT * LIMIT")
            keys = [exp.column("c_custkey")]
        if not all(isinstance(k, exp.Column) for k in keys):
            raise ValueError("No frozen sampling keys for this LIMIT scope")
        select.set(
            "order",
            exp.Order(
                expressions=[
                    exp.Ordered(this=k.copy(), desc=False, nulls_first=False)
                    for k in keys
                ]
            ),
        )
    return tree
