"""No-LLM probes for usable source rows and prompt-input diversity."""

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope


def _relational(tree):
    """Count relational opportunities with semantic predicates accepting all inputs."""

    def replace(node):
        if isinstance(node, exp.Anonymous) and node.name.lower() in (
            "semantic_filter",
            "semantic_join",
        ):
            return exp.true()
        return node

    return tree.copy().transform(replace)


def input_coverage(
    connection, workload, *, minimum_distinct_inputs=128, minimum_result_rows=10
):
    """Measure per-operator distinct input tuples without invoking any model."""
    tree = sqlglot.parse_one(workload["sql"], read="duckdb")
    operators = []
    for scope in traverse_scope(tree):
        for node in scope.walk():
            if not isinstance(node, exp.Anonymous) or node.name.lower() not in (
                "semantic_filter",
                "semantic_join",
            ):
                continue
            arguments = node.expressions[1:]
            probe = scope.expression.copy()
            if tree.args.get("with_") is not None and probe is not tree:
                # A nested CTE scope still needs the enclosing WITH bindings.
                # SQL accepts unused definitions; no inference is performed.
                probe.set("with_", tree.args["with_"].copy())
            probe.set(
                "expressions",
                [
                    exp.alias_(argument.copy(), f"input_{i}")
                    for i, argument in enumerate(arguments)
                ],
            )
            # This scope measures semantic inputs, not final groups or LIMIT.
            for key in ("group", "having", "order", "limit", "distinct", "qualify"):
                probe.set(key, None)
            probe = _relational(probe)
            probe_sql = probe.sql(dialect="duckdb")
            valid = " AND ".join(
                f"input_{i} IS NOT NULL" for i in range(len(arguments))
            )
            count, distinct = connection.execute(
                "SELECT count(*),count(DISTINCT ("
                + ",".join(f"input_{i}" for i in range(len(arguments)))
                + f")) FROM ({probe_sql}) inputs WHERE {valid}"
            )[0]
            operators.append(
                {
                    "kind": node.name.lower(),
                    "rows": int(count),
                    "distinct_inputs": int(distinct),
                }
            )
    if not operators:
        raise ValueError(f"No semantic input opportunities: {workload['id']}")
    maximum = max(o["distinct_inputs"] for o in operators)
    # This rejects empty/singleton predicates and low-diversity label-only SQL.
    if maximum < minimum_distinct_inputs:
        raise ValueError(
            f"Too few distinct semantic inputs for {workload['id']}: {maximum}"
        )
    # Independent branch probes can have many inputs even if an enclosing
    # relational date/key predicate eliminates every final candidate.
    candidate_sql = _relational(tree).sql(dialect="duckdb")
    candidate_rows = int(
        connection.execute(f"SELECT count(*) FROM ({candidate_sql}) candidates")[0][0]
    )
    if candidate_rows < minimum_result_rows:
        raise ValueError(
            f"Too few relational result candidates for {workload['id']}: {candidate_rows} < {minimum_result_rows}"
        )
    return {
        "minimum_required_distinct_inputs": minimum_distinct_inputs,
        "max_operator_distinct_inputs": maximum,
        "operators": operators,
        "relational_result_candidates": candidate_rows,
        "method": "scope-local input opportunities and full relational result candidates; semantic filters and joins accept all; no model requests",
        "actual_requests_guaranteed": False,
    }
