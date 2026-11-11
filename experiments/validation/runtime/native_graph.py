"""Compile selected SQL into one graph; semantic execution belongs to the system.

SQL is used only for deterministic scalar expressions and the source workload's
explicit DISTINCT/ORDER/LIMIT context boundaries. No semantic stage is executed
while constructing the graph. Equality-constrained semantic joins evaluate the
predicate on matching pairs; unconstrained joins remain binary operators.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field

import pandas as pd
import sqlglot
from sqlglot import exp

SEMANTIC = {"semantic_filter", "semantic_join"}


def q(name):
    return '"' + name.replace('"', '""') + '"'


@dataclass(eq=False)
class Node:
    op: str
    fields: dict
    children: tuple = ()
    options: dict = field(default_factory=dict)
    data: object = None


@dataclass
class Relation:
    node: Node
    # Ordered SQL-visible names and unique internal fields; duplicate names stay.
    columns: list


class Scalar:
    """DuckDB's scalar/NULL/type semantics, evaluated in a thread-local host."""

    local = threading.local()
    __name__ = "deterministic_sql_expression"

    def __init__(self, expressions, fields):
        self.expressions = expressions
        self.__name__ = (
            "sql_scalar_"
            + hashlib.sha256(repr((expressions, fields)).encode()).hexdigest()[:16]
        )
        used = {
            c.name
            for e in expressions.values()
            for c in sqlglot.parse_one(e, read="duckdb").find_all(exp.Column)
        }
        self.fields = {k: v for k, v in fields.items() if k in used}
        source = ",".join(
            f"CAST(? AS {kind}) AS {q(name)}" for name, kind in self.fields.items()
        )
        self.projection = "SELECT " + ",".join(
            f"{text} AS {q(name)}" for name, text in expressions.items()
        )
        self.sql = self.projection
        if source:
            self.sql += " FROM (SELECT " + source + ")"

    def connection(self):
        import duckdb

        if not hasattr(self.local, "connection"):
            self.local.connection = duckdb.connect(config={"threads": 1})
        return self.local.connection

    def frame(self, frame):
        """Evaluate deterministic expressions in one typed, order-preserving scan."""
        connection = self.connection()
        source = (
            ",".join(
                f"CAST({q(name)} AS {kind}) AS {q(name)}"
                for name, kind in self.fields.items()
            )
            or "1 AS constant"
        )
        connection.register("__comparison_scalar_input", frame)
        try:
            rows = connection.execute(
                self.projection
                + " FROM (SELECT "
                + source
                + " FROM __comparison_scalar_input)"
            ).fetchall()
        finally:
            connection.unregister("__comparison_scalar_input")
        return pd.DataFrame(
            rows, columns=list(self.expressions), index=frame.index, dtype=object
        )

    def __call__(self, row):
        values = []
        for name in self.fields:
            value = row[name]
            if value is pd.NA:
                value = None
            elif type(value).__module__.startswith("numpy") and hasattr(value, "item"):
                value = value.item()
            values.append(value)
        result = self.connection().execute(self.sql, values).fetchone()
        return dict(zip(self.expressions, result, strict=True))


class Compiler:
    def __init__(self, tables):
        import duckdb

        self.tables = tables
        self.con = duckdb.connect(config={"threads": 1})
        self.serial = 0
        self.cte_cache = {}
        self.cte_cache_hits = 0
        self.sources = []
        for i, (name, frame) in enumerate(tables.items()):
            schema = frame.attrs.get("duckdb_schema")
            if not schema:
                raise ValueError("Native SQL graph requires frozen SQL column types")
            self.con.register(f"input{i}", frame)
            columns = ",".join(
                f"CAST({q(n)} AS {kind}) AS {q(n)}" for n, kind in schema
            )
            # Bind declared SQL types without copying every complete input.
            # Explicit context queries scan only the tables they actually use.
            self.con.execute(f"CREATE VIEW {q(name)} AS SELECT {columns} FROM input{i}")

    def name(self):
        self.serial += 1
        return f"field{self.serial}"

    def source(self, frame, name):
        schema = frame.attrs["duckdb_schema"]
        rename = {n: self.name() for n, _ in schema}
        fields = {rename[n]: kind for n, kind in schema}
        data = frame.rename(columns=rename).copy()
        data.attrs["duckdb_schema"] = list(fields.items())
        node = Node("scan", fields, options={"name": name}, data=data)
        self.sources.append(node)
        return Relation(node, [(n, rename[n]) for n, _ in schema])

    def resolve(self, expression, scope):
        def column(node):
            if not isinstance(node, exp.Column) or isinstance(node.this, exp.Star):
                return node
            matches = [
                internal
                for alias, columns in scope
                for label, internal in columns
                if label.casefold() == node.name.casefold()
                and (not node.table or node.table.casefold() == alias.casefold())
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Unresolved or ambiguous native SQL field: {node.sql()}"
                )
            return exp.column(matches[0], quoted=True)

        return expression.copy().transform(column)

    def mapped(self, node, expressions):
        fields = dict(node.fields)
        declarations = ",".join(
            f"CAST(NULL AS {kind}) AS {q(n)}" for n, kind in fields.items()
        )
        sql = "SELECT " + ",".join(f"{v} AS {q(k)}" for k, v in expressions.items())
        if declarations:
            sql += " FROM (SELECT " + declarations + ")"
        cursor = self.con.execute(sql)
        fields.update((c[0], str(c[1])) for c in cursor.description)
        return Node("map", fields, (node,), {"expressions": expressions})

    def arguments(self, node, expressions):
        """Lower pure SQL casts beside their inputs, without fixing filter order.

        Moving only deterministic argument preparation through relational nodes
        avoids making a single-side predicate artificially depend on a join.
        All semantic operators remain for the native optimizer to arrange.
        """
        dependencies = {
            c.name
            for text in expressions.values()
            for c in sqlglot.parse_one(text, read="duckdb").find_all(exp.Column)
        }
        if node.op in {"scan", "context"}:
            return self.mapped(node, expressions)
        if len(node.children) == 1 and dependencies <= set(node.children[0].fields):
            child = self.arguments(node.children[0], expressions)
            fields = {
                **node.fields,
                **{name: child.fields[name] for name in expressions},
            }
            options = dict(node.options)
            if node.op == "project":
                options["columns"] = list(
                    dict.fromkeys(options["columns"] + list(expressions))
                )
            return Node(node.op, fields, (child,), options)
        if node.op == "join":
            for index, child in enumerate(node.children):
                if dependencies <= set(child.fields):
                    changed = self.arguments(child, expressions)
                    children = list(node.children)
                    children[index] = changed
                    return Node(
                        node.op,
                        {
                            **node.fields,
                            **{name: changed.fields[name] for name in expressions},
                        },
                        tuple(children),
                        dict(node.options),
                    )
        return self.mapped(node, expressions)

    def predicate(self, node, condition, scope):
        if isinstance(condition, exp.Anonymous) and condition.name.lower() in SEMANTIC:
            args = condition.expressions
            if not isinstance(args[0], exp.Literal) or not args[0].is_string:
                raise ValueError("Semantic instructions must be frozen literals")
            expressions = {
                self.name(): self.resolve(a, scope).sql(dialect="duckdb")
                for a in args[1:]
            }
            mapped = self.arguments(node, expressions)
            return Node(
                "filter",
                mapped.fields,
                (mapped,),
                {"semantic": True, "task": args[0].this, "columns": list(expressions)},
            )
        if any(
            isinstance(f, exp.Anonymous) and f.name.lower() in SEMANTIC
            for f in condition.walk()
        ):
            raise ValueError(
                "Nonconjunctive semantic conditions need an explicit native translation"
            )
        expression = self.resolve(condition, scope).sql(dialect="duckdb")
        return Node(
            "filter",
            node.fields,
            (node,),
            {"semantic": False, "expression": expression},
        )

    @staticmethod
    def conjuncts(condition):
        if condition is None:
            return []
        if isinstance(condition, exp.And):
            return Compiler.conjuncts(condition.this) + Compiler.conjuncts(
                condition.expression
            )
        return [condition]

    def pure_context(self, query, definitions):
        def expand(node):
            if isinstance(node, exp.Table) and node.name.casefold() in definitions:
                inner = definitions[node.name.casefold()]
                return exp.Subquery(
                    this=inner.copy().transform(expand),
                    alias=exp.TableAlias(this=exp.to_identifier(node.alias_or_name)),
                )
            return node

        expanded = query.copy().transform(expand)
        if any(
            isinstance(f, exp.Anonymous) and f.name.lower() in SEMANTIC
            for f in expanded.walk()
        ):
            return None
        cursor = self.con.execute(expanded.sql(dialect="duckdb"))
        schema = [(c[0], str(c[1])) for c in cursor.description]
        frame = pd.DataFrame(
            cursor.fetchall(), columns=[n for n, _ in schema], dtype=object
        )
        frame.attrs["duckdb_schema"] = schema
        return self.source(frame, "explicit_sql_context")

    def select(self, query, inherited=None):
        if isinstance(query, exp.Subquery):
            query = query.this
        query = query.copy()
        definitions = dict(inherited or {})
        with_ = query.args.get("with_")
        if with_:
            definitions.update(
                (c.alias_or_name.casefold(), c.this) for c in with_.expressions
            )
            query.set("with_", None)
        has_context = any(query.args.get(k) for k in ("limit", "distinct", "order"))
        if has_context:
            context = self.pure_context(query, definitions)
            if context is not None:
                return context
        if any(query.args.get(key) for key in ("group", "having", "qualify", "offset")):
            raise ValueError(
                "This selected SQL graph compiler does not support grouped SQL"
            )

        def source(ref):
            if isinstance(ref, exp.Subquery):
                relation = self.select(ref.this, definitions)
            elif ref.name.casefold() in definitions:
                definition = definitions[ref.name.casefold()]

                # A CTE can be referenced more than once (often from two
                # downstream joins).  Cache by its canonical definition and
                # the canonical definitions it actually depends on.  The
                # dependency portion is needed for nested scopes: the same
                # text can mean different things when an outer CTE is shadowed.
                def canonical(expression):
                    return sqlglot.parse_one(
                        expression.sql(dialect="duckdb"), read="duckdb"
                    ).sql(dialect="duckdb", pretty=False)

                dependencies = {
                    table.name.casefold()
                    for table in definition.find_all(exp.Table)
                    if table.name.casefold() in definitions
                }
                scope = tuple(
                    (name, canonical(definitions[name]))
                    for name in sorted(dependencies)
                )
                key = (canonical(definition), scope)
                relation = self.cte_cache.get(key)
                if relation is None:
                    relation = self.select(definition, definitions)
                    self.cte_cache[key] = relation
                else:
                    self.cte_cache_hits += 1
            else:
                frame = self.tables[ref.name].copy()
                schema = list(frame.attrs["duckdb_schema"])
                frame["rowid"] = list(range(len(frame)))
                frame.attrs["duckdb_schema"] = schema + [("rowid", "BIGINT")]
                relation = self.source(frame, ref.name)
            return ref.alias_or_name, relation

        alias, left = source(query.args["from_"].this)
        scope = [(alias, left.columns)]
        current = left.node
        all_columns = list(left.columns)
        for join in query.args.get("joins") or []:
            if (
                join.args.get("side")
                or join.args.get("using")
                or join.args.get("method")
            ):
                raise ValueError("Only the selected inner/cross joins are translated")
            alias, right = source(join.this)
            scope.append((alias, right.columns))
            left_fields = set(current.fields)
            right_fields = set(right.node.fields)
            left_input = current
            ordinary_join = False
            fields = {**current.fields, **right.node.fields}
            conditions = self.conjuncts(join.args.get("on"))
            keys = []
            remaining = []
            for condition in conditions:
                resolved = self.resolve(condition, scope)
                if (
                    isinstance(resolved, exp.EQ)
                    and isinstance(resolved.this, exp.Column)
                    and isinstance(resolved.expression, exp.Column)
                ):
                    a, b = resolved.this.name, resolved.expression.name
                    if a in right.node.fields and b in current.fields:
                        a, b = b, a
                    if a in current.fields and b in right.node.fields:
                        keys.append((a, b))
                        continue
                remaining.append(condition)
            semantic = None
            for condition in remaining:
                if (
                    isinstance(condition, exp.Anonymous)
                    and condition.name.lower() == "semantic_join"
                ):
                    semantic = condition
                    break
            if semantic is not None:
                args = semantic.expressions
                expressions = {
                    self.name(): self.resolve(a, scope).sql(dialect="duckdb")
                    for a in args[1:]
                }
                left_expr = {}
                right_expr = {}
                for name, text in expressions.items():
                    dependencies = {
                        c.name
                        for c in sqlglot.parse_one(text, read="duckdb").find_all(
                            exp.Column
                        )
                    }
                    if dependencies <= set(current.fields):
                        left_expr[name] = text
                    elif dependencies <= set(right.node.fields):
                        right_expr[name] = text
                    else:
                        raise ValueError(
                            "Semantic join argument mixes both input sides"
                        )
                if keys:
                    # SQL's ordinary key predicates are part of the semantic
                    # join candidate relation.  Materialize that exact join
                    # first, then evaluate the semantic condition as a filter
                    # over the matching pairs.  Leaving the key predicates
                    # until after sem_join turns an equality-constrained join
                    # into a full Cartesian product and can multiply model
                    # calls by |left|*|right|.
                    current = Node(
                        "join",
                        {**current.fields, **right.node.fields},
                        (current, right.node),
                        {"semantic": False, "keys": keys},
                    )
                    remaining.remove(semantic)
                    current = self.predicate(current, semantic, scope)
                    current.options["source_operator"] = "semantic_join"
                    current.options["candidate_relation"] = "ordinary_inner_join"
                else:
                    left_node = (
                        self.arguments(current, left_expr) if left_expr else current
                    )
                    right_node = (
                        self.arguments(right.node, right_expr)
                        if right_expr
                        else right.node
                    )
                    current = Node(
                        "join",
                        {**left_node.fields, **right_node.fields},
                        (left_node, right_node),
                        {
                            "semantic": True,
                            "task": args[0].this,
                            "columns": list(expressions),
                            "left_columns": list(left_expr),
                            "right_columns": list(right_expr),
                        },
                    )
                    remaining.remove(semantic)
            else:
                ordinary_join = True
                current = Node(
                    "join",
                    fields,
                    (current, right.node),
                    {"semantic": False, "keys": keys},
                )
            # A unary semantic predicate in an INNER/CROSS join condition can
            # be evaluated on its owning side before the join.  This preserves
            # the SQL bag (including duplicate rows and NULL key behavior) and
            # avoids evaluating a customer-only predicate once for every order
            # or lineitem pair.  Predicates that mention both sides remain on
            # the joined candidate relation.
            deferred = []
            for condition in remaining:
                if (
                    ordinary_join
                    and isinstance(condition, exp.Anonymous)
                    and condition.name.lower() == "semantic_filter"
                ):
                    args = condition.expressions
                    dependencies = {
                        c.name
                        for value in args[1:]
                        for c in self.resolve(value, scope).find_all(exp.Column)
                    }
                    if dependencies <= left_fields:
                        left_input = self.predicate(left_input, condition, scope)
                        continue
                    if dependencies <= right_fields:
                        filtered = self.predicate(right.node, condition, scope)
                        right = Relation(filtered, right.columns)
                        continue
                deferred.append(condition)
            if ordinary_join:
                current = Node(
                    "join",
                    {**left_input.fields, **right.node.fields},
                    (left_input, right.node),
                    {"semantic": False, "keys": keys},
                )
            for condition in deferred:
                current = self.predicate(current, condition, scope)
            all_columns += right.columns
        where = query.args.get("where")
        for condition in self.conjuncts(where.this if where else None):
            current = self.predicate(current, condition, scope)
        columns = []
        for expression in query.expressions:
            is_star = (
                isinstance(expression, exp.Star)
                or isinstance(expression, exp.Column)
                and isinstance(expression.this, exp.Star)
            )
            if is_star:
                expanded = [
                    (label, internal)
                    for alias, cols in scope
                    for label, internal in cols
                    if not isinstance(expression, exp.Column)
                    or not expression.table
                    or expression.table.casefold() == alias.casefold()
                ]
            else:
                value = (
                    expression.this if isinstance(expression, exp.Alias) else expression
                )
                if not isinstance(value, exp.Column):
                    raise ValueError(
                        "Selected SQL exports require column-only ordinary projections"
                    )
                expanded = [(expression.alias_or_name, self.resolve(value, scope).name)]
            for label, old in expanded:
                if label.casefold() == "rowid" and is_star:
                    continue
                # SQL aliases bind names to existing values; copying every
                # field through an extra native UDF would obstruct rewrites.
                columns.append((label, old))
        selected = {name: current.fields[name] for _, name in columns}
        current = Node("project", selected, (current,), {"columns": list(selected)})
        if has_context:
            order = []
            output_scope = [("", columns)]
            if query.args.get("order"):
                for term in query.args["order"].expressions:
                    value = self.resolve(term.this, output_scope)
                    if not isinstance(value, exp.Column):
                        raise ValueError(
                            "Selected context ordering requires a projected column"
                        )
                    order.append(
                        (
                            value.name,
                            not bool(term.args.get("desc")),
                            bool(term.args.get("nulls_first")),
                        )
                    )
            limit = query.args.get("limit")
            current = Node(
                "context",
                selected,
                (current,),
                {
                    "distinct": bool(query.args.get("distinct")),
                    "order": order,
                    "limit": int(limit.expression.this) if limit else None,
                    "reason": "explicit source SQL DISTINCT/ORDER/LIMIT boundary",
                },
            )
        return Relation(current, columns)

    def compile(self, sql):
        try:
            return self.select(sqlglot.parse_one(sql, read="duckdb"))
        finally:
            self.con.close()


def describe(node, _seen=None):
    """Serialize the graph while retaining shared CTE edges as references."""
    if _seen is None:
        _seen = {}
    identity = id(node)
    if identity in _seen:
        return {"ref": _seen[identity]}
    node_id = len(_seen) + 1
    _seen[identity] = node_id
    return {
        "node_id": node_id,
        "operator": node.op,
        "fields": node.fields,
        "options": node.options,
        "source_rows": len(node.data) if node.op == "scan" else None,
        "children": [describe(child, _seen) for child in node.children],
    }
