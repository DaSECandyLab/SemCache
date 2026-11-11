"""Execute ordinary SQL in a disclosed host and semantic work in native API systems.

This is a query conversion adapter, not a replacement optimizer. It supports the
selected inner-join/CTE SQL and preserves bag rows, typed fields and SQL NULLs.
"""

from __future__ import annotations

import re

import sqlglot
from sqlglot import exp

SEMANTIC = {
    "semantic_filter": "filter",
    "semantic_join": "join",
    "semantic_project": "projection",
}


def quoted(name):
    return '"' + name.replace('"', '""') + '"'


class SQLPort:
    def __init__(self, connection, semantic):
        self.con = connection
        self.semantic = semantic
        self.serial = 0
        self.stages = []
        self.result_names = {}

    def table(self, prefix):
        self.serial += 1
        return f"__port_{prefix}_{self.serial}"

    def execute(self, sql):
        table = self.select(sqlglot.parse_one(sql, read="duckdb"), {})
        columns = self.con.execute("DESCRIBE SELECT * FROM " + quoted(table)).fetchall()
        expressions = ",".join(
            quoted(column[0]) + " AS " + quoted(original)
            for column, original in zip(columns, self.result_names[table])
        )
        return self.con.execute("SELECT " + expressions + " FROM " + quoted(table))

    def select(self, query, inherited):
        if isinstance(query, exp.Subquery):
            query = query.this
        if not isinstance(query, exp.Select):
            raise ValueError("SQL port requires SELECT queries")
        query = query.copy()
        scope = dict(inherited)
        with_ = query.args.get("with_")
        if with_:
            for cte in with_.expressions:
                scope[cte.alias_or_name.lower()] = self.select(cte.this, scope)
            query.set("with_", None)
        # Keep the physical relation name alongside the SQL alias.  This lets
        # us materialize a unary semantic predicate on its owning input before
        # the join candidate relation is built.
        sources = []
        joins = query.args.get("joins") or []
        for join in joins:
            if (
                join.args.get("side")
                or join.args.get("method")
                or join.args.get("using")
            ):
                raise ValueError("Only ordinary inner/cross ON joins are ported")
        from_ = query.args.get("from_")
        refs = ([from_.this] if from_ else []) + [j.this for j in joins]
        for ref in refs:
            alias = ref.alias_or_name
            if isinstance(ref, exp.Subquery):
                name = self.select(ref.this, scope)
            elif isinstance(ref, exp.Table):
                name = scope.get(ref.name.lower(), ref.name)
                alias = alias or ref.name
            else:
                raise ValueError(f"Unsupported FROM source: {type(ref).__name__}")
            if not alias:
                raise ValueError("Every subquery must have an alias")
            replacement = exp.Table(
                this=exp.to_identifier(name),
                alias=exp.TableAlias(this=exp.to_identifier(alias)),
            )
            ref.replace(replacement)
            columns = self.con.execute(
                "DESCRIBE SELECT * FROM " + quoted(name)
            ).fetchall()
            sources.append((alias, name, columns))
        mapping, plain, fields, source_exprs = {}, {}, [], []
        for alias, _name, columns in sources:
            for column, *_ in columns:
                private = f"__v_{len(fields)}"
                mapping[(alias.lower(), column.lower())] = private
                plain.setdefault(column.lower(), []).append(private)
                fields.append((alias, column, private))
                source_exprs.append(
                    exp.alias_(
                        exp.column(column, table=alias, quoted=True),
                        private,
                        quoted=True,
                    )
                )
            # Preserve SQL's current physical source rowid only when it is used.
            if any(
                c.table.lower() == alias.lower() and c.name.lower() == "rowid"
                for c in query.find_all(exp.Column)
            ):
                private = f"__v_rowid_{len(fields)}"
                mapping[(alias.lower(), "rowid")] = private
                source_exprs.append(
                    exp.alias_(
                        exp.column("rowid", table=alias, quoted=True),
                        private,
                        quoted=True,
                    )
                )

        def remap(node):
            if not isinstance(node, exp.Column) or isinstance(node.this, exp.Star):
                return node
            table, field = node.table.lower(), node.name.lower()
            target = mapping.get((table, field)) if table else None
            if target is None and not table:
                candidates = plain.get(field, [])
                if len(candidates) == 1:
                    target = candidates[0]
                elif len(candidates) > 1:
                    raise ValueError(f"Ambiguous SQL column {node.sql()}")
            if target is not None:
                return exp.column(target, quoted=True)
            # Struct-field references such as p.Genre are parsed as Columns.
            candidates = plain.get(table, [])
            if table and len(candidates) == 1:
                return exp.Dot(
                    this=exp.column(candidates[0], quoted=True),
                    expression=node.this.copy(),
                )
            return node

        predicates = []

        def semantic_key(function):
            return function.sql(dialect="duckdb")

        def referenced_aliases(function):
            """Return the input aliases referenced by a semantic filter.

            A predicate that reads one relation only is safe to evaluate before
            an inner/cross join.  Unqualified columns are resolved only when
            they identify exactly one source; ambiguous references stay on the
            joined candidate relation.
            """
            aliases = set()
            for value in function.expressions[1:]:
                for column in value.find_all(exp.Column):
                    if column.table:
                        aliases.add(column.table.lower())
                        continue
                    matches = [
                        alias.lower()
                        for alias, _name, columns in sources
                        if any(
                            column.name.lower() == item[0].lower() for item in columns
                        )
                    ]
                    if len(matches) != 1:
                        return set()
                    aliases.add(matches[0])
            return aliases

        def source_expression(node, alias):
            """Bind a source-only semantic argument to its table columns."""

            def bind(column):
                if not isinstance(column, exp.Column) or isinstance(
                    column.this, exp.Star
                ):
                    return column
                if column.table and column.table.lower() != alias.lower():
                    return column
                return exp.column(column.name, quoted=True)

            return node.copy().transform(bind).sql(dialect="duckdb")

        def is_conjunctive(function):
            parent = function.parent
            while parent is not None and parent is not query:
                if isinstance(parent, (exp.Where, exp.Join)):
                    return True
                if not isinstance(parent, (exp.And, exp.Paren)):
                    return False
                parent = parent.parent
            return True

        def apply_source_filter(alias, source_name, columns, function):
            args = function.expressions
            if (
                not args
                or not isinstance(args[0], exp.Literal)
                or not args[0].is_string
            ):
                raise ValueError("Literal semantic task required")
            task = args[0].this
            values = args[1:]
            if any(int(i) >= len(values) for i in re.findall(r"\{(\d+)\}", task)):
                raise ValueError("Unbound semantic task field")
            expressions = [source_expression(value, alias) for value in values]
            source_id = self.table("source_rows")
            column_sql = ", ".join(quoted(column[0]) for column in columns)
            preserves_rowid = any(
                column.table
                and column.table.lower() == alias.lower()
                and column.name.lower() == "rowid"
                for column in query.find_all(exp.Column)
            )
            source_rowid = ', rowid AS "__source_rowid"' if preserves_rowid else ""
            self.con.execute(
                f"CREATE TEMP TABLE {quoted(source_id)} AS "
                f"SELECT row_number() OVER ()-1 AS __source_id{source_rowid}, {column_sql} "
                f"FROM {quoted(source_name)}"
            )
            rows = self.con.execute(
                "SELECT __source_id"
                + ("," + ",".join(expressions) if expressions else "")
                + " FROM "
                + quoted(source_id)
                + " ORDER BY __source_id"
            ).fetchall()
            answers = (
                self.semantic("filter", task, [row[1:] for row in rows], [])
                if rows
                else []
            )
            if len(answers) != len(rows) or any(
                type(answer) is not bool for answer in answers
            ):
                raise ValueError(
                    "Semantic predicates require validated boolean outputs"
                )
            accepted = [row[0] for row, answer in zip(rows, answers) if answer]
            filtered = self.table("filtered_source")
            output_columns = (
                '"__source_rowid" AS "rowid", ' + column_sql
                if preserves_rowid
                else column_sql
            )
            if accepted:
                self.con.execute(
                    f"CREATE TEMP TABLE {quoted(filtered)} AS SELECT {output_columns} "
                    f"FROM {quoted(source_id)} WHERE __source_id IN (SELECT unnest(?))",
                    [accepted],
                )
            else:
                self.con.execute(
                    f"CREATE TEMP TABLE {quoted(filtered)} AS SELECT {output_columns} "
                    f"FROM {quoted(source_id)} WHERE FALSE"
                )
            self.stages.append(
                {
                    "kind": "filter",
                    "task": task,
                    "input_rows": len(rows),
                    "output_rows": len(accepted),
                    "output_fields": [],
                    "input_expressions": expressions,
                    "pushed_to_input": alias,
                }
            )
            return filtered

        # Find side-only filters before replacing them with TRUE in the
        # candidate query.  This is the relational equivalent of predicate
        # pushdown: it preserves inner-join bag semantics while preventing a
        # customer-only test from being repeated for every joined order.
        pushed = {}
        source_by_alias = {
            alias.lower(): (alias, name, columns) for alias, name, columns in sources
        }
        candidates_for_push = []
        if query.args.get("where"):
            candidates_for_push.extend(query.args["where"].find_all(exp.Anonymous))
        for join in query.args.get("joins") or []:
            if join.args.get("on"):
                candidates_for_push.extend(join.args["on"].find_all(exp.Anonymous))
        for function in candidates_for_push:
            if function.name.lower() != "semantic_filter":
                continue
            aliases = referenced_aliases(function)
            if len(aliases) != 1 or next(iter(aliases)) not in source_by_alias:
                continue
            # Only conjunctive predicates can move across an inner join.
            if not is_conjunctive(function):
                continue
            alias, source_name, columns = source_by_alias[next(iter(aliases))]
            key = semantic_key(function)
            pushed.setdefault(key, []).append((alias, source_name, columns, function))

        for entries in pushed.values():
            for alias, source_name, columns, function in entries:
                filtered = apply_source_filter(alias, source_name, columns, function)
                for index, (candidate_alias, _old_name, candidate_columns) in enumerate(
                    sources
                ):
                    if candidate_alias.lower() == alias.lower():
                        sources[index] = (candidate_alias, filtered, candidate_columns)
                        # ``ref.replace`` above may have detached the object
                        # kept in ``refs``; update only the live top-level
                        # source references, leaving nested scopes untouched.
                        live_refs = (
                            [query.args["from_"].this]
                            if query.args.get("from_")
                            else []
                        ) + [j.this for j in query.args.get("joins") or []]
                        for table in live_refs:
                            if (
                                isinstance(table, exp.Table)
                                and table.alias_or_name.lower() == alias.lower()
                            ):
                                table.set("this", exp.to_identifier(filtered))
                        break

        def strip(node):
            if isinstance(node, exp.Anonymous) and node.name.lower() in SEMANTIC:
                if SEMANTIC[node.name.lower()] == "projection":
                    raise ValueError("Semantic projections are only ported in SELECT")
                # Only conjunctive filters may move after candidate materialization.
                parent = node.parent
                while (
                    parent is not None
                    and parent is not query
                    and not isinstance(parent, (exp.Where, exp.Join))
                ):
                    if not isinstance(parent, (exp.And, exp.Paren)):
                        raise ValueError(
                            "Nonconjunctive semantic predicate is not ported"
                        )
                    parent = parent.parent
                key = semantic_key(node)
                entries = pushed.get(key) or []
                if entries:
                    # Consume one occurrence.  Identical predicates appearing
                    # twice must still make two model calls in the source
                    # workload, so this is occurrence-based rather than a set.
                    entries.pop(0)
                    return exp.true()
                predicates.append(node.copy())
                return exp.true()
            return node

        candidate = query.copy()
        if candidate.args.get("where"):
            candidate.set("where", candidate.args["where"].transform(strip))
        for join in candidate.args.get("joins") or []:
            if join.args.get("on"):
                join.set("on", join.args["on"].transform(strip))
        candidate.set(
            "expressions", source_exprs or [exp.alias_(exp.Literal.number(1), "__unit")]
        )
        for key in [
            "distinct",
            "group",
            "having",
            "qualify",
            "order",
            "limit",
            "offset",
        ]:
            candidate.set(key, None)
        name = self.table("candidates")
        self.con.execute(
            f"CREATE TEMP TABLE {quoted(name)} AS SELECT row_number() OVER ()-1 AS __candidate_id, * FROM ({candidate.sql(dialect='duckdb')}) candidate"
        )

        def apply(function):
            kind = SEMANTIC[function.name.lower()]
            args = function.expressions
            if (
                not args
                or not isinstance(args[0], exp.Literal)
                or not args[0].is_string
            ):
                raise ValueError("Literal semantic task required")
            task = args[0].this
            values = args[1:-1] if kind == "projection" else args[1:]
            fields_out = []
            sql_type = None
            if kind == "projection":
                contract = args[-1]
                if (
                    not isinstance(contract, exp.Cast)
                    or contract.args["to"].this != exp.DataType.Type.STRUCT
                ):
                    raise ValueError("Explicit STRUCT projection contract required")
                sql_type = contract.args["to"].sql(dialect="duckdb")
                fields_out = [
                    {"name": f.name, "type": f.args["kind"].sql(dialect="duckdb")}
                    for f in contract.args["to"].expressions
                ]
            if any(int(i) >= len(values) for i in re.findall(r"\{(\d+)\}", task)):
                raise ValueError("Unbound semantic task field")
            expressions = [
                v.copy().transform(remap).sql(dialect="duckdb") for v in values
            ]
            rows = self.con.execute(
                "SELECT __candidate_id"
                + ("," + ",".join(expressions) if expressions else "")
                + " FROM "
                + quoted(name)
                + " ORDER BY __candidate_id"
            ).fetchall()
            answers = (
                self.semantic(kind, task, [r[1:] for r in rows], fields_out)
                if rows
                else []
            )
            if len(answers) != len(rows):
                raise ValueError(
                    "Native semantic output cardinality differs from candidates"
                )
            metric = {
                "kind": kind,
                "task": task,
                "input_rows": len(rows),
                "output_fields": fields_out,
                "input_expressions": [v.sql(dialect="duckdb") for v in values],
            }
            if kind == "projection":
                column = f"__semantic_{len(self.stages)}"
                self.con.execute(
                    f"ALTER TABLE {quoted(name)} ADD COLUMN {quoted(column)} {sql_type}"
                )
                for (identifier, *_), answer in zip(rows, answers):
                    if not isinstance(answer, dict) or set(answer) != {
                        f["name"] for f in fields_out
                    }:
                        raise ValueError(
                            "Native projection violates the canonical field contract"
                        )
                    self.con.execute(
                        f"UPDATE {quoted(name)} SET {quoted(column)}=CAST(? AS {sql_type}) WHERE __candidate_id=?",
                        [answer, identifier],
                    )
                self.stages.append(metric)
                return exp.column(column, quoted=True)
            if any(type(answer) is not bool for answer in answers):
                raise ValueError(
                    "Semantic predicates require validated boolean outputs"
                )
            accepted = [row[0] for row, answer in zip(rows, answers) if answer]
            if accepted:
                self.con.execute(
                    f"DELETE FROM {quoted(name)} WHERE __candidate_id NOT IN (SELECT unnest(?))",
                    [accepted],
                )
            else:
                self.con.execute(f"DELETE FROM {quoted(name)}")
            metric["output_rows"] = len(accepted)
            self.stages.append(metric)
            return None

        for predicate in predicates:
            apply(predicate)
        output = []

        def project(node):
            if isinstance(node, exp.Anonymous) and node.name.lower() in SEMANTIC:
                if node.name.lower() != "semantic_project":
                    raise ValueError("Semantic predicates in SELECT are not ported")
                return apply(node)
            return remap(node)

        for expression in query.expressions:
            star = (
                expression
                if isinstance(expression, exp.Star)
                else expression.this
                if isinstance(expression, exp.Column)
                and isinstance(expression.this, exp.Star)
                else None
            )
            if star is not None:
                excluded = {c.name.lower() for c in star.args.get("except_") or []}
                if star.args.get("replace") or star.args.get("rename"):
                    raise ValueError("Star REPLACE/RENAME is not ported")
                for alias, column, private in fields:
                    if column.lower() not in excluded and (
                        not isinstance(expression, exp.Column)
                        or not expression.table
                        or alias.lower() == expression.table.lower()
                    ):
                        output.append(
                            exp.alias_(
                                exp.column(private, quoted=True), column, quoted=True
                            )
                        )
            else:
                changed = expression.copy().transform(project)
                if not isinstance(expression, exp.Alias) and isinstance(
                    expression, (exp.Column, exp.Dot)
                ):
                    changed = exp.alias_(changed, expression.output_name, quoted=True)
                output.append(changed)
        final = query.copy()
        final.set("expressions", output)
        final.set("from_", exp.From(this=exp.to_table(name)))
        final.set("joins", None)
        final.set("where", None)
        for key in ["group", "having", "qualify", "order"]:
            if final.args.get(key):
                final.set(key, final.args[key].transform(remap))
        result = self.table("result")
        text = final.sql(dialect="duckdb")
        self.result_names[result] = [
            column[0] for column in self.con.execute("DESCRIBE " + text).fetchall()
        ]
        self.con.execute(f"CREATE TEMP TABLE {quoted(result)} AS {text}")
        return result
