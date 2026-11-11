"""Scope-aware SQL fingerprints preserving dependencies while erasing names and constants."""

import hashlib

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

FINGERPRINT_VERSION = 1


def structural_fingerprint(sql):
    """Alpha-normalize a DuckDB AST without discarding its dependency edges."""
    statements = sqlglot.parse(sql, read="duckdb")
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        raise ValueError("A workload must contain exactly one SQL query")
    tree = statements[0]
    scopes = list(traverse_scope(tree))
    scope_numbers = {id(scope): index for index, scope in enumerate(scopes)}
    bindings, scope_sources, outputs = {}, {}, {}
    sources, fields, members, source_fields = {}, {}, {}, {}
    ctes = {
        cte.alias: f"cte{index}" for index, cte in enumerate(tree.find_all(exp.CTE))
    }

    def name(mapping, original, prefix):
        key = original.lower()
        if key not in mapping:
            mapping[key] = f"{prefix}{len(mapping)}"
        return mapping[key]

    def canonical_identifier(value):
        identifier = exp.to_identifier(value)
        identifier.meta["fingerprint_canonical"] = True
        return identifier

    for scope in scopes:
        index = scope_numbers[id(scope)]
        scope_sources[id(scope)] = {
            alias.lower(): pair for alias, pair in scope.selected_sources.items()
        }
        outputs[id(scope)] = {
            projection.alias_or_name.lower(): f"out{index}_{ordinal}"
            for ordinal, projection in enumerate(scope.expression.selects)
            if projection.alias_or_name and not projection.is_star
        }

    def source_field(source, original):
        if isinstance(source, exp.Table):
            source_name = name(sources, source.name, "source")
            mapping = source_fields.setdefault(source_name, {})
            return name(mapping, original, f"{source_name}_field")
        selected = outputs.get(id(source), {})
        if original.lower() in selected:
            return selected[original.lower()]
        # SELECT * forwards fields from its source; follow that lineage when
        # unambiguous rather than tying identifiers from unrelated sources.
        inputs = list(scope_sources.get(id(source), {}).values())
        if len(inputs) == 1:
            return source_field(inputs[0][1], original)
        mapping = source_fields.setdefault(f"scope{scope_numbers[id(source)]}", {})
        return name(mapping, original, f"scope{scope_numbers[id(source)]}_field")

    # Normalize declarations and references in their own scope. The same alias
    # may be reused by several nested queries without denoting the same input.
    for scope in scopes:
        index = scope_numbers[id(scope)]
        local = {}
        for ordinal, (alias, (node, source)) in enumerate(
            scope_sources[id(scope)].items()
        ):
            canonical = f"r{index}_{ordinal}"
            local[alias.lower()] = canonical
        bindings[id(scope)] = local

    for scope in scopes:
        for column in scope.columns:
            qualifier = column.table.lower()
            owner = scope
            while owner is not None and qualifier not in bindings.get(id(owner), {}):
                owner = owner.parent
            if owner is not None:
                source = scope_sources[id(owner)][qualifier][1]
                column.set(
                    "this", canonical_identifier(source_field(source, column.name))
                )
                column.set(
                    "table", canonical_identifier(bindings[id(owner)][qualifier])
                )
            elif not qualifier and len(bindings[id(scope)]) == 1:
                _, source = next(iter(scope_sources[id(scope)].values()))
                column.set(
                    "this", canonical_identifier(source_field(source, column.name))
                )
                column.set(
                    "table",
                    canonical_identifier(next(iter(bindings[id(scope)].values()))),
                )
            elif qualifier:
                # DuckDB struct access p.field is parsed as a qualified Column,
                # although p may be a projection output rather than a relation.
                inputs = list(scope_sources[id(scope)].values())
                scalar = (
                    source_field(inputs[0][1], column.table)
                    if len(inputs) == 1
                    else name(fields, column.table, "field")
                )
                column.set("table", canonical_identifier(scalar))
                column.set(
                    "this", canonical_identifier(name(members, column.name, "member"))
                )
            else:
                column.set(
                    "this", canonical_identifier(name(fields, column.name, "field"))
                )
            column.set("db", None)
            column.set("catalog", None)

    for scope in scopes:
        index = scope_numbers[id(scope)]
        # Explicit canonical output names preserve references through nested
        # projections even when a schema happens to reuse an input-field name.
        projections = []
        for ordinal, projection in enumerate(scope.expression.selects):
            if projection.is_star:
                projections.append(projection)
            else:
                value = (
                    projection.this if isinstance(projection, exp.Alias) else projection
                )
                projections.append(
                    exp.Alias(
                        this=value, alias=canonical_identifier(f"out{index}_{ordinal}")
                    )
                )
        if isinstance(scope.expression, exp.Select):
            scope.expression.set("expressions", projections)
        for alias, (node, source) in scope_sources[id(scope)].items():
            declaration = node.parent if isinstance(node.parent, exp.Subquery) else node
            declaration.set(
                "alias",
                exp.TableAlias(this=canonical_identifier(bindings[id(scope)][alias])),
            )
            if isinstance(node, exp.Table):
                original = node.name
                source_name = (
                    ctes[original]
                    if original in ctes
                    else name(sources, original, "source")
                )
                node.set("this", canonical_identifier(source_name))
                node.set("db", None)
                node.set("catalog", None)

    for node in tree.walk():
        node.comments = None
        if isinstance(node, exp.Literal):
            node.set("this", "constant" if node.is_string else "0")
        elif isinstance(node, exp.Boolean):
            node.set("this", True)
        elif isinstance(node, exp.Anonymous):
            node.set("this", node.name.lower())
        elif isinstance(node, exp.Identifier):
            parent = node.parent
            if node.meta.get("fingerprint_canonical"):
                pass
            elif isinstance(parent, exp.Table) or isinstance(parent, exp.TableAlias):
                if isinstance(parent, exp.TableAlias) and isinstance(
                    parent.parent, exp.CTE
                ):
                    node.set("this", ctes.get(node.name, node.name))
            elif isinstance(parent, exp.Column) and node.arg_key == "table":
                pass
            elif isinstance(parent, (exp.ColumnDef, exp.Dot)):
                node.set("this", name(members, node.name, "member"))
            else:
                node.set("this", name(fields, node.name, "field"))
            node.set("quoted", False)
    canonical = tree.sql(dialect="duckdb", pretty=False)
    return hashlib.sha256(
        f"ast-v{FINGERPRINT_VERSION}:{canonical}".encode()
    ).hexdigest()
