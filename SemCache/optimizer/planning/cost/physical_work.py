"""Conservative physical work overlays; logical cardinalities remain unchanged.

Only trusted bound SQL and the current state's single-consumer topology certify
an equijoin. Missing/unsupported evidence retains the ordinary logical costs.
"""

import json
import re
from collections import Counter, OrderedDict
from functools import lru_cache

import sqlglot
from sqlglot import exp


def _name(field):
    if isinstance(field, dict):
        field = field.get("binding", field.get("name"))
    if isinstance(field, str):
        return field
    return json.dumps(field, separators=(",", ":"))


def _kind(op):
    return op.get("kind", op.get("type", "")).upper()


def _sql(op):
    return op.get("sql", op.get("sampling_sql", ""))


def _trusted_filter(op):
    return (
        _kind(op) == "FILTER"
        and not op.get("semantic", False)
        and op.get("sampling_row_local") is True
        and isinstance(op.get("sampling_required_fields"), (list, tuple))
        and op.get("sampling_row_local_sql") == _sql(op)
    )


@lru_cache(maxsize=2048)
def _equalities(sql):
    """Accept only the exporter's SELECT * and AND of quoted column = column."""
    prefix = "SELECT * FROM $INPUT WHERE "
    if not sql.startswith(prefix):
        return ()
    if not re.fullmatch(r'(?:"(?:[^"]|"")*"|[()=]|\s+|AND)+', sql[len(prefix) :]):
        return ()
    try:
        statements = sqlglot.parse(
            sql.replace("$INPUT", "__semcache_input"), read="duckdb"
        )
        if len(statements) != 1 or not isinstance(statements[0], exp.Select):
            return ()
        tree = statements[0]
        if any(
            value
            for key, value in tree.args.items()
            if key not in {"expressions", "from_", "where"}
        ):
            return ()
        source = tree.args.get("from_")
        if (
            len(tree.expressions) != 1
            or not isinstance(tree.expressions[0], exp.Star)
            or source is None
            or not isinstance(source.this, exp.Table)
            or source.this.name != "__semcache_input"
        ):
            return ()

        def visit(expression):
            if isinstance(expression, exp.Paren):
                return visit(expression.this)
            if isinstance(expression, exp.And):
                return visit(expression.this) + visit(expression.expression)
            if not isinstance(expression, exp.EQ):
                raise ValueError("not a column equality")
            names = []
            for column in (expression.this, expression.expression):
                if (
                    not isinstance(column, exp.Column)
                    or column.table
                    or not isinstance(column.this, exp.Identifier)
                    or not column.this.args.get("quoted")
                ):
                    raise ValueError("not an unqualified bound column")
                names.append(column.name)
            return [tuple(names)]

        return tuple(visit(tree.args["where"].this))
    except (sqlglot.errors.SqlglotError, KeyError, ValueError, TypeError):
        return ()


@lru_cache(maxsize=2048)
def _identity_projection(sql):
    """The projection must retain column identities, with no computed values."""
    if not sql.startswith("SELECT ") or not sql.endswith(" FROM $INPUT"):
        return frozenset()
    body = sql[len("SELECT ") : -len(" FROM $INPUT")]
    column = r'"(?:[^"]|"")*"'
    item = rf"(?:__sample_id|{column}\s+AS\s+{column})"
    if not re.fullmatch(rf"\s*{item}(?:\s*,\s*{item})*\s*", body):
        return frozenset()
    try:
        tree = sqlglot.parse_one(
            sql.replace("$INPUT", "__semcache_input"), read="duckdb"
        )
        if not isinstance(tree, exp.Select) or any(
            value
            for key, value in tree.args.items()
            if key not in {"expressions", "from_"}
        ):
            return frozenset()
        columns = set()
        for expression in tree.expressions:
            if isinstance(expression, exp.Column) and expression.name == "__sample_id":
                continue
            if (
                not isinstance(expression, exp.Alias)
                or not isinstance(expression.this, exp.Column)
                or expression.this.table
                or not expression.this.this.args.get("quoted")
                or not expression.args["alias"].args.get("quoted")
                or expression.this.name != expression.alias
            ):
                return frozenset()
            columns.add(expression.alias)
        return frozenset(columns)
    except (sqlglot.errors.SqlglotError, KeyError, ValueError, TypeError):
        return frozenset()


def physical_work(nodes, operators, logical):
    """Return cost-only overrides for completed, certified equijoin chains.

    Each overlay has cost_kind, input_nodes, fusion_owner, input_units_basis.
    It never changes the state, output rows, selectivity, or semantic requests.
    """
    nodes = tuple(nodes)
    descriptions = {
        node.node_id: operators.get(node.operator_id, logical.get(node.node_id, {}))
        for node in nodes
    }
    certificates = {nid: _certificate(op) for nid, op in descriptions.items()}
    if not any(row[1] for row in certificates.values()) or not any(
        row[0] == "CROSS_PRODUCT" for row in certificates.values()
    ):
        return {}
    fields = {}
    for node in nodes:
        op = descriptions[node.node_id]
        if getattr(node, "fields", ()):
            fields[node.node_id] = {_name(field) for field in node.fields}
        elif node.operator_id is None:
            fields[node.node_id] = {
                _name(field) for field in op.get("output_schema", [])
            }
        else:
            original = logical.get(node.node_id, op)
            inherited = {
                _name(field)
                for child in original.get("children", [])
                for field in logical.get(child, {}).get("output_schema", [])
            }
            produced = {
                _name(field) for field in original.get("output_schema", [])
            } - inherited
            fields[node.node_id] = (
                set().union(*(fields[child] for child in node.children)) | produced
            )
    return _identify(nodes, certificates, fields)


def _certificate(op):
    kind = _kind(op)
    trusted_filter = _trusted_filter(op)
    return (
        kind,
        _equalities(_sql(op)) if trusted_filter else (),
        trusted_filter,
        _identity_projection(_sql(op))
        if kind == "PROJECTION" and not op.get("semantic", False)
        else frozenset(),
    )


def _identify(nodes, certificates, fields):
    by_id = {node.node_id: node for node in nodes}
    consumers = Counter(child for node in nodes for child in node.children)
    overrides = {}
    for owner in nodes:
        equalities = certificates[owner.node_id][1]
        required = {name for pair in equalities for name in pair}
        if len(owner.children) != 1 or not equalities:
            continue
        current = by_id[owner.children[0]]
        intermediates = []
        while consumers[current.node_id] == 1:
            kind, _, trusted_filter, projection = certificates[current.node_id]
            if kind == "CROSS_PRODUCT" and len(current.children) == 2:
                left, right = (fields[child] for child in current.children)
                if not left.isdisjoint(right) or not all(
                    (a in left and b in right) or (a in right and b in left)
                    for a, b in equalities
                ):
                    break
                if current.node_id in overrides:
                    break
                overrides[current.node_id] = {
                    "cost_kind": "CROSS_PRODUCT",
                    "input_nodes": (),
                    "fusion_owner": owner.node_id,
                    "input_units_basis": "fused into certified equijoin; no separate Cartesian work",
                }
                for intermediate in intermediates:
                    overrides[intermediate] = {
                        "cost_kind": certificates[intermediate][0],
                        "input_nodes": (),
                        "fusion_owner": owner.node_id,
                        "input_units_basis": "trusted residual predicate or identity projection fused into certified equijoin",
                    }
                overrides[owner.node_id] = {
                    "cost_kind": "COMPARISON_JOIN",
                    "input_nodes": tuple(current.children),
                    "fusion_owner": owner.node_id,
                    "input_units_basis": "sum of certified equijoin child rows",
                }
                break
            if len(current.children) != 1 or not (
                trusted_filter or kind == "PROJECTION" and required <= projection
            ):
                break
            intermediates.append(current.node_id)
            current = by_id[current.children[0]]
    return overrides


class PhysicalWorkContext:
    """Query-owned static certificates and bounded immutable node field names.

    The caller creates this after stage preparation and retains it only for that
    query. Cache values never retain state or node handles from a pruned beam.
    ``physical_work`` remains uncached for callers with mutable dictionaries.
    """

    def __init__(self, operators, logical, field_registry=(), *, cache_size=8192):
        descriptions = dict(logical)
        descriptions.update((op["node_id"], op) for op in operators.values())
        self.certificates = {nid: _certificate(op) for nid, op in descriptions.items()}
        self.schemas = {
            nid: frozenset(_name(field) for field in op.get("output_schema", []))
            for nid, op in logical.items()
        }
        self.produced = {
            nid: self.schemas[nid]
            - set().union(*(self.schemas[child] for child in op.get("children", [])))
            for nid, op in logical.items()
        }
        self.field_names = tuple(_name(field) for field in field_registry)
        self.cross_nodes = {
            nid
            for nid, certificate in self.certificates.items()
            if certificate[0] == "CROSS_PRODUCT"
        }
        self.has_equalities = any(row[1] for row in self.certificates.values())
        self.cache_size = max(0, cache_size)
        self.node_fields = OrderedDict()

    def work(self, nodes):
        nodes = tuple(nodes)
        if not self.has_equalities or not any(
            node.node_id in self.cross_nodes for node in nodes
        ):
            return {}
        fields = {}
        for node in nodes:
            handle = getattr(node, "_handle", node)
            key = getattr(handle, "handle_id", None)
            names = self.node_fields.get(key) if key is not None else None
            if names is not None:
                self.node_fields.move_to_end(key)
            else:
                field_ids = getattr(handle, "field_ids", None)
                if field_ids is not None and self.field_names:
                    names = frozenset(self.field_names[index] for index in field_ids)
                elif getattr(node, "fields", ()):
                    names = frozenset(_name(field) for field in node.fields)
                elif node.operator_id is None:
                    names = self.schemas[node.node_id]
                else:
                    names = frozenset().union(
                        *(fields[child] for child in node.children),
                        self.produced[node.node_id],
                    )
                if key is not None and self.cache_size:
                    self.node_fields[key] = names
                    if len(self.node_fields) > self.cache_size:
                        self.node_fields.popitem(last=False)
            fields[node.node_id] = names
        return _identify(nodes, self.certificates, fields)
