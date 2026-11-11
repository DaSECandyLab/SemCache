"""Immutable row samples and exact, unexpanded Cartesian populations."""

from collections.abc import Sequence
from dataclasses import dataclass

from planning.observations.evidence import freeze_evidence, is_frozen_evidence

from ..state import digest


@dataclass(frozen=True)
class SampleRow:
    identity: str
    values: dict
    lineage: tuple = ()


def columns(rows):
    if isinstance(rows, CartesianRows):
        return rows.columns
    return tuple(dict.fromkeys(key for row in rows for key in row.values))


def lineages(rows):
    if isinstance(rows, CartesianRows):
        return rows.lineages()
    return {pair for row in rows for pair in row.lineage}


class CartesianRows(Sequence):
    """Exact bag product; independent lineage domains keep weights factorizable.

    No pair rows or IDs are allocated until a consumer requests that pair.
    Inputs are published immutable samples, with disjoint output column names.
    """

    def __init__(self, left, right):
        def published(rows):
            if (
                isinstance(rows, CartesianRows)
                or type(rows) is tuple
                and all(is_frozen_evidence(row.values) for row in rows)
            ):
                return rows
            return tuple(
                SampleRow(row.identity, freeze_evidence(row.values), row.lineage)
                for row in rows
            )

        self.left, self.right = published(left), published(right)
        left, right = self.left, self.right
        left_columns, right_columns = columns(left), columns(right)
        if set(left_columns) & set(right_columns):
            raise ValueError("Factorized products require disjoint field names")
        if {rel for rel, _ in lineages(left)} & {rel for rel, _ in lineages(right)}:
            raise ValueError("Factorized products require independent lineage domains")
        self.columns = left_columns + right_columns
        # Membership validation recovers one row per leaf from its lineage.
        # Grouped outputs can share empty origins: preserve them via the exact
        # materialized backend instead of constructing an ambiguous product.
        for side in (left, right):
            if not isinstance(side, CartesianRows) and len(
                {tuple(sorted(row.lineage, key=repr)) for row in side}
            ) != len(side):
                raise ValueError("Factorized leaves require unique lineage identities")
        self._indexes = None

    def __len__(self):
        return len(self.left) * len(self.right)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        li, ri = divmod(index, len(self.right))
        return self.combine(self.left[li], self.right[ri])

    @staticmethod
    def combine(left, right):
        return SampleRow(
            digest([left.identity, right.identity]),
            freeze_evidence({**left.values, **right.values}),
            tuple(sorted(set(left.lineage) | set(right.lineage), key=repr)),
        )

    def leaves(self):
        def visit(rows):
            if isinstance(rows, CartesianRows):
                yield from rows.leaves()
            else:
                yield rows

        yield from visit(self.left)
        yield from visit(self.right)

    def lineages(self):
        return lineages(self.left) | lineages(self.right) if self else set()

    def mass(self, weight):
        def mass(rows):
            return (
                rows.mass(weight)
                if isinstance(rows, CartesianRows)
                else sum(weight(row) for row in rows)
            )

        return mass(self.left) * mass(self.right)

    def effective(self, weight):
        def effective(rows):
            if isinstance(rows, CartesianRows):
                return rows.effective(weight)
            weights = [weight(row) for row in rows]
            squares = sum(value * value for value in weights)
            return sum(weights) ** 2 / squares if squares else 0.0

        return effective(self.left) * effective(self.right)

    def contains(self, row):
        return row.identity == self.identity_for(row.lineage)

    def identity_for(self, lineage):
        if not self:
            return None
        if self._indexes is None:
            self._indexes = [
                (
                    leaf,
                    {rel for r in leaf for rel, _ in r.lineage},
                    {tuple(sorted(r.lineage, key=repr)): r for r in leaf},
                )
                for leaf in self.leaves()
            ]
        selected = {}
        origins = set(lineage)
        for leaf, relation_ids, index in self._indexes:
            origin = tuple(
                sorted((pair for pair in origins if pair[0] in relation_ids), key=repr)
            )
            source = index.get(origin)
            if source is None:
                return None
            selected[id(leaf)] = source.identity
        return self.identity(selected)

    def identity(self, selected):
        def value(rows):
            return (
                rows.identity(selected)
                if isinstance(rows, CartesianRows)
                else selected[id(rows)]
            )

        return digest([value(self.left), value(self.right)])

    def leaf_for(self, keys):
        required = set(keys)
        return next(
            (leaf for leaf in self.leaves() if required <= set(columns(leaf))), None
        )

    def representative(self, leaf, row):
        def choose(rows):
            if rows is leaf:
                return row
            if isinstance(rows, CartesianRows):
                return rows.representative(leaf, row)
            return rows[0]

        return self.combine(choose(self.left), choose(self.right))

    def replace_leaf(self, leaf, replacement):
        def update(rows):
            if rows is leaf:
                return replacement
            return (
                rows.replace_leaf(leaf, replacement)
                if isinstance(rows, CartesianRows)
                else rows
            )

        return CartesianRows(update(self.left), update(self.right))


class ProductFieldValues(Sequence):
    """The full ordered field multiset, without allocating repeated row values."""

    def __init__(self, rows, key, serialize):
        self.size = len(rows)
        self.stride = 1
        if isinstance(rows, CartesianRows):
            if key in columns(rows.left):
                self.source = ProductFieldValues(rows.left, key, serialize)
                self.stride = len(rows.right)
            else:
                self.source = ProductFieldValues(rows.right, key, serialize)
        else:
            self.source = tuple(serialize(row.values[key]) for row in rows)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.source[(index // self.stride) % len(self.source)]
