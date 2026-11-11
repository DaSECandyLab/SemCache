"""Immutable partial plan trees for research design §4.1."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PartialPlanNode:
    node_id: int
    children: tuple = ()
    height: int = 0
    operator_id: object = None
    fields: tuple = ()
    relations: frozenset = frozenset()
    pending_join: object = None
    pair_columns: tuple = ()


@dataclass(frozen=True)
class PartialPlanSet:
    nodes: tuple
    roots: tuple
    remaining: tuple
    inputs: tuple


def descendants(inputs):
    children = dict(inputs)
    result, active = {}, set()

    def visit(key):
        if key in active:
            raise ValueError("Logical plan contains a cycle")
        if key not in children:
            raise ValueError("Unknown logical child")
        if key not in result:
            active.add(key)
            sources = set(children[key])
            for child in children[key]:
                sources.update(visit(child))
            active.remove(key)
            result[key] = frozenset(sources)
        return result[key]

    for key in children:
        visit(key)
    return result
