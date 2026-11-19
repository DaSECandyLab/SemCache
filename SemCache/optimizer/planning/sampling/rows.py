"""Immutable row samples."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SampleRow:
    identity: str
    values: dict
    lineage: tuple = ()
