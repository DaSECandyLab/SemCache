"""Sampling contracts; production sampling is resident in the native engine."""

from typing import TYPE_CHECKING

from .quality import update_quality as update_quality
from .rows import SampleRow as SampleRow

if TYPE_CHECKING:
    from .coordinator import SamplingCoordinator as SamplingCoordinator

__all__ = ["SampleRow", "SamplingCoordinator", "update_quality"]


def __getattr__(name):
    if name == "SamplingCoordinator":
        from .coordinator import SamplingCoordinator

        globals()[name] = SamplingCoordinator
        return SamplingCoordinator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
