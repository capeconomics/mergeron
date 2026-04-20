"""Constants, types, objects and functions used within this sub-package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import frozen

from .. import VERSION
from .. import ArrayDouble

if TYPE_CHECKING:
    from collections.abc import Callable

__version__ = VERSION


@frozen
class GuidelinesBoundaryCallable:
    """A function to generate Guidelines boundary points, along with area and knot."""

    boundary_function: Callable[[ArrayDouble], ArrayDouble]
    area: float
    s_naught: float = 0
