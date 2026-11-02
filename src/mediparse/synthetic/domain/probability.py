"""Pravděpodobnost a kontrola, že pravděpodobnosti tvoří rozdělení."""

from __future__ import annotations

from math import isclose
from typing import TYPE_CHECKING, Annotated, Final

from pydantic import Field

if TYPE_CHECKING:
    from collections.abc import Iterable

Probability = Annotated[float, Field(ge=0.0, le=1.0)]

_SUM_TOLERANCE: Final = 1e-9


def sums_to_one(values: Iterable[float]) -> bool:
    """Ověří, že hodnoty dávají dohromady 1 s tolerancí plovoucí čárky.

    Returns:
        Zda se součet liší od 1 nejvýš o toleranci; prázdný vstup má součet 0.
    """
    return isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=_SUM_TOLERANCE)
