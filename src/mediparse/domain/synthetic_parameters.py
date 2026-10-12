"""Parametry pětice z exportu sešitu 8: prevalence, společný výskyt a míry zmínek pro plán syntetiky."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Annotated, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mediparse.domain.labels import Probability
from mediparse.domain.mentions import DiagnosisMentions

if TYPE_CHECKING:
    from collections.abc import Mapping

_SAMPLE_FRACTION: Final = 0.2
_MIN_POSITIVES: Final = 5
_NOTES_STEP: Final = 100
_CEIL_PRECISION: Final = 9


class SyntheticParameters(BaseModel):
    """Export sešitu 8: vybraná pětice a její čísla, všechna zveřejnitelná (bez potlačených buněk)."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    selection: Annotated[tuple[str, ...], Field(min_length=1)]
    prevalence: dict[str, Annotated[float, Field(gt=0.0, le=1.0)]]
    conditional: dict[str, dict[str, Probability]]
    mentions: dict[str, DiagnosisMentions]

    @model_validator(mode="after")
    def _selection_has_no_duplicates(self) -> Self:
        if len(set(self.selection)) != len(self.selection):
            msg = "Výběr (selection) nesmí obsahovat diagnózu dvakrát."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _keys_match_selection(self) -> Self:
        expected = set(self.selection)
        maps = (self.prevalence, self.conditional, self.mentions)
        rows = (set(row) for row in self.conditional.values())
        if any(set(block) != expected for block in maps) or any(
            row != expected for row in rows
        ):
            msg = "Klíče prevalence, matice (i jejích řádků) a zmínek musí odpovídat výběru (selection)."
            raise ValueError(msg)
        return self


def required_notes(prevalence: Mapping[str, float]) -> int:
    """Počet zpráv korpusu: nejmenší stovka, při níž má nejvzácnější label v pětině korpusu očekávaně aspoň 5 pozitivních zpráv.

    Returns:
        Počet zpráv, násobek stovky.
    """
    rarest = min(prevalence.values())
    hundreds = math.ceil(
        round(
            _MIN_POSITIVES / (_SAMPLE_FRACTION * rarest * _NOTES_STEP), _CEIL_PRECISION
        )
    )
    return _NOTES_STEP * hundreds
