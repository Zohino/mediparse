"""Experimentální matice: deklarace experimentů, které se mají spustit, a jejich invarianty."""

from __future__ import annotations

from collections import Counter
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

if TYPE_CHECKING:
    from collections.abc import Iterable

RowId = Annotated[str, Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")]


class Language(StrEnum):
    """Jazyk vstupních zpráv; čeština vzniká výhradně strojovým překladem."""

    EN = "en"
    CS = "cs"


class Tier(StrEnum):
    """Stupeň modelu od klasického strojového učení po transformer."""

    CLASSICAL = "classical"
    INTERMEDIATE = "intermediate"
    TRANSFORMER = "transformer"


class Model(StrEnum):
    """Konkrétní model; stupeň se z něj odvozuje, v manifestu se neuvádí."""

    LOGREG = "logreg"
    SVM = "svm"
    FASTTEXT = "fasttext"
    XLM_R_BASE = "xlm-r-base"
    MDEBERTA_V3_BASE = "mdeberta-v3-base"

    @property
    def tier(self) -> Tier:
        """Stupeň, do kterého model patří."""
        return _MODEL_TIERS[self]


_MODEL_TIERS = MappingProxyType({
    Model.LOGREG: Tier.CLASSICAL,
    Model.SVM: Tier.CLASSICAL,
    Model.FASTTEXT: Tier.INTERMEDIATE,
    Model.XLM_R_BASE: Tier.TRANSFORMER,
    Model.MDEBERTA_V3_BASE: Tier.TRANSFORMER,
})


class MatrixRow(BaseModel):
    """Jeden experiment: kombinace os matice a rozpočet ladění."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    id: RowId
    language: Language
    model: Model
    strip_diagnostic_sections: bool
    tuning_trials: PositiveInt

    @property
    def axes(self) -> tuple[Language, Model, bool]:
        """Osy, které experiment jednoznačně určují."""
        return self.language, self.model, self.strip_diagnostic_sections


class ExperimentMatrix(BaseModel):
    """Všechny deklarované experimenty a invarianty, které musí splňovat společně."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    rows: Annotated[tuple[MatrixRow, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _ids_are_unique(self) -> Self:
        duplicates = _duplicates(row.id for row in self.rows)
        if duplicates:
            msg = f"Duplicitní id řádků: {sorted(duplicates)}"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _axes_are_unique(self) -> Self:
        duplicates = _duplicates(row.axes for row in self.rows)
        if duplicates:
            msg = f"Více řádků má stejné osy: {sorted(duplicates)}"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _tuning_budget_is_shared(self) -> Self:
        budgets = {row.tuning_trials for row in self.rows}
        if len(budgets) > 1:
            msg = f"Řádky mají rozdílný rozpočet ladění: {sorted(budgets)}"
            raise ValueError(msg)
        return self


def _duplicates[T](items: Iterable[T]) -> set[T]:
    return {item for item, count in Counter(items).items() if count > 1}
