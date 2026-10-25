"""Experimentální matice: deklarace experimentů, které se mají spustit, a jejich invarianty."""

from __future__ import annotations

from collections import Counter
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from mediparse.domain.clinical_text import NORMALIZATION_LANGUAGE
from mediparse.domain.identity import RowId
from mediparse.domain.language import Language
from mediparse.domain.section_headers import LEXICON_LANGUAGE

if TYPE_CHECKING:
    from collections.abc import Iterable


class Tier(StrEnum):
    """Stupeň modelu od klasického strojového učení po transformer."""

    CLASSICAL = "classical"
    INTERMEDIATE = "intermediate"
    TRANSFORMER = "transformer"


class ModelName(StrEnum):
    """Jméno modelu v matici experimentů; stupeň se z něj odvozuje, v manifestu se neuvádí."""

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
    ModelName.LOGREG: Tier.CLASSICAL,
    ModelName.SVM: Tier.CLASSICAL,
    ModelName.FASTTEXT: Tier.INTERMEDIATE,
    ModelName.XLM_R_BASE: Tier.TRANSFORMER,
    ModelName.MDEBERTA_V3_BASE: Tier.TRANSFORMER,
})


class MatrixRow(BaseModel):
    """Jeden experiment: kombinace os matice a rozpočet ladění."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    id: RowId
    language: Language
    model: ModelName
    strip_diagnostic_sections: bool
    normalize_clinical_text: bool
    tuning_trials: PositiveInt

    @model_validator(mode="after")
    def _strip_is_english_only(self) -> Self:
        if self.strip_diagnostic_sections and self.language is not LEXICON_LANGUAGE:
            msg = (
                f"Řádek {self.id}: strip diagnostických sekcí zná jen jazyk "
                f"lexikonu sekcí ({LEXICON_LANGUAGE})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _normalization_is_english_only(self) -> Self:
        if self.normalize_clinical_text and self.language is not NORMALIZATION_LANGUAGE:
            msg = (
                f"Řádek {self.id}: normalizace klinického textu zná jen jazyk "
                f"svých pravidel ({NORMALIZATION_LANGUAGE})"
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _normalization_is_classical_only(self) -> Self:
        if self.normalize_clinical_text and self.model.tier is not Tier.CLASSICAL:
            msg = (
                f"Řádek {self.id}: normalizace klinického textu je příznak "
                "klasického modelu (TF-IDF), ne modelu stupně "
                f"{self.model.tier}"
            )
            raise ValueError(msg)
        return self

    @property
    def axes(self) -> tuple[Language, ModelName, bool, bool]:
        """Osy, které experiment jednoznačně určují."""
        return (
            self.language,
            self.model,
            self.strip_diagnostic_sections,
            self.normalize_clinical_text,
        )


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
