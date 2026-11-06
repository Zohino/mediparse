"""Načtení variant názvů pětice pro měření EDA."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from mediparse.domain.diagnosis import Diagnosis, PerDiagnosis
from mediparse.infrastructure.input_file import parse_file

if TYPE_CHECKING:
    from pathlib import Path


class _Variants(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    variants_en: PerDiagnosis[
        Annotated[
            list[
                Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
            ],
            Field(min_length=1),
        ]
    ]

    @model_validator(mode="after")
    def _unique_variants(self) -> _Variants:
        seen: set[str] = set()
        for variants in self.variants_en.values():
            for variant in variants:
                key = variant.lower()
                if key in seen:
                    msg = f"Duplicitní varianta: {variant}."
                    raise ValueError(msg)
                seen.add(key)
        return self


def load_diagnosis_variants(path: Path) -> dict[str, Diagnosis]:
    """Přečte striktní konfiguraci s variantami všech diagnóz pětice.

    Returns:
        Seřazený slovník varianta → diagnóza.
    """
    return parse_file(path, _parse)


def _parse(text: str) -> dict[str, Diagnosis]:
    parsed = _Variants.model_validate_json(text)
    return dict(
        sorted(
            (variant, diagnosis)
            for diagnosis, variants in parsed.variants_en.items()
            for variant in variants
        )
    )
