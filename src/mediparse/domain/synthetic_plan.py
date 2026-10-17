"""Konfigurace vzorkovače plánů syntetických zpráv: parametry, ze kterých se plány losují, včetně modelu pacientů."""

from __future__ import annotations

from typing import Annotated, Final, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)

from mediparse.domain.identity import SubjectId
from mediparse.domain.labels import LabelModel
from mediparse.domain.mentions import MentionModel
from mediparse.domain.note_structure import StructureModel
from mediparse.domain.probability import Probability

# Model pacientů je část schématu configu vzorkovače, proto neleží u algoritmu: v synthetic_patients
# by na modulu losování stály config i typy plánu (SDP, hlídá kontrakt import-linteru).
MIN_MULTIPLE_NOTES: Final = 2


class PatientModel(BaseModel):
    """Velikost korpusu a rozdělení počtu zpráv na pacienta."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    notes: PositiveInt
    first_subject_id: SubjectId
    single_note_probability: Probability
    max_notes: Annotated[int, Field(ge=MIN_MULTIPLE_NOTES)]
    mean_notes_when_multiple: Annotated[float, Field(gt=MIN_MULTIPLE_NOTES)]
    max_attempts: PositiveInt

    @model_validator(mode="after")
    def _mean_is_reachable(self) -> Self:
        if self.mean_notes_when_multiple >= (MIN_MULTIPLE_NOTES + self.max_notes) / 2:
            msg = "Průměr musí ležet pod středem rozsahu, jinak ho klesající rozdělení nedosáhne."
            raise ValueError(msg)
        return self


class SamplerConfig(BaseModel):
    """Všechny parametry vzorkovače; každý má zdroj uvedený v podkladu notes-synthesis."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    seed: NonNegativeInt
    labels: LabelModel
    patients: PatientModel
    structure: StructureModel
    mentions: MentionModel
