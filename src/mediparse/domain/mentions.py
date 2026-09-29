"""Plán zmínek diagnóz: zda text diagnózu zmíní klíčovým slovem, v kterých sekcích a s jakým statusem."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Final, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from mediparse.domain.labels import Diagnosis, Probability

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from random import Random

    from mediparse.domain.note_structure import NoteStructure
    from mediparse.domain.synthetic_patients import SyntheticNote

_WEIGHT_TOLERANCE: Final = 1e-9


class MentionStatus(StrEnum):
    """Jak text diagnózu zmiňuje; u pozitivního labelu vždy potvrzeně."""

    AFFIRMED = "affirmed"
    NEGATED = "negated"
    FAMILY_HISTORY = "family_history"
    UNCERTAIN = "uncertain"
    AFFIRMED_UNCODED = "affirmed_uncoded"


class DiagnosisMentions(BaseModel):
    """Pravděpodobnost zmínky podle labelu a klíčová slova, která zmínku tvoří."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    given_positive: Probability
    given_negative: Probability
    keywords: Annotated[tuple[str, ...], Field(min_length=1)]


class Negation(BaseModel):
    """Pravidlo negace z EDA: negační výraz nejvýš ``window_chars`` znaků před klíčovým slovem."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    cues: Annotated[tuple[str, ...], Field(min_length=1)]
    window_chars: PositiveInt


class MentionModel(BaseModel):
    """Parametry zmínek: pravděpodobnosti podle labelu, umístění a statusy u negativního labelu."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    diagnoses: dict[Diagnosis, DiagnosisMentions]
    diagnosis_section: str
    family_section: str
    diagnosis_section_probability: Probability
    narrative_probability: Probability
    chronic_sections: dict[str, PositiveFloat]
    acute_sections: dict[str, PositiveFloat]
    negative_statuses: dict[MentionStatus, Probability]
    negation: Negation

    @model_validator(mode="after")
    def _covers_all_diagnoses(self) -> Self:
        if set(self.diagnoses) != set(Diagnosis):
            msg = "Parametry zmínek musí pokrývat všech pět diagnóz."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _negative_statuses_are_a_distribution(self) -> Self:
        if MentionStatus.AFFIRMED in self.negative_statuses:
            msg = "Zmínka u negativního labelu nemůže být potvrzená s kódem."
            raise ValueError(msg)
        if abs(sum(self.negative_statuses.values()) - 1.0) > _WEIGHT_TOLERANCE:
            msg = "Pravděpodobnosti statusů u negativního labelu musí dávat 1."
            raise ValueError(msg)
        return self


@dataclass(frozen=True)
class Mention:
    """Plánovaná zmínka diagnózy: status a sekce v pořadí zprávy."""

    diagnosis: Diagnosis
    status: MentionStatus
    sections: tuple[str, ...]


def sample_mentions(
    notes: Sequence[SyntheticNote],
    structures: Sequence[NoteStructure],
    model: MentionModel,
    rng: Random,
) -> tuple[tuple[Mention, ...], ...]:
    """Vylosuje zmínky každé zprávy; zmínka stojí jen v sekci, kterou struktura zprávy má.

    Returns:
        Zmínky každé zprávy v pořadí diagnóz, ve stejném pořadí jako zprávy.
    """
    return tuple(
        tuple(
            mention
            for diagnosis in Diagnosis
            if (mention := _mention(diagnosis, note, structure, model, rng)) is not None
        )
        for note, structure in zip(notes, structures, strict=True)
    )


def _mention(
    diagnosis: Diagnosis,
    note: SyntheticNote,
    structure: NoteStructure,
    model: MentionModel,
    rng: Random,
) -> Mention | None:
    positive = diagnosis in note.labels
    parameters = model.diagnoses[diagnosis]
    probability = parameters.given_positive if positive else parameters.given_negative
    if rng.random() >= probability:
        return None
    if positive:
        status = MentionStatus.AFFIRMED
        chosen = _positive_sections(diagnosis, structure, model, rng)
    else:
        status = _negative_status(model, rng)
        chosen = _negative_sections(diagnosis, status, structure, model, rng)
    sections = tuple(s for s in structure.sections if s in chosen)
    return Mention(diagnosis, status, sections) if sections else None


def _positive_sections(
    diagnosis: Diagnosis, structure: NoteStructure, model: MentionModel, rng: Random
) -> list[str]:
    in_diagnosis = (
        model.diagnosis_section in structure.sections
        and rng.random() < model.diagnosis_section_probability
    )
    narrative = rng.random() < model.narrative_probability or not in_diagnosis
    chosen = [model.diagnosis_section] if in_diagnosis else []
    if narrative:
        chosen += _weighted(_narrative_weights(diagnosis, model), structure, rng)
    return chosen


def _negative_sections(
    diagnosis: Diagnosis,
    status: MentionStatus,
    structure: NoteStructure,
    model: MentionModel,
    rng: Random,
) -> list[str]:
    if (
        status is MentionStatus.FAMILY_HISTORY
        and model.family_section in structure.sections
    ):
        return [model.family_section]
    return _weighted(_narrative_weights(diagnosis, model), structure, rng)


def _negative_status(model: MentionModel, rng: Random) -> MentionStatus:
    statuses = list(model.negative_statuses)
    return rng.choices(
        statuses, weights=[model.negative_statuses[s] for s in statuses]
    )[0]


def _narrative_weights(
    diagnosis: Diagnosis, model: MentionModel
) -> Mapping[str, float]:
    return model.chronic_sections if diagnosis.chronic else model.acute_sections


def _weighted(
    weights: Mapping[str, float], structure: NoteStructure, rng: Random
) -> list[str]:
    present = [section for section in weights if section in structure.sections]
    if not present:
        return []
    return rng.choices(present, weights=[weights[section] for section in present])
