"""Struktura plánu syntetické zprávy: pohlaví pacienta, věková značka, sekce a podnadpisy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mediparse.domain.labels import Probability

if TYPE_CHECKING:
    from collections.abc import Sequence
    from random import Random

    from mediparse.domain.synthetic_patients import SyntheticNote


class Sex(StrEnum):
    """Pohlaví pacienta, které se ve zprávě projeví polem Sex a zájmeny."""

    FEMALE = "female"
    MALE = "male"


class Subheading(BaseModel):
    """Podnadpis uvnitř sekce a pravděpodobnost jeho výskytu."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    header: str
    probability: Probability


class SectionModel(BaseModel):
    """Sekce zprávy: kanonický klíč, hlavička, pravděpodobnost výskytu a skupiny podnadpisů.

    Skupina s více podnadpisy jsou vzájemně se vylučující varianty, losuje se nejvýš jedna.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    key: str
    header: str
    probability: Probability
    subheadings: tuple[Annotated[tuple[Subheading, ...], Field(min_length=1)], ...] = ()

    @model_validator(mode="after")
    def _groups_are_probabilities(self) -> Self:
        if any(sum(s.probability for s in group) > 1.0 for group in self.subheadings):
            msg = f"Varianty podnadpisu v sekci {self.key} mají součet pravděpodobností nad 1."
            raise ValueError(msg)
        return self


class StructureModel(BaseModel):
    """Sekce v pořadí, v němž se ve zprávě vyskytují, a atributy pacienta a zprávy."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    sections: Annotated[tuple[SectionModel, ...], Field(min_length=1)]
    female_probability: Probability
    age_marker_probability: Probability

    @model_validator(mode="after")
    def _keys_are_unique(self) -> Self:
        keys = [section.key for section in self.sections]
        if len(set(keys)) != len(keys):
            msg = "Klíče sekcí se opakují."
            raise ValueError(msg)
        return self


@dataclass(frozen=True)
class NoteStructure:
    """Plánovaná struktura jedné zprávy."""

    note_id: str
    sex: Sex
    age_marker: bool
    sections: tuple[str, ...]
    subheadings: tuple[str, ...]


def sample_structure(
    notes: Sequence[SyntheticNote], model: StructureModel, rng: Random
) -> tuple[NoteStructure, ...]:
    """Vylosuje strukturu každé zprávy; pohlaví se losuje jednou za pacienta.

    Returns:
        Struktury ve stejném pořadí jako zprávy.
    """
    sexes: dict[int, Sex] = {}
    structures: list[NoteStructure] = []
    for note in notes:
        if note.subject_id not in sexes:
            sexes[note.subject_id] = _sex(model, rng)
        structures.append(_structure(note.note_id, sexes[note.subject_id], model, rng))
    return tuple(structures)


def _sex(model: StructureModel, rng: Random) -> Sex:
    return Sex.FEMALE if rng.random() < model.female_probability else Sex.MALE


def _structure(
    note_id: str, sex: Sex, model: StructureModel, rng: Random
) -> NoteStructure:
    age_marker = rng.random() < model.age_marker_probability
    present = tuple(s for s in model.sections if rng.random() < s.probability)
    subheadings = tuple(
        header
        for section in present
        for group in section.subheadings
        if (header := _pick(group, rng)) is not None
    )
    return NoteStructure(
        note_id=note_id,
        sex=sex,
        age_marker=age_marker,
        sections=tuple(section.key for section in present),
        subheadings=subheadings,
    )


def _pick(group: Sequence[Subheading], rng: Random) -> str | None:
    draw = rng.random()
    for subheading in group:
        if draw < subheading.probability:
            return subheading.header
        draw -= subheading.probability
    return None
