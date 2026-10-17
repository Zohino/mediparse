"""Plán syntetické zprávy: vše, co verbalizace převádí do textu a co kontroly souladu ověřují."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict, NonNegativeInt, model_validator

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.identity import SubjectId
from mediparse.domain.mentions import Mention
from mediparse.domain.note import subject_of
from mediparse.domain.note_structure import Sex

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.domain.note_structure import NoteStructure
    from mediparse.domain.synthetic_note import SyntheticNote


class NotePlan(BaseModel):
    """Plán jedné zprávy; kolekce jsou uspořádané, aby zápis nezávisel na procesu."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    note_id: str
    subject_id: SubjectId
    labels: tuple[Diagnosis, ...]
    sex: Sex
    age_marker: bool
    sections: tuple[str, ...]
    subheadings: tuple[str, ...]
    section_words: dict[str, NonNegativeInt]
    narrative_deid: NonNegativeInt
    mentions: tuple[Mention, ...]

    @model_validator(mode="after")
    def _subject_matches_note_id(self) -> Self:
        if subject_of(self.note_id) != self.subject_id:
            msg = f"Zpráva {self.note_id} nepatří pacientovi subject_id {self.subject_id}."
            raise ValueError(msg)
        return self


class MismatchedPlanPartsError(ValueError):
    """Části plánu nepatří ke stejné zprávě."""


def note_plan(
    note: SyntheticNote, structure: NoteStructure, mentions: Sequence[Mention]
) -> NotePlan:
    """Složí plán zprávy z labelů, struktury a zmínek.

    Returns:
        Plán s labely v pořadí diagnóz.

    Raises:
        MismatchedPlanPartsError: Struktura patří jiné zprávě.
    """
    if structure.note_id != note.note_id:
        msg = f"Struktura {structure.note_id} nepatří ke zprávě {note.note_id}."
        raise MismatchedPlanPartsError(msg)
    return NotePlan(
        note_id=note.note_id,
        subject_id=note.subject_id,
        labels=tuple(d for d in Diagnosis if d in note.labels),
        sex=structure.sex,
        age_marker=structure.age_marker,
        sections=structure.sections,
        subheadings=structure.subheadings,
        # Struktura nese Mapping, striktní pydantic přijme jen dict.
        section_words=dict(structure.section_words),
        narrative_deid=structure.narrative_deid,
        mentions=tuple(mentions),
    )
