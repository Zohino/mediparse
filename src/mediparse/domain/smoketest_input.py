"""Vstupní tabulka smoketestu: zprávy jednoho jazyka spárované s labely svých plánů."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mediparse.domain.labels import Diagnosis
    from mediparse.domain.note_plan import NotePlan


@dataclass(frozen=True)
class InputNote:
    """Řádek vstupní tabulky: text zprávy v jednom jazyce a labely z jejího plánu."""

    note_id: str
    subject_id: int
    language: str
    text: str
    labels: frozenset[Diagnosis]


class UnpairedNotesError(InvalidInputError):
    """Plány a zprávy jednoho jazyka si neodpovídají jedna k jedné."""


def input_notes(
    plans: Sequence[NotePlan], texts: Mapping[str, str], language: str
) -> tuple[InputNote, ...]:
    """Spáruje plány se zprávami jednoho jazyka.

    Args:
        plans: Plány zpráv korpusu.
        texts: Texty zpráv klíčované note_id.
        language: Jazyk zpráv.

    Returns:
        Řádky seřazené podle note_id.

    Raises:
        UnpairedNotesError: Korpus je prázdný, plán je duplicitní nebo plán
            a zpráva nemají protějšek.
    """
    counts = Counter(plan.note_id for plan in plans)
    if problems := _pairing_problems(counts, texts, language):
        raise UnpairedNotesError(" ".join(problems))
    by_id = {plan.note_id: plan for plan in plans}
    return tuple(
        InputNote(
            note_id,
            by_id[note_id].subject_id,
            language,
            texts[note_id],
            frozenset(by_id[note_id].labels),
        )
        for note_id in sorted(by_id)
    )


def _pairing_problems(
    counts: Counter[str], texts: Mapping[str, str], language: str
) -> list[str]:
    if not counts and not texts:
        return [f"Korpus nemá v jazyce {language} žádnou zprávu ani plán."]
    problems = []
    if duplicates := sorted(note_id for note_id, count in counts.items() if count > 1):
        problems.append(f"Duplicitní plány: {', '.join(duplicates)}.")
    if without_text := sorted(counts.keys() - texts.keys()):
        problems.append(
            f"Plány bez zprávy v jazyce {language}: {', '.join(without_text)}."
        )
    if without_plan := sorted(texts.keys() - counts.keys()):
        problems.append(
            f"Zprávy bez plánu v jazyce {language}: {', '.join(without_plan)}."
        )
    return problems
