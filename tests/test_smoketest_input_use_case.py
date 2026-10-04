"""Use case vstupní tabulky smoketestu nad fakes: výběr anglických zpráv a zápis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from mediparse.application.smoketest_input import SmoketestInput
from mediparse.domain.smoketest_input import InputNote, UnpairedNotesError
from tests.support import PlansFake

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tests.conftest import PlannedNote


@dataclass(frozen=True)
class _Corpus:
    texts: dict[str, str]

    def notes(self) -> dict[str, str]:
        return self.texts


@dataclass
class _Table:
    written: list[tuple[InputNote, ...]] = field(default_factory=list)

    def write(self, notes: Sequence[InputNote]) -> None:
        self.written.append(tuple(notes))


def test_writes_english_notes_with_plan_labels(planned_note: PlannedNote) -> None:
    """Anglické zprávy jdou do tabulky s labely plánu; české se ignorují."""
    plan = planned_note.plan
    corpus = _Corpus({
        f"en/{plan.note_id}.txt": planned_note.text,
        f"cs/{plan.note_id}.txt": "český překlad",
    })
    table = _Table()

    count = SmoketestInput(PlansFake((plan,)), corpus, table).run()

    assert count == 1
    assert table.written == [
        (
            InputNote(
                plan.note_id,
                plan.subject_id,
                "en",
                planned_note.text,
                frozenset(plan.labels),
            ),
        )
    ]


def test_unpaired_corpus_writes_nothing(planned_note: PlannedNote) -> None:
    """Chyba párování projde k vstupnímu bodu dřív, než něco vznikne."""
    table = _Table()

    with pytest.raises(UnpairedNotesError):
        SmoketestInput(PlansFake((planned_note.plan,)), _Corpus({}), table).run()

    assert table.written == []
