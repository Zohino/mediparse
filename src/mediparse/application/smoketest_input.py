"""Use case vstupní tabulky smoketestu: anglické zprávy korpusu s labely svých plánů."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.application.corpus_texts import LANGUAGE, texts_in_language
from mediparse.domain.smoketest_input import input_notes

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.application.ports import NoteSource, PlanSource
    from mediparse.domain.smoketest_input import InputNote


class NoteTable(Protocol):
    """Vstupní tabulka kroků pipeline."""

    def write(self, notes: Sequence[InputNote]) -> None:
        """Zapíše řádky v daném pořadí a nahradí předchozí obsah."""


@dataclass(frozen=True)
class SmoketestInput:
    """Vstupní tabulka smoketestu z plánů a anglických zpráv syntetického korpusu."""

    plans: PlanSource
    corpus: NoteSource
    table: NoteTable

    def run(self) -> int:
        """Spáruje anglické zprávy s plány a zapíše je do tabulky.

        Returns:
            Počet zapsaných zpráv.
        """
        texts = texts_in_language(self.corpus.notes(), LANGUAGE)
        notes = input_notes(self.plans.load(), texts, LANGUAGE)
        self.table.write(notes)
        return len(notes)
