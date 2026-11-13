"""Obal vstupní tabulky, který ze zpráv odstraní diagnostické sekce (ablace D2)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.section_headers import (
    LEXICON_LANGUAGE,
    strip_diagnostic_sections,
)

if TYPE_CHECKING:
    from mediparse.application.ports.experiment import PatientNoteTable
    from mediparse.domain.identity import SubjectId
    from mediparse.domain.input_note import InputNote
    from mediparse.domain.section_headers import SectionLexicon


@dataclass(frozen=True)
class DiagnosticStripTable:
    """Vstupní tabulka se zprávami bez diagnostických sekcí lexikonu.

    Trénink i inference ji skládají stejně, aby klasifikovaly týž text.
    """

    table: PatientNoteTable
    lexicon: SectionLexicon

    def read(self) -> tuple[InputNote, ...]:
        """Přečte řádky obalené tabulky a ze zpráv odstraní diagnostické sekce.

        Returns:
            Řádky v pořadí tabulky, změněný je jen text.

        Raises:
            InvalidInputError: Zpráva není v jazyce lexikonu, strip by nic
                neodstranil.
        """
        return tuple(self._stripped(note) for note in self.table.read())

    def subjects(self) -> frozenset[SubjectId]:
        """Vrátí pacienty obalené tabulky bez čtení textů.

        Returns:
            Všichni pacienti zdroje.
        """
        return self.table.subjects()

    def read_patients(self, subjects: frozenset[SubjectId]) -> tuple[InputNote, ...]:
        """Přečte a stripuje jen zprávy vybraných pacientů v pořadí tabulky.

        Returns:
            Vybrané zprávy se změněným textem.
        """
        return tuple(
            self._stripped(note) for note in self.table.read_patients(subjects)
        )

    def _stripped(self, note: InputNote) -> InputNote:
        if note.language is not LEXICON_LANGUAGE:
            # note_id nese subject_id pacienta MIMIC; hláška ani log ho nesmí nést.
            msg = (
                f"Zpráva v jazyce {note.language}, lexikon sekcí "
                f"je v jazyce {LEXICON_LANGUAGE} a strip by nic neodstranil."
            )
            raise InvalidInputError(msg)
        return replace(note, text=strip_diagnostic_sections(note.text, self.lexicon))
