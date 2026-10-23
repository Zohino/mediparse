"""Obal vstupní tabulky, který ze zpráv odstraní diagnostické sekce (ablace D2)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.language import Language
from mediparse.domain.section_headers import strip_diagnostic_sections

if TYPE_CHECKING:
    from mediparse.application.ports.experiment import InputTable
    from mediparse.domain.input_note import InputNote
    from mediparse.domain.section_headers import SectionLexicon


@dataclass(frozen=True)
class DiagnosticStripTable:
    """Vstupní tabulka se zprávami bez diagnostických sekcí lexikonu.

    Trénink i inference ji skládají stejně, aby klasifikovaly týž text.
    """

    table: InputTable
    lexicon: SectionLexicon

    def read(self) -> tuple[InputNote, ...]:
        """Přečte řádky obalené tabulky a ze zpráv odstraní diagnostické sekce.

        Returns:
            Řádky v pořadí tabulky, změněný je jen text.

        Raises:
            InvalidInputError: Zpráva není anglická; lexikon je anglický a
                strip by nic neodstranil.
        """
        return tuple(self._stripped(note) for note in self.table.read())

    def _stripped(self, note: InputNote) -> InputNote:
        if note.language is not Language.EN:
            msg = (
                f"Zpráva {note.note_id} je v jazyce {note.language}, "
                "lexikon sekcí je anglický a strip by nic neodstranil."
            )
            raise InvalidInputError(msg)
        return replace(note, text=strip_diagnostic_sections(note.text, self.lexicon))
