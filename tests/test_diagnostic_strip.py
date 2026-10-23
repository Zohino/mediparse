"""Obal vstupní tabulky, který ze zpráv odstraní diagnostické sekce."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from mediparse.application.diagnostic_strip import DiagnosticStripTable
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.input_note import InputNote
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.language import Language
from mediparse.domain.note import note_id
from mediparse.domain.section_headers import strip_diagnostic_sections
from tests.section_support import REPOSITORY_LEXICON, mimic_shaped_notes


@dataclass(frozen=True)
class _Table:
    notes: tuple[InputNote, ...]

    def read(self) -> tuple[InputNote, ...]:
        return self.notes


def _notes(language: Language = Language.EN) -> tuple[InputNote, ...]:
    return tuple(
        InputNote(
            note_id(order, 1),
            order,
            language,
            text,
            frozenset({Diagnosis.DIABETES}) if order % 2 else frozenset(),
        )
        for order, text in enumerate(mimic_shaped_notes().values(), start=1)
    )


def _stripped(notes: tuple[InputNote, ...]) -> tuple[InputNote, ...]:
    return DiagnosticStripTable(_Table(notes), REPOSITORY_LEXICON).read()


def test_text_equals_domain_strip_and_other_fields_are_kept() -> None:
    """Text prochází doménovým stripem, ostatní pole a pořadí zůstávají."""
    notes = _notes()

    result = _stripped(notes)

    assert [n.text for n in result] == [
        strip_diagnostic_sections(n.text, REPOSITORY_LEXICON) for n in notes
    ]
    assert [(n.note_id, n.subject_id, n.language, n.labels) for n in result] == [
        (n.note_id, n.subject_id, n.language, n.labels) for n in notes
    ]


def test_note_without_diagnostic_section_is_unchanged() -> None:
    """Zpráva bez diagnostické sekce zůstane po bajtech stejná."""
    text = mimic_shaped_notes()["headerless_addendum"]
    note = InputNote(note_id(1, 1), 1, Language.EN, text, frozenset())

    assert _stripped((note,))[0].text == text


def test_empty_table_gives_empty_tuple() -> None:
    """Prázdná tabulka dá prázdnou n-tici."""
    assert _stripped(()) == ()


def test_non_english_note_is_rejected_without_leaking_text() -> None:
    """Neanglická zpráva se odmítne, hláška nese ``note_id`` a ne text."""
    notes = _notes(Language.CS)

    with pytest.raises(InvalidInputError, match=notes[0].note_id) as error:
        _stripped(notes)

    assert notes[0].text[:40] not in str(error.value)
