"""Vazba pracovního adresáře překladu na anglický a český korpus."""

from __future__ import annotations

from mediparse.domain.translation_consistency import (
    TranslatedNote,
    binding_violations,
    joined_translation,
)
from tests.workdir_support import sha256_text


def _note(note_id: str, source: str, *parts: str) -> TranslatedNote:
    return TranslatedNote(note_id, sha256_text(source), parts)


def test_joined_translation_matches_collect_normalization() -> None:
    """Části se spojí prázdným řádkem, řádky ztratí koncové mezery a text končí LF."""
    assert joined_translation([" a  \nb ", "\nc\n"]) == "a\nb\n\nc\n"


def test_matching_binding_has_no_violations() -> None:
    """Originál, množina note_id i složený text sedí."""
    notes = [_note("n1", "one", "jedna ", " dva")]

    assert binding_violations(notes, {"n1": "one"}, {"n1": "jedna\n\ndva\n"}) == ()


def test_changed_original_is_reported() -> None:
    """Jiná verze EN originálu, než kterou překládal běh."""
    (reason,) = binding_violations(
        [_note("n1", "one", "jedna")], {"n1": "changed"}, {"n1": "jedna\n"}
    )

    assert "en/n1" in reason


def test_note_ids_must_match_both_ways() -> None:
    """note_id navíc i chybějící v cs/ proti požadavkům."""
    notes = [_note("n1", "one", "jedna"), _note("n2", "two", "dva")]
    english = {"n1": "one", "n2": "two", "n3": "three"}
    czech = {"n1": "jedna\n", "n3": "tři\n"}

    reasons = binding_violations(notes, english, czech)

    assert any("cs/n2" in reason and "chybí v cs/" in reason for reason in reasons)
    assert any(
        "cs/n3" in reason and "není v požadavcích" in reason for reason in reasons
    )


def test_czech_text_must_be_joined_outputs() -> None:
    """Ručně upravený cs/ text neodpovídá výstupům."""
    (reason,) = binding_violations(
        [_note("n1", "one", "jedna")], {"n1": "one"}, {"n1": "upraveno\n"}
    )

    assert "cs/n1" in reason
