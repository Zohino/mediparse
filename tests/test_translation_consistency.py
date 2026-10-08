"""Kontroly českého překladu: úplnost, neprázdnost, poměr délky a počet značek ___."""

from __future__ import annotations

from typing import Final

from mediparse.domain.translation_consistency import (
    LENGTH_RATIO,
    deid_markers,
    marker_mismatches,
    translation_violations,
)

TEXT = "a" * 10
TWO: Final = 2


def test_marker_is_exactly_three_underscores() -> None:
    """Značka je právě ___, delší i kratší řady podtržítek se nepočítají."""
    assert deid_markers("x ___ y") == 1
    assert deid_markers("____") == 0
    assert deid_markers("__") == 0
    assert deid_markers("___ ___") == TWO


def test_corpus_without_czech_has_no_violations() -> None:
    """Korpus před překladem nic neporušuje."""
    assert translation_violations({"n1": TEXT}, {}) == ()


def test_czech_without_original_is_reported() -> None:
    """Překlad bez anglického originálu se hlásí."""
    (reason,) = translation_violations({"n1": TEXT}, {"n1": TEXT, "n2": TEXT})

    assert "cs/n2" in reason


def test_original_without_czech_is_reported() -> None:
    """Originál bez překladu se hlásí, jakmile cs obsahuje cokoli."""
    (reason,) = translation_violations({"n1": TEXT, "n2": TEXT}, {"n1": TEXT})

    assert "cs/n2" in reason


def test_empty_translation_is_reported() -> None:
    """Prázdný nebo jen bílé znaky je prázdný překlad."""
    (reason,) = translation_violations({"n1": TEXT}, {"n1": " \n"})

    assert "cs/n1" in reason


def test_length_ratio_outside_bounds_is_reported() -> None:
    """Překlad kratší nebo delší než meze poměru znaků se hlásí."""
    low, high = LENGTH_RATIO

    assert translation_violations({"n1": TEXT}, {"n1": "a" * 7})
    assert translation_violations({"n1": TEXT}, {"n1": "a" * 17})
    assert low < high


def test_length_ratio_on_bounds_passes() -> None:
    """Poměr přesně na mezi je v pořádku."""
    assert translation_violations({"n1": TEXT}, {"n1": "a" * 8}) == ()
    assert translation_violations({"n1": TEXT}, {"n1": "a" * 16}) == ()


def test_mismatches_are_sorted_and_only_differing() -> None:
    """Neshoda počtu značek se hlásí seřazeně a jen u zpráv, kde se počty liší."""
    english = {"b": "___ ___", "a": "___", "c": "___"}
    czech = {"b": "___", "a": "", "c": "___", "d": "___"}

    assert marker_mismatches(english, czech) == ("a", "b")


def test_mismatches_ignore_longer_underscore_runs() -> None:
    """Řada ____ není značka, takže neshodu nevyrobí."""
    assert marker_mismatches({"a": "___"}, {"a": "___ ____"}) == ()


def test_empty_original_is_reported() -> None:
    """Prázdný anglický originál se hlásí vlastním důvodem, poměr délky se nepočítá."""
    (reason,) = translation_violations({"n1": ""}, {"n1": TEXT})

    assert "en/n1" in reason
    assert "prázdný" in reason
