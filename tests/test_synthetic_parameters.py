"""Parametry pětice z exportu sešitu 8: validace exportu a odvození počtu zpráv."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from mediparse.domain.synthetic_parameters import SyntheticParameters, required_notes
from tests.support import export_payload

KEYS = ("a", "b", "c")


def _parse(payload: dict[str, Any]) -> SyntheticParameters:
    return SyntheticParameters.model_validate_json(json.dumps(payload))


def test_valid_export_is_parsed() -> None:
    """Platný export nese výběr, prevalence, matici a míry zmínek ve zvoleném pořadí."""
    parameters = _parse(export_payload(KEYS))

    assert parameters.selection == KEYS
    assert parameters.prevalence["b"] == pytest.approx(0.2)
    assert parameters.conditional["a"]["a"] == pytest.approx(1.0)
    assert parameters.mentions["c"].keywords == ("c phrase", "C")


@pytest.mark.parametrize("block", ["prevalence", "conditional", "mentions"])
def test_missing_key_in_block_is_rejected(block: str) -> None:
    """Blok bez klíče z výběru nejde použít."""
    payload = export_payload(KEYS)
    del payload[block]["a"]

    with pytest.raises(ValidationError, match="selection"):
        _parse(payload)


def test_key_outside_selection_is_rejected() -> None:
    """Klíč mimo výběr je chyba, i když výběr samotný je úplný."""
    payload = export_payload(KEYS)
    payload["prevalence"]["z"] = 0.1

    with pytest.raises(ValidationError, match="selection"):
        _parse(payload)


def test_matrix_row_with_foreign_or_missing_column_is_rejected() -> None:
    """Řádek matice musí mít právě sloupce výběru."""
    payload = export_payload(KEYS)
    del payload["conditional"]["b"]["c"]

    with pytest.raises(ValidationError, match="selection"):
        _parse(payload)


def test_null_in_prevalence_is_rejected() -> None:
    """Nedefinovaná prevalence (null) export znehodnotí."""
    payload = export_payload(KEYS)
    payload["prevalence"]["a"] = None

    with pytest.raises(ValidationError):
        _parse(payload)


def test_null_in_matrix_is_rejected() -> None:
    """Nedefinovaná buňka matice (null) export znehodnotí."""
    payload = export_payload(KEYS)
    payload["conditional"]["a"]["b"] = None

    with pytest.raises(ValidationError):
        _parse(payload)


@pytest.mark.parametrize("field", ["given_positive", "given_negative"])
def test_null_in_mention_rate_is_rejected(field: str) -> None:
    """Nedefinovaná míra zmínky (null) export znehodnotí."""
    payload = export_payload(KEYS)
    payload["mentions"]["a"][field] = None

    with pytest.raises(ValidationError):
        _parse(payload)


def test_empty_selection_is_rejected() -> None:
    """Export z prázdného korpusu má prázdný výběr a nejde použít."""
    with pytest.raises(ValidationError):
        _parse(export_payload(()))


def test_duplicate_selection_is_rejected() -> None:
    """Výběr nesmí obsahovat tutéž diagnózu dvakrát."""
    payload = export_payload(KEYS)
    payload["selection"] = ["a", "a", "b"]

    with pytest.raises(ValidationError):
        _parse(payload)


def test_zero_prevalence_is_rejected() -> None:
    """Nulová prevalence by dala nulový počet zpráv, takže ji validace odmítne."""
    with pytest.raises(ValidationError):
        _parse(export_payload(KEYS, prevalence=0.0))


def test_empty_keywords_are_rejected() -> None:
    """Zmínka bez klíčového slova nejde vygenerovat."""
    payload = export_payload(KEYS)
    payload["mentions"]["a"]["keywords"] = []

    with pytest.raises(ValidationError):
        _parse(payload)


@pytest.mark.parametrize(
    ("rarest", "expected"),
    [(0.0636, 400), (0.0625, 400), (0.06, 500), (0.2, 200), (0.9, 100)],
)
def test_required_notes_rounds_up_to_whole_hundreds(
    rarest: float, expected: int
) -> None:
    """Nejmenší stovka, při níž nejvzácnější label v pětině korpusu čeká aspoň 5 pozitivních zpráv."""
    prevalence = {"a": 0.4, "b": rarest, "c": 0.3}

    assert required_notes(prevalence) == expected
