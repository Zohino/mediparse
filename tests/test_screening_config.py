"""Mřížka klasického screeningu: hodnoty protokolu a odmítnutí neplatného configu."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mediparse.domain.classical_screening import ClassicalScreeningConfig
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.screening_config import load_classical_screening_config

WORD = {"name": "word-1-2", "analyzer": "word", "ngram_range": [1, 2]}
CHAR = {"name": "char_wb-3-5", "analyzer": "char_wb", "ngram_range": [3, 5]}


def _config() -> dict[str, object]:
    return {
        "seed": 0,
        "folds": 5,
        "threshold_folds": 3,
        "lsa_components": 100,
        "input": {
            "strip_diagnostic_sections": False,
            "text_preprocessing": "dictionary",
        },
        "vectorizers": [WORD, CHAR],
        "families": ["complement_nb", "logreg", "linear_svm", "nb_svm", "lightgbm_lsa"],
    }


def test_repository_config_matches_protocol() -> None:
    """Soubor deklaruje přesně mřížku a volby vstupu schválené v briefu."""
    path = Path(__file__).parents[1] / "config/screening_classical.json"

    assert load_classical_screening_config(path).model_dump(mode="json") == _config()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vectorizers", [WORD, WORD]),
        ("families", ["logreg", "logreg"]),
        ("vectorizers", [WORD | {"ngram_range": [3, 2]}]),
        ("folds", 1),
        ("threshold_folds", 1),
        ("lsa_components", 0),
        ("lsa_components", True),
        ("unknown", 0),
        ("vectorizers", []),
        ("families", []),
        ("vectorizers", [WORD | {"name": ""}]),
        ("vectorizers", [WORD | {"analyzer": "char"}]),
        ("vectorizers", [WORD | {"ngram_range": [0, 2]}]),
        ("vectorizers", [WORD | {"ngram_range": ["1", "2"]}]),
        ("vectorizers", [WORD | {"unknown": 0}]),
        ("families", ["unknown"]),
        ("folds", "5"),
        ("seed", -1),
        ("seed", 2**32),
        ("input", {"strip_diagnostic_sections": False, "text_preprocessing": "bad"}),
    ],
    ids=[
        "duplicate-vectorizer",
        "duplicate-family",
        "reversed-ngram-range",
        "one-fold",
        "one-threshold-fold",
        "zero-lsa-components",
        "non-strict-lsa-components",
        "unknown-key",
        "no-vectorizers",
        "no-families",
        "empty-name",
        "unknown-analyzer",
        "zero-ngram",
        "non-strict-ngram",
        "unknown-vectorizer-key",
        "unknown-family",
        "non-strict-folds",
        "negative-seed",
        "seed-out-of-range",
        "unknown-preprocessing",
    ],
)
def test_invalid_config_is_rejected(field: str, value: object) -> None:
    """Mřížka odmítne duplicity, neplatné rozsahy, prázdné položky i chybný typ."""
    body = _config() | {field: value}

    with pytest.raises(ValidationError):
        ClassicalScreeningConfig.model_validate_json(json.dumps(body))


def test_missing_file_is_invalid_input(tmp_path: Path) -> None:
    """Chybějící config se přeloží na doménovou chybu."""
    with pytest.raises(InvalidInputError, match="neexistuje"):
        load_classical_screening_config(tmp_path / "missing.json")


def test_missing_lsa_components_is_rejected() -> None:
    """Počet komponent je povinná vědecká volba i pro mřížku bez LightGBM."""
    body = _config()
    body["families"] = ["logreg"]
    del body["lsa_components"]

    with pytest.raises(ValidationError, match="lsa_components"):
        ClassicalScreeningConfig.model_validate_json(json.dumps(body))


def test_invalid_file_is_invalid_input(tmp_path: Path) -> None:
    """Chyba validace souboru se přeloží na doménovou chybu."""
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(_config() | {"folds": 1}), encoding="utf-8")

    with pytest.raises(InvalidInputError, match="neodpovídá schématu"):
        load_classical_screening_config(path)
