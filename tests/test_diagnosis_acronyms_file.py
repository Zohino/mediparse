"""Zkratky pětice z configu kandidátů."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.diagnosis_acronyms_file import load_text_normalization
from tests.support import REPOSITORY_DIAGNOSIS_CANDIDATES

if TYPE_CHECKING:
    from pathlib import Path


def _write(tmp_path: Path, tree: dict[str, Any]) -> Path:
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(tree), encoding="utf-8")
    return path


_NAMES: dict[str, dict[str, list[str]]] = {
    "diabetes": {"keywords_en": ["diabetes", "diabetic"], "acronyms_en": ["DM", "DM2"]},
    "copd": {
        "keywords_en": ["chronic obstructive pulmonary disease"],
        "acronyms_en": ["COPD"],
    },
    "ckd": {
        "keywords_en": ["chronic kidney disease", "end-stage renal disease"],
        "acronyms_en": ["CKD", "ESRD"],
    },
    "pneumonia": {"keywords_en": ["pneumonia"], "acronyms_en": []},
    "atrial_fibrillation": {
        "keywords_en": ["atrial fibrillation", "afib"],
        "acronyms_en": ["AF"],
    },
}


def _named(**changes: dict[str, list[str]] | None) -> dict[str, Any]:
    names = {**_NAMES, "heart_failure": {"keywords_en": ["x"], "acronyms_en": ["MI"]}}
    for key, value in changes.items():
        if value is None:
            del names[key]
        else:
            names[key] = value
    return {"candidates": names}


def test_repository_config_gives_exactly_the_five_acronym_pairs() -> None:
    """Reálný config dá šest dvojic pětice a zkratky ostatních kandidátů vynechá."""
    normalization = load_text_normalization(REPOSITORY_DIAGNOSIS_CANDIDATES)

    assert normalization.acronyms == {
        "AF": "atrial fibrillation",
        "CKD": "chronic kidney disease",
        "COPD": "chronic obstructive pulmonary disease",
        "DM": "diabetes",
        "DM2": "diabetes",
        "ESRD": "chronic kidney disease",
    }


def test_canonical_form_is_first_keyword_and_other_candidates_are_ignored(
    tmp_path: Path,
) -> None:
    """Tvar je první klíčové slovo; zkratka ostatních kandidátů se nenačte."""
    normalization = load_text_normalization(_write(tmp_path, _named()))

    assert normalization.acronyms["ESRD"] == "chronic kidney disease"
    assert "MI" not in normalization.acronyms


def test_missing_diagnosis_in_names_is_invalid_input(tmp_path: Path) -> None:
    """Chybějící diagnóza pětice je chyba vstupu."""
    with pytest.raises(InvalidInputError, match="Chybí diagnózy: ckd"):
        load_text_normalization(_write(tmp_path, _named(ckd=None)))


def test_acronym_without_keywords_is_invalid_input(tmp_path: Path) -> None:
    """Zkratka bez klíčových slov nemá kanonický tvar."""
    tree = _named(copd={"keywords_en": [], "acronyms_en": ["COPD"]})

    with pytest.raises(InvalidInputError, match=r"candidates\.json"):
        load_text_normalization(_write(tmp_path, tree))


def test_acronym_of_two_diagnoses_is_invalid_input(tmp_path: Path) -> None:
    """Zkratka u dvou diagnóz by tiše přepsala slovník."""
    tree = _named(pneumonia={"keywords_en": ["pneumonia"], "acronyms_en": ["AF"]})

    with pytest.raises(InvalidInputError, match="AF"):
        load_text_normalization(_write(tmp_path, tree))


def test_names_missing_file_and_invalid_json_are_invalid_input(
    tmp_path: Path,
) -> None:
    """Chybějící soubor a neplatný JSON jsou chyba vstupu."""
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")

    with pytest.raises(InvalidInputError):
        load_text_normalization(tmp_path / "none.json")
    with pytest.raises(InvalidInputError):
        load_text_normalization(broken)
