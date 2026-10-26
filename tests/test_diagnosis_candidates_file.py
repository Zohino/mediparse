"""Načtení předpon pětice z configu kandidátů."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.icd import IcdCode, IcdVersion
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.diagnosis_candidates_file import (
    load_diagnosis_codes,
    load_diagnosis_keywords,
    load_text_normalization,
)
from tests.cohort_support import candidate_prefixes
from tests.support import REPOSITORY_DIAGNOSIS_CANDIDATES

if TYPE_CHECKING:
    from pathlib import Path


def _candidates(skip: str | None = None) -> dict[str, Any]:
    candidates: dict[str, Any] = {
        key: {"label_cs": key, "charlson": ["x"], **prefixes}
        for key, prefixes in candidate_prefixes().items()
        if key != skip
    }
    candidates["sepsis"] = {"icd9_prefixes": ["038"], "icd10_prefixes": ["A41"]}
    return {"candidates": candidates}


def _write(tmp_path: Path, tree: dict[str, Any]) -> Path:
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(tree), encoding="utf-8")
    return path


def test_loads_five_and_ignores_other_candidates(tmp_path: Path) -> None:
    """Pětice se načte a kandidát navíc s dalšími poli nevadí."""
    codes = load_diagnosis_codes(_write(tmp_path, _candidates()))

    assert codes.diagnosis_of(IcdCode(version=IcdVersion.TEN, code="J449")) is (
        Diagnosis.COPD
    )
    assert codes.diagnosis_of(IcdCode(version=IcdVersion.TEN, code="A419")) is None


def test_missing_diagnosis_is_invalid_input(tmp_path: Path) -> None:
    """Chybějící diagnóza pětice je chyba vstupu."""
    with pytest.raises(InvalidInputError, match="Chybí diagnózy: ckd"):
        load_diagnosis_codes(_write(tmp_path, _candidates(skip="ckd")))


def test_missing_file_is_invalid_input(tmp_path: Path) -> None:
    """Chybějící soubor je chyba vstupu."""
    with pytest.raises(InvalidInputError):
        load_diagnosis_codes(tmp_path / "none.json")


def test_invalid_json_is_invalid_input(tmp_path: Path) -> None:
    """Neplatný JSON je chyba vstupu."""
    path = tmp_path / "candidates.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(InvalidInputError):
        load_diagnosis_codes(path)


def test_repository_config_loads() -> None:
    """Config kandidátů v repu se načte."""
    codes = load_diagnosis_codes(REPOSITORY_DIAGNOSIS_CANDIDATES)

    assert set(codes.prefixes) == set(Diagnosis)


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

    with pytest.raises(InvalidInputError, match="keywords_en"):
        load_text_normalization(_write(tmp_path, tree))


def test_acronym_of_two_diagnoses_is_invalid_input(tmp_path: Path) -> None:
    """Zkratka u dvou diagnóz by tiše přepsala slovník."""
    tree = _named(pneumonia={"keywords_en": ["pneumonia"], "acronyms_en": ["AF"]})

    with pytest.raises(InvalidInputError, match="u více diagnóz"):
        load_text_normalization(_write(tmp_path, tree))


def test_missing_acronyms_field_is_invalid_input(tmp_path: Path) -> None:
    """Pole ``acronyms_en`` je povinné, chybějící se nedoplní prázdným výchozím."""
    tree = _named(copd={"keywords_en": ["chronic obstructive pulmonary disease"]})

    with pytest.raises(InvalidInputError, match="acronyms_en"):
        load_text_normalization(_write(tmp_path, tree))


def test_candidate_without_names_and_acronyms_is_accepted(tmp_path: Path) -> None:
    """Diagnóza bez klíčových slov i zkratek slovník nezměří, ale nevadí."""
    tree = _named(pneumonia={"keywords_en": [], "acronyms_en": []})

    normalization = load_text_normalization(_write(tmp_path, tree))

    assert "AF" in normalization.acronyms


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


def test_keywords_follow_diagnosis_order_and_skip_other_candidates(
    tmp_path: Path,
) -> None:
    """Klíčová slova pětice jdou v pořadí ``Diagnosis``; kandidáti mimo pětici se neberou."""
    keywords = load_diagnosis_keywords(_write(tmp_path, _named()))

    assert keywords == (
        "diabetes",
        "diabetic",
        "chronic obstructive pulmonary disease",
        "chronic kidney disease",
        "end-stage renal disease",
        "pneumonia",
        "atrial fibrillation",
        "afib",
    )


def test_keywords_of_missing_diagnosis_are_invalid_input(tmp_path: Path) -> None:
    """Chybějící diagnóza pětice je chyba vstupu."""
    with pytest.raises(InvalidInputError, match="Chybí diagnózy: ckd"):
        load_diagnosis_keywords(_write(tmp_path, _named(ckd=None)))
