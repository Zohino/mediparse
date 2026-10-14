"""Načtení předpon pětice z configu kandidátů."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from mediparse.domain.icd import IcdCode, IcdVersion
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.labels import Diagnosis
from mediparse.infrastructure.diagnosis_codes_file import (
    DIAGNOSIS_CANDIDATES_PATH,
    load_diagnosis_codes,
)
from tests.support import REPOSITORY

if TYPE_CHECKING:
    from pathlib import Path

_FIVE: dict[str, tuple[list[str], list[str]]] = {
    "diabetes": (["250"], ["E11"]),
    "copd": (["496"], ["J44"]),
    "ckd": (["585"], ["N18"]),
    "pneumonia": (["486"], ["J18"]),
    "atrial_fibrillation": (["4273"], ["I48"]),
}


def _candidates(skip: str | None = None) -> dict[str, Any]:
    candidates: dict[str, Any] = {
        key: {
            "label_cs": key,
            "icd9_prefixes": nine,
            "icd10_prefixes": ten,
            "charlson": ["x"],
        }
        for key, (nine, ten) in _FIVE.items()
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
    with pytest.raises(InvalidInputError, match="ckd"):
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
    codes = load_diagnosis_codes(REPOSITORY / DIAGNOSIS_CANDIDATES_PATH)

    assert set(codes.prefixes) == set(Diagnosis)
