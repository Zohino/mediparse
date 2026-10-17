"""Diagnóza: pětice a jediná akutní diagnóza."""

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from mediparse.domain.diagnosis import ACUTE_DIAGNOSIS, Diagnosis, PerDiagnosis


def test_five_follow_notebook_8_with_one_acute_diagnosis() -> None:
    """Pětice sešitu 8 v pořadí kandidátů; jedinou akutní diagnózou je pneumonie."""
    assert [d.value for d in Diagnosis] == [
        "diabetes",
        "copd",
        "ckd",
        "pneumonia",
        "atrial_fibrillation",
    ]
    assert [d for d in Diagnosis if not d.chronic] == [ACUTE_DIAGNOSIS]
    assert ACUTE_DIAGNOSIS is Diagnosis.PNEUMONIA


class _Counts(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    values: PerDiagnosis[int]


class _Matrix(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    rows: PerDiagnosis[PerDiagnosis[int]]


def test_complete_mapping_is_accepted() -> None:
    """Slovník s hodnotou každé diagnózy projde a zůstane obyčejným slovníkem."""
    counts = _Counts(values=dict.fromkeys(Diagnosis, 1))

    assert counts.values[Diagnosis.COPD] == 1


def test_missing_diagnoses_are_listed_sorted() -> None:
    """Hláška vyjmenuje chybějící diagnózy seřazeně."""
    values = dict.fromkeys(Diagnosis, 1)
    del values[Diagnosis.DIABETES]
    del values[Diagnosis.COPD]

    with pytest.raises(ValidationError, match=r"Chybí diagnózy: copd, diabetes\."):
        _Counts(values=values)


def test_empty_mapping_is_rejected() -> None:
    """Prázdný slovník nemá žádnou diagnózu."""
    with pytest.raises(ValidationError, match="Chybí diagnózy"):
        _Counts(values={})


def test_missing_inner_key_reports_path_to_row() -> None:
    """Chybějící vnitřní klíč nese v chybě cestu k řádku."""
    rows = {d: dict.fromkeys(Diagnosis, 1) for d in Diagnosis}
    del rows[Diagnosis.COPD][Diagnosis.CKD]

    with pytest.raises(ValidationError, match="ckd") as raised:
        _Matrix(rows=rows)

    assert raised.value.errors()[0]["loc"] == ("rows", "copd")
