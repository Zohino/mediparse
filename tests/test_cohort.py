"""Doména kohorty: přiřazení diagnóz kódům, trychtýř, odmítnutí malých buněk a manifest."""

from __future__ import annotations

from typing import Final

import pytest

from mediparse.domain.cohort import RawCode, label_codes
from mediparse.domain.icd import DiagnosisCodes, IcdVersion
from mediparse.domain.labels import Diagnosis

CODES: Final = DiagnosisCodes(
    prefixes={
        Diagnosis.DIABETES: {IcdVersion.NINE: ("250",), IcdVersion.TEN: ("E11",)},
        Diagnosis.COPD: {IcdVersion.NINE: ("496",), IcdVersion.TEN: ("J44",)},
        Diagnosis.CKD: {IcdVersion.NINE: ("585",), IcdVersion.TEN: ("N18",)},
        Diagnosis.PNEUMONIA: {IcdVersion.NINE: ("486",), IcdVersion.TEN: ("J18",)},
        Diagnosis.ATRIAL_FIBRILLATION: {
            IcdVersion.NINE: ("4273",),
            IcdVersion.TEN: ("I48",),
        },
    }
)


def test_valid_codes_of_both_versions_get_diagnosis() -> None:
    """Platný kód ICD-9 i ICD-10 dostane diagnózu podle předpon."""
    labels = label_codes([("9", "25000"), ("10", "E119"), ("10", "I4891")], CODES)

    assert labels.diagnoses == {
        ("9", "25000"): Diagnosis.DIABETES,
        ("10", "E119"): Diagnosis.DIABETES,
        ("10", "I4891"): Diagnosis.ATRIAL_FIBRILLATION,
    }
    assert labels.invalid == 0


def test_code_outside_five_is_absent_and_valid() -> None:
    """Platný kód mimo pětici v mapování chybí a není neplatný."""
    labels = label_codes([("10", "S72001A")], CODES)

    assert labels.diagnoses == {}
    assert labels.invalid == 0


def test_whitespace_and_case_are_normalized() -> None:
    """Mezery a malá písmena se normalizují, klíč zůstává surový."""
    labels = label_codes([("10", " e119 ")], CODES)

    assert labels.diagnoses == {("10", " e119 "): Diagnosis.DIABETES}
    assert labels.invalid == 0


@pytest.mark.parametrize(
    "raw",
    [
        ("10", ""),
        ("10", None),
        ("10", "E11.9"),
        ("11", "E119"),
        (None, "E119"),
    ],
)
def test_invalid_pair_has_no_diagnosis_and_is_counted(raw: RawCode) -> None:
    """Dvojice, kterou nelze převést na kód, nemá diagnózu a zvyšuje počet neplatných."""
    labels = label_codes([raw], CODES)

    assert labels.diagnoses == {}
    assert labels.invalid == 1


def test_duplicate_raw_pairs_are_counted_once() -> None:
    """Neplatná dvojice se počítá jednou, i když se opakuje na vstupu."""
    labels = label_codes([("10", ""), ("10", ""), ("10", "E119")], CODES)

    assert labels.invalid == 1
    assert len(labels.diagnoses) == 1
