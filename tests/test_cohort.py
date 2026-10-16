"""Doména kohorty: přiřazení diagnóz kódům, trychtýř, odmítnutí malých buněk a manifest."""

from __future__ import annotations

from typing import Final

import pytest
from pydantic import ValidationError

from mediparse.domain.cohort import (
    MIN_CELL,
    CohortCounts,
    CohortManifest,
    FunnelStep,
    FunnelStepName,
    RawCode,
    ensure_publishable,
    label_codes,
)
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.labels import Diagnosis
from mediparse.domain.run_manifest import SourceRevision
from tests.cohort_support import CODES

POSITIVE: Final = 20


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


def _funnel(
    *notes: int, patients: tuple[int, ...] | None = None
) -> tuple[FunnelStep, ...]:
    counted = patients or tuple(max(n // 2, 1) for n in notes)
    return tuple(
        FunnelStep(step=step, notes=n, patients=p)
        for step, n, p in zip(FunnelStepName, notes, counted, strict=False)
    )


def _counts(
    positives: dict[Diagnosis, int] | None = None,
    funnel: tuple[FunnelStep, ...] | None = None,
) -> CohortCounts:
    return CohortCounts(
        funnel=funnel or _funnel(100, 80, 60, 50),
        positives=positives or dict.fromkeys(Diagnosis, POSITIVE),
    )


def test_counts_with_ten_and_ten_pass() -> None:
    """Deset pozitivních a deset negativních zpráv je nejmenší přípustná buňka."""
    counts = _counts(dict.fromkeys(Diagnosis, MIN_CELL), _funnel(30, 30, 30, 20))

    ensure_publishable(counts)

    assert counts.positives[Diagnosis.CKD] == MIN_CELL


@pytest.mark.parametrize("positives", [0, 9])
def test_diagnosis_with_few_positives_is_refused_without_number(
    positives: int,
) -> None:
    """Chyba jmenuje diagnózu a neprozradí počet."""
    with pytest.raises(InvalidInputError, match="copd") as error:
        ensure_publishable(
            _counts({**dict.fromkeys(Diagnosis, POSITIVE), Diagnosis.COPD: positives})
        )

    assert str(positives) not in str(error.value).replace("10", "")


def test_diagnosis_with_few_negatives_is_refused() -> None:
    """Diagnóza, kterou má skoro každá zpráva, nemá dost negativních."""
    with pytest.raises(InvalidInputError, match="diabetes"):
        ensure_publishable(
            _counts({**dict.fromkeys(Diagnosis, POSITIVE), Diagnosis.DIABETES: 41})
        )


def test_empty_cohort_is_refused() -> None:
    """Kohorta bez zpráv nemá z čeho vybírat."""
    with pytest.raises(InvalidInputError, match="prázdná"):
        ensure_publishable(
            _counts(funnel=_funnel(100, 80, 60, 0, patients=(50, 40, 30, 0)))
        )


def test_funnel_must_have_four_steps_in_order() -> None:
    """Trychtýř má právě čtyři kroky v pořadí sešitu 5."""
    steps = _funnel(100, 80, 60, 50)

    with pytest.raises(ValidationError, match="trychtýře"):
        _counts(funnel=steps[:3])
    with pytest.raises(ValidationError, match="trychtýře"):
        _counts(funnel=(steps[1], steps[0], *steps[2:]))


def test_funnel_must_not_grow() -> None:
    """Každý krok vybírá z předchozího, počty nerostou."""
    with pytest.raises(ValidationError, match="trychtýře"):
        _counts(funnel=_funnel(100, 80, 90, 50))


def test_funnel_step_rejects_numeric_string() -> None:
    """Vnořený model je přísný sám: počet jako řetězec neprojde."""
    with pytest.raises(ValidationError):
        FunnelStep.model_validate({
            "step": FunnelStepName.ALL_NOTES,
            "notes": "5",
            "patients": 5,
        })


def _manifest(**overrides: object) -> CohortManifest:
    fields: dict[str, object] = {
        "cohort_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "sources": {"discharge.csv.gz": "c" * 64},
        "packages": {"python": "3.13.0"},
        "source": SourceRevision(commit="d" * 40, dirty=False),
        "counts": _counts(),
        "invalid_codes": 0,
    }
    return CohortManifest.model_validate(fields | overrides)


def test_manifest_serializes_counts_by_diagnosis_name() -> None:
    """Pozitivní zprávy jsou v JSON pod jménem diagnózy."""
    dumped = _manifest().model_dump(mode="json")

    assert dumped["counts"]["positives"]["atrial_fibrillation"] == POSITIVE
    assert dumped["counts"]["funnel"][0]["step"] == "all_notes"


def test_manifest_rejects_unknown_field_and_bad_digest() -> None:
    """Manifest je přísný: cizí pole a otisk, který není sha256, neprojdou."""
    with pytest.raises(ValidationError):
        _manifest(unknown=1)
    with pytest.raises(ValidationError):
        _manifest(cohort_sha256="xyz")
