"""Doména kohorty: přiřazení diagnóz kódům, trychtýř, odmítnutí malých buněk a manifest."""

from __future__ import annotations

import re

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
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.environment import Environment, SourceRevision
from mediparse.domain.inputs import InvalidInputError
from tests.cohort_support import (
    CODES,
    EMPTY_COHORT,
    FEW_NEGATIVES,
    FEW_POSITIVES,
    UNIQUE,
    make_counts,
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


def _funnel(
    *notes: int, patients: tuple[int, ...] | None = None
) -> tuple[FunnelStep, ...]:
    counted = patients or tuple(max(n // 2, 1) for n in notes)
    return tuple(
        FunnelStep(step=step, notes=n, patients=p)
        for step, n, p in zip(FunnelStepName, notes, counted, strict=False)
    )


def test_counts_with_ten_and_ten_pass() -> None:
    """Deset pozitivních a deset negativních zpráv je nejmenší přípustná buňka."""
    counts = make_counts(
        dict.fromkeys(Diagnosis, MIN_CELL), funnel=_funnel(30, 30, 30, 20)
    )

    ensure_publishable(counts)

    assert counts.positives[Diagnosis.CKD] == MIN_CELL


def _numbers(error: pytest.ExceptionInfo[InvalidInputError]) -> set[str]:
    return set(re.findall(r"\d+", str(error.value)))


@pytest.mark.parametrize(
    "counts",
    [
        FEW_POSITIVES,
        make_counts({**dict.fromkeys(Diagnosis, UNIQUE // 2), Diagnosis.COPD: 0}),
    ],
    ids=["seven", "zero"],
)
def test_diagnosis_with_few_positives_is_refused_without_number(
    counts: CohortCounts,
) -> None:
    """Hláška jmenuje diagnózu a z čísel nese nejvýš práh."""
    with pytest.raises(InvalidInputError, match="copd") as error:
        ensure_publishable(counts)

    assert _numbers(error) <= {str(MIN_CELL)}


def test_diagnosis_with_few_negatives_is_refused_without_number() -> None:
    """Diagnóza, kterou má skoro každá zpráva, nemá dost negativních a hláška je bez počtů."""
    with pytest.raises(InvalidInputError, match="diabetes") as error:
        ensure_publishable(FEW_NEGATIVES)

    assert _numbers(error) <= {str(MIN_CELL)}


def test_empty_cohort_is_refused_without_number() -> None:
    """Kohorta bez zpráv nemá z čeho vybírat; hláška je bez počtů."""
    with pytest.raises(InvalidInputError, match="prázdná") as error:
        ensure_publishable(EMPTY_COHORT)

    assert _numbers(error) <= {str(MIN_CELL)}


def test_funnel_must_have_four_steps_in_order() -> None:
    """Trychtýř má právě čtyři kroky v pořadí sešitu 5."""
    steps = _funnel(100, 80, 60, 50)

    with pytest.raises(ValidationError, match="trychtýře"):
        make_counts(funnel=steps[:3])
    with pytest.raises(ValidationError, match="trychtýře"):
        make_counts(funnel=(steps[1], steps[0], *steps[2:]))


def test_funnel_must_not_grow() -> None:
    """Každý krok vybírá z předchozího, počty nerostou."""
    with pytest.raises(ValidationError, match="trychtýře"):
        make_counts(funnel=_funnel(100, 80, 90, 50))


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
        "environment": Environment(
            packages={"python": "3.13.0"},
            source=SourceRevision(commit="d" * 40, dirty=False),
        ),
        "counts": make_counts(),
        "invalid_codes": 0,
    }
    return CohortManifest.model_validate(fields | overrides)


def test_manifest_serializes_counts_by_diagnosis_name() -> None:
    """Pozitivní zprávy jsou v JSON pod jménem diagnózy."""
    dumped = _manifest().model_dump(mode="json")

    assert dumped["counts"]["positives"]["atrial_fibrillation"] == UNIQUE // 2
    assert dumped["counts"]["funnel"][0]["step"] == "all_notes"


def test_manifest_rejects_unknown_field_and_bad_digest() -> None:
    """Manifest je přísný: cizí pole a otisk, který není sha256, neprojdou."""
    with pytest.raises(ValidationError):
        _manifest(unknown=1)
    with pytest.raises(ValidationError):
        _manifest(cohort_sha256="xyz")


def test_manifest_nests_environment_instead_of_flat_fields() -> None:
    """Serializovaný manifest nese ``environment`` a nemá ``packages`` ani ``source`` nahoře."""
    dumped = _manifest().model_dump(mode="json")

    assert dumped["environment"] == {
        "packages": {"python": "3.13.0"},
        "source": {"commit": "d" * 40, "dirty": False},
    }
    assert "packages" not in dumped
    assert "source" not in dumped
