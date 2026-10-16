"""Identita pacienta a řádku matice: kladné číslo a skladba id, kterou vydává matice."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from mediparse.domain.identity import RowId, SubjectId


class _Holder(BaseModel):
    row_id: RowId


def test_row_id_accepts_matrix_composition() -> None:
    """Malá písmena a číslice oddělené jednou pomlčkou projdou."""
    assert _Holder(row_id="smoketest-en-svm").row_id == "smoketest-en-svm"


@pytest.mark.parametrize("row_id", ["", "Smoke", "a b", "-a", "a--b"])
def test_row_id_rejects_other_composition(row_id: str) -> None:
    """Prázdné id, velká písmena, mezera a špatně umístěné pomlčky neprojdou."""
    with pytest.raises(ValidationError):
        _Holder(row_id=row_id)


class _Patient(BaseModel):
    subject_id: SubjectId


@pytest.mark.parametrize("subject_id", [0, -1])
def test_subject_id_rejects_non_positive(subject_id: int) -> None:
    """Pacient je kladné číslo."""
    with pytest.raises(ValidationError):
        _Patient(subject_id=subject_id)


def test_subject_id_accepts_positive() -> None:
    """Kladné číslo projde."""
    subject = 10000032

    assert _Patient(subject_id=subject).subject_id == subject
