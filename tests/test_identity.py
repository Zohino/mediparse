"""Identita řádku matice: skladba, kterou vydává matice a přijímá trénink."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from mediparse.domain.identity import RowId


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
