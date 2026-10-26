"""Adresář exportu tabulek odvozený z explicitního prostředí Quarta."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.entrypoints.eda_export import export

if TYPE_CHECKING:
    from pathlib import Path


def test_document_export_directory(tmp_path: Path) -> None:
    """Kmen dokumentu a rodič adresáře EDA určují místo exportu."""
    result = export({
        "QUARTO_DOCUMENT_PATH": str(tmp_path / "eda"),
        "QUARTO_DOCUMENT_FILE": "1-dataset.qmd",
    })
    assert result.directory == tmp_path / "results" / "eda" / "1-dataset"
    assert not result.directory.exists()


@pytest.mark.parametrize("variable", ["QUARTO_DOCUMENT_PATH", "QUARTO_DOCUMENT_FILE"])
@pytest.mark.parametrize("value", [None, ""])
def test_missing_quarto_variable(variable: str, value: str | None) -> None:
    """Chybějící i prázdná proměnná se odmítne s uvedením jejího jména."""
    environ = {"QUARTO_DOCUMENT_PATH": "eda", "QUARTO_DOCUMENT_FILE": "a.qmd"}
    if value is None:
        del environ[variable]
    else:
        environ[variable] = value
    with pytest.raises(InvalidInputError, match=variable):
        export(environ)
