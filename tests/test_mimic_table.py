"""Jméno tabulky MIMIC odvozené ze jména staženého souboru."""

from __future__ import annotations

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.mimic_table import table_name


@pytest.mark.parametrize(
    ("file_name", "expected"),
    [
        pytest.param("discharge.csv.gz", "discharge", id="gzip"),
        pytest.param("mimic-iv-bhc.csv", "mimic-iv-bhc", id="prosté-csv"),
    ],
)
def test_table_name_drops_csv_suffix(file_name: str, expected: str) -> None:
    """Jméno tabulky je jméno souboru bez přípony CSV, komprimovaného i prostého."""
    assert table_name(file_name) == expected


@pytest.mark.parametrize("file_name", ["notes.zip", "discharge.csv.bz2", "csv.gz"])
def test_unsupported_file_is_refused(file_name: str) -> None:
    """Soubor, který řetězec stažení a převodu neumí přečíst, je chyba vstupu."""
    with pytest.raises(InvalidInputError, match=file_name.replace(".", r"\.")):
        table_name(file_name)
