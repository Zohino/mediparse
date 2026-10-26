"""Rozhraní databáze EDA pro R a sešity: adresář z prostředí, nastavení jako texty."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mediparse.domain.data_access import DataAccessError
from mediparse.domain.inputs import InvalidInputError
from mediparse.entrypoints.eda_database import (
    connection_settings,
    database,
    parquet_dir,
)
from tests.eda_support import write_tables

if TYPE_CHECKING:
    from pathlib import Path


def test_parquet_dir_prefers_explicit_variable() -> None:
    """MEDIPARSE_PARQUET přebije kořen projektu."""
    environ = {"MEDIPARSE_PARQUET": "/data/p", "PIXI_PROJECT_ROOT": "/koren"}

    assert parquet_dir(environ) == "/data/p"


def test_parquet_dir_defaults_under_project_root() -> None:
    """Bez proměnné platí adresář pod kořenem projektu, i když je proměnná prázdná."""
    expected = "/koren/resources/mimic/parquet"

    assert parquet_dir({"PIXI_PROJECT_ROOT": "/koren"}) == expected
    empty = {"MEDIPARSE_PARQUET": "", "PIXI_PROJECT_ROOT": "/koren"}
    assert parquet_dir(empty) == expected


def test_parquet_dir_without_any_variable_is_refused() -> None:
    """Bez adresáře i kořene by vznikla relativní cesta, a proto chyba."""
    with pytest.raises(InvalidInputError, match="MEDIPARSE_PARQUET"):
        parquet_dir({})


def test_connection_settings_are_plain_text(tmp_path: Path) -> None:
    """Pro reticulate jen texty a slovník textů."""
    parquet = write_tables(tmp_path / "parquet")

    result = connection_settings({"MEDIPARSE_PARQUET": str(parquet)})

    assert result == {
        "memory_limit": "8GB",
        "temp_directory": str(tmp_path.resolve() / "duckdb-tmp"),
        "views": {
            "admissions": str(parquet / "admissions.parquet"),
            "discharge": str(parquet / "discharge.parquet"),
        },
    }


def test_connection_settings_apply_the_guard(tmp_path: Path) -> None:
    """Data bez značky pod CLAUDECODE: R nedostane nic."""
    parquet = write_tables(tmp_path / "parquet")
    environ = {"MEDIPARSE_PARQUET": str(parquet), "CLAUDECODE": "1"}

    with pytest.raises(DataAccessError):
        connection_settings(environ)


def test_database_uses_parquet_dir(tmp_path: Path) -> None:
    """Databáze čte z adresáře určeného prostředím."""
    parquet = write_tables(tmp_path / "parquet")
    environ = {"MEDIPARSE_PARQUET": str(parquet)}

    result = database(environ)

    assert result.parquet == parquet
    assert result.environ == environ
