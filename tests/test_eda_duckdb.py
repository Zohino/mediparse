"""Připojení DuckDB pro EDA: pohledy nad parquetem, nastavení a stráž proti čtení skutečných dat."""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
import pytest

from mediparse.domain.data_access import DataAccessError
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.eda_duckdb import DEFAULT_MEMORY, EdaDatabase
from mediparse.infrastructure.parquet_file import write_string_table
from mediparse.infrastructure.synthetic_marker import (
    MARKER_NAME,
    SyntheticMarker,
    write_marker,
)

if TYPE_CHECKING:
    from pathlib import Path

CLAUDE_CODE = {"CLAUDECODE": "1"}
DEFAULT_MEMORY_SHOWN = "7.4 GiB"


def _write_tables(
    directory: Path, names: tuple[str, ...] = ("discharge", "admissions")
) -> Path:
    for name in names:
        write_string_table(directory / f"{name}.parquet", ["a"], [{"a": "x"}])
    return directory


def _synthetic(directory: Path) -> Path:
    write_marker(directory, SyntheticMarker(prefix="t", rows=1))
    return directory


def test_views_are_sorted_by_name(tmp_path: Path) -> None:
    """Pohled nese jméno souboru bez přípony, v pořadí jmen."""
    parquet = _write_tables(tmp_path / "parquet", ("zeta", "mimic-iv-bhc", "alfa"))

    views = EdaDatabase(parquet, {}).views()

    assert list(views) == ["alfa", "mimic-iv-bhc", "zeta"]
    assert views["alfa"] == parquet / "alfa.parquet"


def test_connect_creates_view_for_every_parquet(tmp_path: Path) -> None:
    """Pohled je i pro jméno s pomlčkou a čte řádky souboru."""
    parquet = _write_tables(tmp_path / "parquet", ("discharge", "mimic-iv-bhc"))

    with EdaDatabase(parquet, {}).connect() as con:
        counts = [
            con.table(name).count("*").fetchone()
            for name in ("discharge", "mimic-iv-bhc")
        ]

    assert counts == [(1,), (1,)]


def test_connect_closes_the_connection(tmp_path: Path) -> None:
    """Po výstupu z kontextu je spojení zavřené."""
    parquet = _write_tables(tmp_path / "parquet")

    with EdaDatabase(parquet, {}).connect() as con:
        pass

    with pytest.raises(duckdb.ConnectionException):
        con.sql("SELECT 1")


def test_default_memory(tmp_path: Path) -> None:
    """Bez proměnné i s prázdnou proměnnou platí výchozí paměť."""
    parquet = _write_tables(tmp_path / "parquet")

    assert DEFAULT_MEMORY == "8GB"
    assert EdaDatabase(parquet, {}).settings().memory_limit == DEFAULT_MEMORY
    empty = {"MEDIPARSE_DUCKDB_MEMORY": ""}
    assert EdaDatabase(parquet, empty).settings().memory_limit == DEFAULT_MEMORY


def test_memory_from_environment(tmp_path: Path) -> None:
    """Proměnná prostředí přebije výchozí paměť."""
    parquet = _write_tables(tmp_path / "parquet")
    environ = {"MEDIPARSE_DUCKDB_MEMORY": "3GiB"}

    assert EdaDatabase(parquet, environ).settings().memory_limit == "3GiB"


def test_spill_is_next_to_parquet_directory(tmp_path: Path) -> None:
    """Spill leží vedle adresáře parquetu, i když adresář je symlink."""
    real = tmp_path / "real"
    parquet = _write_tables(real / "parquet")
    link = tmp_path / "link"
    link.symlink_to(parquet, target_is_directory=True)

    settings = EdaDatabase(link, {}).settings()

    assert settings.temp_directory == real.resolve() / "duckdb-tmp"


def test_connect_applies_settings(tmp_path: Path) -> None:
    """Spojení má paměť a spill z nastavení."""
    parquet = _write_tables(tmp_path / "parquet")
    database = EdaDatabase(parquet, {})

    with database.connect() as con:
        memory = con.sql("SELECT current_setting('memory_limit')").fetchone()
        temp = con.sql("SELECT current_setting('temp_directory')").fetchone()

    assert memory == (DEFAULT_MEMORY_SHOWN,)
    assert temp == (str(database.settings().temp_directory),)


def test_missing_directory_is_refused(tmp_path: Path) -> None:
    """Chybějící adresář se hlásí jako neexistující."""
    with pytest.raises(InvalidInputError, match="neexistuje"):
        EdaDatabase(tmp_path / "none", {}).views()


def test_directory_without_parquet_is_refused_before_guard(tmp_path: Path) -> None:
    """Adresář bez parquetu se odmítne dřív než stráží, i pod CLAUDECODE."""
    with pytest.raises(InvalidInputError, match="žádný parquet") as refusal:
        EdaDatabase(tmp_path, CLAUDE_CODE).check_access()

    assert not isinstance(refusal.value, DataAccessError)


def test_real_data_are_refused_in_claude_code(tmp_path: Path) -> None:
    """Data bez značky pod CLAUDECODE: spojení nevznikne."""
    parquet = _write_tables(tmp_path / "parquet")

    with (
        pytest.raises(DataAccessError, match="Claude Code"),
        EdaDatabase(parquet, CLAUDE_CODE).connect(),
    ):
        pytest.fail("spojení nemělo vzniknout")

    assert not (tmp_path / "duckdb-tmp").exists()


def test_synthetic_data_pass_in_claude_code(tmp_path: Path) -> None:
    """Se značkou syntetiky stráž pustí."""
    parquet = _synthetic(_write_tables(tmp_path / "parquet"))

    with EdaDatabase(parquet, {"CI": "true", **CLAUDE_CODE}).connect() as con:
        assert con.sql("SELECT count(*) FROM discharge").fetchone() == (1,)


def test_invalid_marker_is_refused(tmp_path: Path) -> None:
    """Prázdná značka není syntetika: neplatný soubor je chyba, ne průchod."""
    parquet = _write_tables(tmp_path / "parquet")
    (parquet / MARKER_NAME).write_text("{}", encoding="utf-8")

    with pytest.raises(InvalidInputError) as refusal:
        EdaDatabase(parquet, CLAUDE_CODE).check_access()

    assert not isinstance(refusal.value, DataAccessError)


def test_unreadable_parquet_is_refused_without_error_text(tmp_path: Path) -> None:
    """Text výjimky DuckDB nese hodnoty buněk, proto hláška jmenuje jen soubor a typ."""
    parquet = tmp_path / "parquet"
    parquet.mkdir()
    (parquet / "broken.parquet").write_text("CELL_CANARY", encoding="utf-8")

    with (
        pytest.raises(InvalidInputError) as refusal,
        EdaDatabase(parquet, {}).connect(),
    ):
        pytest.fail("spojení nemělo vzniknout")

    assert "broken.parquet" in str(refusal.value)
    assert "CELL_CANARY" not in str(refusal.value)
    assert refusal.value.__cause__ is None
    assert refusal.value.__suppress_context__
