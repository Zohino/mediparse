"""Připojení DuckDB pro EDA: pohledy nad parquetem, nastavení a stráž proti čtení skutečných dat."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import pytest

from mediparse.domain.data_access import DataAccessError
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure import eda_duckdb
from mediparse.infrastructure.eda_duckdb import DEFAULT_MEMORY, EdaDatabase
from mediparse.infrastructure.synthetic_marker import MARKER_NAME
from tests.eda_support import mark_synthetic, write_tables

if TYPE_CHECKING:
    from collections.abc import Iterator

CLAUDE_CODE = {"CLAUDECODE": "1"}
DEFAULT_MEMORY_SHOWN = "7.4 GiB"


def test_views_are_sorted_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pohled nese jméno souboru bez přípony, v pořadí jmen."""
    parquet = write_tables(tmp_path / "parquet", ("zeta", "mimic-iv-bhc", "alfa"))
    glob = Path.glob

    def reversed_glob(self: Path, pattern: str) -> Iterator[Path]:
        return iter(sorted(glob(self, pattern), reverse=True))

    # Pořadí souborů ve výpisu adresáře není zaručené, a proto ho test vynutí opačné.
    monkeypatch.setattr(Path, "glob", reversed_glob)

    views = EdaDatabase(parquet, {}).guarded_views()

    assert list(views) == ["alfa", "mimic-iv-bhc", "zeta"]
    assert views["alfa"] == parquet / "alfa.parquet"


def test_repr_hides_environment(tmp_path: Path) -> None:
    """Prostředí nese tokeny, a proto se nevypisuje do logu ani do výstupu dokumentu."""
    database = EdaDatabase(tmp_path, {"API_TOKEN": "hodnota-z-prostredi"})

    assert "hodnota-z-prostredi" not in repr(database)


def test_invalid_memory_names_variable_and_value(tmp_path: Path) -> None:
    """Procenta DuckDB nebere; hláška jmenuje proměnnou, hodnotu a typ výjimky."""
    parquet = write_tables(tmp_path / "parquet")
    environ = {"MEDIPARSE_DUCKDB_MEMORY": "50%"}

    with (
        pytest.raises(InvalidInputError) as refusal,
        EdaDatabase(parquet, environ).connect(),
    ):
        pytest.fail("spojení nemělo vzniknout")

    message = str(refusal.value)
    assert "MEDIPARSE_DUCKDB_MEMORY" in message
    assert "50%" in message
    assert refusal.value.__cause__ is None


def test_connect_creates_view_for_every_parquet(tmp_path: Path) -> None:
    """Pohled je i pro jméno s pomlčkou a čte řádky souboru."""
    parquet = write_tables(tmp_path / "parquet", ("discharge", "mimic-iv-bhc"))

    with EdaDatabase(parquet, {}).connect() as con:
        counts = [
            con.table(name).count("*").fetchone()
            for name in ("discharge", "mimic-iv-bhc")
        ]

    assert counts == [(1,), (1,)]


def test_connect_closes_the_connection(tmp_path: Path) -> None:
    """Po výstupu z kontextu je spojení zavřené."""
    parquet = write_tables(tmp_path / "parquet")

    with EdaDatabase(parquet, {}).connect() as con:
        pass

    with pytest.raises(duckdb.ConnectionException):
        con.sql("SELECT 1")


def test_default_memory(tmp_path: Path) -> None:
    """Bez proměnné i s prázdnou proměnnou platí výchozí paměť."""
    parquet = write_tables(tmp_path / "parquet")

    assert DEFAULT_MEMORY == "8GB"
    assert EdaDatabase(parquet, {}).settings().memory_limit == DEFAULT_MEMORY
    empty = {"MEDIPARSE_DUCKDB_MEMORY": ""}
    assert EdaDatabase(parquet, empty).settings().memory_limit == DEFAULT_MEMORY


def test_memory_from_environment(tmp_path: Path) -> None:
    """Proměnná prostředí přebije výchozí paměť."""
    parquet = write_tables(tmp_path / "parquet")
    environ = {"MEDIPARSE_DUCKDB_MEMORY": "3GiB"}

    assert EdaDatabase(parquet, environ).settings().memory_limit == "3GiB"


def test_spill_is_next_to_parquet_directory(tmp_path: Path) -> None:
    """Spill leží vedle adresáře parquetu, i když adresář je symlink."""
    real = tmp_path / "real"
    parquet = write_tables(real / "parquet")
    link = tmp_path / "link"
    link.symlink_to(parquet, target_is_directory=True)

    settings = EdaDatabase(link, {}).settings()

    assert settings.temp_directory == real.resolve() / "duckdb-tmp"


def test_connect_applies_settings(tmp_path: Path) -> None:
    """Spojení má paměť a spill z nastavení."""
    parquet = write_tables(tmp_path / "parquet")
    database = EdaDatabase(parquet, {})

    with database.connect() as con:
        memory = con.sql("SELECT current_setting('memory_limit')").fetchone()
        temp = con.sql("SELECT current_setting('temp_directory')").fetchone()

    assert memory == (DEFAULT_MEMORY_SHOWN,)
    assert temp == (str(database.settings().temp_directory),)


def test_missing_directory_is_refused(tmp_path: Path) -> None:
    """Chybějící adresář se hlásí jako neexistující."""
    with pytest.raises(InvalidInputError, match="neexistuje"):
        EdaDatabase(tmp_path / "none", {}).guarded_views()


def test_directory_without_parquet_is_refused_before_guard(tmp_path: Path) -> None:
    """Adresář bez parquetu se odmítne dřív než stráží, i pod CLAUDECODE."""
    with pytest.raises(InvalidInputError, match="žádný parquet") as refusal:
        EdaDatabase(tmp_path, CLAUDE_CODE).guarded_views()

    assert not isinstance(refusal.value, DataAccessError)


def test_real_data_are_refused_in_claude_code(tmp_path: Path) -> None:
    """Data bez značky pod CLAUDECODE: spojení nevznikne."""
    parquet = write_tables(tmp_path / "parquet")

    with (
        pytest.raises(DataAccessError, match="Claude Code"),
        EdaDatabase(parquet, CLAUDE_CODE).connect(),
    ):
        pytest.fail("spojení nemělo vzniknout")

    assert not (tmp_path / "duckdb-tmp").exists()


def test_synthetic_data_pass_in_claude_code(tmp_path: Path) -> None:
    """Se značkou syntetiky stráž pustí."""
    parquet = mark_synthetic(write_tables(tmp_path / "parquet"))

    with EdaDatabase(parquet, {"CI": "true", **CLAUDE_CODE}).connect() as con:
        assert con.sql("SELECT count(*) FROM discharge").fetchone() == (1,)


def test_invalid_marker_is_refused(tmp_path: Path) -> None:
    """Prázdná značka není syntetika: neplatný soubor je chyba, ne průchod."""
    parquet = write_tables(tmp_path / "parquet")
    (parquet / MARKER_NAME).write_text("{}", encoding="utf-8")

    with pytest.raises(InvalidInputError, match=r"synthetic\.json") as refusal:
        EdaDatabase(parquet, CLAUDE_CODE).guarded_views()

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


def test_record_counts_are_sorted_python_ints(tmp_path: Path) -> None:
    """Počty všech pohledů mají deterministické pořadí i jméno s pomlčkou."""
    parquet = mark_synthetic(
        write_tables(tmp_path / "parquet", ("zeta", "mimic-iv-bhc", "alfa"))
    )
    result = EdaDatabase(parquet, CLAUDE_CODE).record_counts()
    assert result == {"alfa": 1, "mimic-iv-bhc": 1, "zeta": 1}
    assert list(result) == ["alfa", "mimic-iv-bhc", "zeta"]
    assert all(type(value) is int for value in result.values())


def test_record_counts_guard_precedes_parquet_read(tmp_path: Path) -> None:
    """I poškozený soubor bez značky nejprve odmítne stráž."""
    (tmp_path / "broken.parquet").write_text("CELL_CANARY", encoding="utf-8")
    with pytest.raises(DataAccessError):
        EdaDatabase(tmp_path, CLAUDE_CODE).record_counts()


def test_record_counts_refuses_corrupt_parquet(tmp_path: Path) -> None:
    """Poškozený parquet nesmí dostat text chyby DuckDB do sešitu."""
    mark_synthetic(tmp_path)
    (tmp_path / "broken.parquet").write_text("CELL_CANARY", encoding="utf-8")
    with pytest.raises(InvalidInputError, match="broken") as refusal:
        EdaDatabase(tmp_path, CLAUDE_CODE).record_counts()
    assert "CELL_CANARY" not in str(refusal.value)
    assert refusal.value.__cause__ is None
    assert refusal.value.__suppress_context__


def test_record_counts_sanitizes_query_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chyba samotného count po vytvoření pohledu neprozradí text výjimky."""
    parquet = mark_synthetic(write_tables(tmp_path / "parquet", ("discharge",)))

    def fail_count(*_args: object, **_kwargs: object) -> None:
        msg = "CELL_CANARY"
        raise duckdb.InvalidInputException(msg)

    monkeypatch.setattr(duckdb.DuckDBPyRelation, "count", fail_count)
    with pytest.raises(
        InvalidInputError, match=r"discharge.*InvalidInputException"
    ) as refusal:
        EdaDatabase(parquet, CLAUDE_CODE).record_counts()
    assert "CELL_CANARY" not in str(refusal.value)
    assert refusal.value.__cause__ is None
    assert refusal.value.__suppress_context__


def test_record_counts_use_the_guarded_view_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Další pohled v katalogu DuckDB nezmění počty spravovaných parquetů."""
    parquet = mark_synthetic(write_tables(tmp_path / "parquet"))
    original_views = EdaDatabase.guarded_views
    original_create = eda_duckdb.create_parquet_view
    lookups: list[Path] = []

    def guarded_views(self: EdaDatabase) -> dict[str, Path]:
        lookups.append(self.parquet)
        return original_views(self)

    def create_with_foreign_view(
        con: duckdb.DuckDBPyConnection, name: str, path: Path
    ) -> None:
        original_create(con, name, path)
        con.sql("CREATE OR REPLACE VIEW unrelated AS SELECT 1")

    monkeypatch.setattr(EdaDatabase, "guarded_views", guarded_views)
    monkeypatch.setattr(eda_duckdb, "create_parquet_view", create_with_foreign_view)
    result = EdaDatabase(parquet, CLAUDE_CODE).record_counts()
    assert result == {"admissions": 1, "discharge": 1}
    assert list(result) == ["admissions", "discharge"]
    assert lookups == [parquet]
