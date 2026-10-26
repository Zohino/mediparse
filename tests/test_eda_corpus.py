"""První sešit přes entrypoint, syntetický parquet a veřejný manifest v tmp_path."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from mediparse.domain.data_access import DataAccessError
from mediparse.entrypoints.eda_corpus import corpus_overview
from mediparse.infrastructure.parquet_file import write_string_table
from tests.eda_support import mark_synthetic

if TYPE_CHECKING:
    from pathlib import Path


def _fixture(tmp_path: Path, *, agrees: bool) -> dict[str, str]:
    parquet = tmp_path / "parquet"
    counts = {"discharge": 3, "radiology": 2, "mimic-iv-bhc": 2}
    tables = []
    for name, count in counts.items():
        write_string_table(
            parquet / f"{name}.parquet", ["a"], [{"a": "syntetika"}] * count
        )
        tables.append({
            "file": f"{name}.csv",
            "records": count if agrees else 10,
            "columns": ["a"],
            "bytes": 2726465013 if name == "mimic-iv-bhc" else 2500000,
        })
    mark_synthetic(parquet)
    manifest = tmp_path / "results" / "mimic" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"tables": tables}), encoding="utf-8")
    return {
        "PIXI_PROJECT_ROOT": str(tmp_path),
        "MEDIPARSE_PARQUET": str(parquet),
        "CLAUDECODE": "1",
    }


@pytest.mark.parametrize("agrees", [True, False])
def test_corpus_overview(tmp_path: Path, *, agrees: bool) -> None:
    """Počty z DuckDB, MB z manifestu a text úplnosti tvoří jeden přehled."""
    environ = _fixture(tmp_path, agrees=agrees)
    result = corpus_overview(environ)
    assert result.table.rows == (
        (
            "discharge",
            "MIMIC-IV-Note (note)",
            "Propouštěcí zprávy; primární korpus práce.",
            "3",
            "1",
            "2",
        ),
        (
            "mimic-iv-bhc",
            "MIMIC-IV-Ext-BHC",
            "Brief Hospital Course vytažený regexem autorů datasetu; reference validace parseru sekcí.",
            "2",
            "1",
            "2\u00a0726",
        ),
        (
            "radiology",
            "MIMIC-IV-Note (note)",
            "Radiologické zprávy; doplňkový korpus.",
            "2",
            "1",
            "2",
        ),
    )
    verb = "odpovídají" if agrees else "neodpovídají"
    assert (
        result.records
        == f"Nejvíce záznamů má tabulka discharge. Počty záznamů v databázi {verb} počtům v manifestu validace."
    )
    assert (result.tables, result.discharge, result.radiology) == ("3", "3", "2")
    if agrees:
        assert result.summary.startswith("Počty záznamů u všech tabulek odpovídají")
    else:
        assert "discharge, mimic-iv-bhc, radiology" in result.summary


def test_corpus_overview_uses_guard(tmp_path: Path) -> None:
    """Bez značky syntetiky se nečtou data ani manifest."""
    environ = _fixture(tmp_path, agrees=True)
    (tmp_path / "parquet" / "synthetic.json").unlink()
    (tmp_path / "results" / "mimic" / "manifest.json").unlink()
    with pytest.raises(DataAccessError):
        corpus_overview(environ)
