"""Přehled korpusu bez databáze a renderovacího enginu."""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING

import pytest

from mediparse.domain import corpus_overview as overview_module
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.mimic_table import TableShape

if TYPE_CHECKING:
    from collections.abc import Mapping

CATALOG_SIZE = 13


def _shapes(counts: Mapping[str, int]) -> dict[str, TableShape]:
    return {
        name: TableShape(records=count, columns=("a", "b"))
        for name, count in counts.items()
    }


def test_table_layout_order_and_megabytes() -> None:
    """Rozvržení i čísla patří do domény; MB nepřetečou a poloviny jdou k sudé."""
    counts = {
        "radiology": 1139000,
        "mimic-iv-bhc": 2,
        "discharge": 1000,
        "admissions": 5,
    }
    sizes = {
        "radiology": 1139000000,
        "mimic-iv-bhc": 2726465013,
        "discharge": 2600000,
        "admissions": 2500000,
    }
    result = overview_module.summarize_corpus(counts, _shapes(counts), sizes)
    table = result.table
    assert table.header == (
        "Tabulka",
        "Zdroj",
        "Obsah a využití",
        "Záznamů",
        "Sloupců",
        "MB",
    )
    assert table.align == "lllrrr"
    assert table.widths == (22, 16, 30, 10, 12, 10)
    assert table.code == frozenset({"Tabulka"})
    assert [row[0] for row in table.rows] == [
        "admissions",
        "discharge",
        "mimic-iv-bhc",
        "radiology",
    ]
    assert [row[3:] for row in table.rows] == [
        ("5", "2", "2"),
        ("1\u00a0000", "2", "3"),
        ("2", "2", "2\u00a0726"),
        ("1\u00a0139\u00a0000", "2", "1\u00a0139"),
    ]
    assert (result.tables, result.discharge, result.radiology) == (
        "4",
        "1\u00a0000",
        "1\u00a0139\u00a0000",
    )


@pytest.mark.parametrize("agrees", [True, False])
@pytest.mark.parametrize("tie", [True, False])
def test_sentences(*, agrees: bool, tie: bool) -> None:
    """Všechny inline hodnoty jsou neprázdné i bez jednoznačného maxima."""
    counts = {"radiology": 3 if tie else 2, "discharge": 3}
    shapes = _shapes(counts if agrees else {"radiology": 10, "discharge": 10})
    result = overview_module.summarize_corpus(counts, shapes, dict.fromkeys(counts, 0))
    prefix = "" if tie else "Nejvíce záznamů má tabulka discharge. "
    verb = "odpovídají" if agrees else "neodpovídají"
    assert (
        result.records
        == prefix + f"Počty záznamů v databázi {verb} počtům v manifestu validace."
    )
    if agrees:
        assert result.summary == (
            "Počty záznamů u všech tabulek odpovídají manifestu validace. "
            "Data jsou proto úplná a další části analýzy mohou stavět na parquetu bez další kontroly počtu záznamů."
        )
    else:
        assert result.summary == (
            "Počty záznamů se liší od manifestu validace u tabulek: discharge, radiology. "
            "Data se proto nesmějí považovat za úplná a před dalším použitím je nutné rozdíl vysvětlit."
        )
    texts = [
        getattr(result, field.name) for field in fields(result) if field.name != "table"
    ]
    texts.extend(cell for row in result.table.rows for cell in row)
    assert all(isinstance(text, str) and text and "shod" not in text for text in texts)


@pytest.mark.parametrize("missing", ["catalog", "shapes", "sizes"])
def test_unknown_table_is_refused(missing: str) -> None:
    """Chyba pojmenuje tabulku mimo katalog nebo neúplný manifest."""
    name = "unknown" if missing == "catalog" else "discharge"
    counts = {name: 1}
    shapes = {} if missing == "shapes" else _shapes(counts)
    sizes = {} if missing == "sizes" else {name: 1}
    with pytest.raises(InvalidInputError, match=name):
        overview_module.summarize_corpus(counts, shapes, sizes)


def test_catalog() -> None:
    """Katalog přenáší třináct jedinečných tabulek a tři zdroje z R."""
    catalog = overview_module.CATALOG
    assert len(catalog) == len({entry.table for entry in catalog}) == CATALOG_SIZE
    assert all(entry.description for entry in catalog)
    assert {entry.source for entry in catalog} == {
        "MIMIC-IV (hosp)",
        "MIMIC-IV-Note (note)",
        "MIMIC-IV-Ext-BHC",
    }
    assert (
        next(entry.source for entry in catalog if entry.table == "mimic-iv-bhc")
        == "MIMIC-IV-Ext-BHC"
    )
