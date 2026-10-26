"""Export stejné tabulky do souboru pro práci a markdownu pro Quarto."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.thesis_export import MarkdownOutput, ThesisExport

FIXTURES = Path(__file__).parent / "fixtures" / "thesis_table"
CASE = json.loads((FIXTURES / "cases.json").read_text(encoding="utf-8"))[0]


def test_export_writes_golden_and_returns_markdown(tmp_path: Path) -> None:
    """Jediné volání uloží LaTeX a vrátí zobrazitelný markdown se stejnými řádky."""
    directory = tmp_path / "a" / "b"
    result = ThesisExport(directory).table(
        CASE["rows"],
        header=CASE["header"],
        align=CASE["align"],
        code=CASE["code"],
        label="tbl-a",
        caption="Popisek.",
    )
    assert (directory / "tbl-a.tex").read_bytes() == (
        FIXTURES / "kody-dua.tex"
    ).read_bytes()
    assert isinstance(result, MarkdownOutput)
    assert result._repr_markdown_() == (
        (FIXTURES / "kody-dua.md").read_bytes().decode() + "\n\n: Popisek. {#tbl-a}"
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"label": "../tbl-a"},
        {"caption": ""},
        {"align": "ll"},
        {"rows": ((None,),)},
        {"widths": (False,)},
    ],
)
def test_invalid_export_does_not_create_directory(
    tmp_path: Path, overrides: dict
) -> None:
    """Veškerá validace předchází vytvoření adresáře a zápisu souboru."""
    directory = tmp_path / "a" / "b"
    arguments: dict = {
        "rows": (("x",),),
        "header": ("A",),
        "align": "l",
        "label": "tbl-a",
        "caption": "Popisek.",
    }
    with pytest.raises(InvalidInputError):
        ThesisExport(directory).table(**(arguments | overrides))
    assert not directory.exists()


def test_repeated_label_overwrites_file(tmp_path: Path) -> None:
    """Nový export téhož labelu nahradí původní obsah."""
    exporter = ThesisExport(tmp_path)
    arguments: dict = {
        "header": ("A",),
        "align": "l",
        "label": "tbl-a",
        "caption": "Popisek.",
    }
    exporter.table((("původní",),), **arguments)
    exporter.table((("nový",),), **arguments)
    tex = (tmp_path / "tbl-a.tex").read_text(encoding="utf-8")
    assert "nový" in tex
    assert "původní" not in tex


@pytest.mark.parametrize(
    "overrides", [{"header": "A"}, {"rows": ("x",)}, {"rows": "x"}, {"code": "A"}]
)
def test_export_rejects_strings_instead_of_collections(
    tmp_path: Path, overrides: dict
) -> None:
    """Holý text se nesmí rozpadnout na znaky hlavičky, řádku ani code."""
    directory = tmp_path / "export"
    arguments: dict = {
        "rows": (("x",),),
        "header": ("A",),
        "align": "l",
        "label": "tbl-a",
        "caption": "Popisek.",
        **overrides,
    }
    with pytest.raises(InvalidInputError):
        ThesisExport(directory).table(**arguments)
    assert not directory.exists()
