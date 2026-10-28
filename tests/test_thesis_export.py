"""Export stejné tabulky do souboru pro práci a markdownu pro Quarto."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from matplotlib.figure import Figure

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.synthetic_parameters import SyntheticParameters
from mediparse.domain.thesis_table import ThesisTable
from mediparse.infrastructure import thesis_figure
from mediparse.infrastructure.synthetic_parameters_file import SyntheticParametersFile
from mediparse.infrastructure.thesis_export import MarkdownOutput, ThesisExport
from tests.support import export_payload
from tests.thesis_table_support import FIXTURES, golden_case

if TYPE_CHECKING:
    from pathlib import Path

CASE = golden_case("kody-dua")


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


def test_write_domain_table_matches_table_export(tmp_path: Path) -> None:
    """Hotová doménová tabulka má bajtově stejný export jako vstup z řádků."""
    table = ThesisTable(
        header=tuple(CASE["header"]),
        rows=tuple(tuple(row) for row in CASE["rows"]),
        align=CASE["align"],
        code=frozenset(CASE["code"]),
    )
    result = ThesisExport(tmp_path / "domain").write(
        table, label="tbl-a", caption="Popisek."
    )
    rows_result = ThesisExport(tmp_path / "rows").table(
        CASE["rows"],
        header=CASE["header"],
        align=CASE["align"],
        code=CASE["code"],
        label="tbl-a",
        caption="Popisek.",
    )
    assert result == rows_result
    assert (tmp_path / "domain" / "tbl-a.tex").read_bytes() == (
        FIXTURES / "kody-dua.tex"
    ).read_bytes()
    assert (tmp_path / "domain" / "tbl-a.tex").read_bytes() == (
        tmp_path / "rows" / "tbl-a.tex"
    ).read_bytes()


def test_figure_export_returns_same_figure(tmp_path: Path) -> None:
    """Export vytvoří adresář a vrátí tentýž objekt pro výstup buňky Quarta."""
    with thesis_figure.figure(6, font="DejaVu Serif") as (figure, axes):
        axes.set_title("Syntetický graf")
    directory = tmp_path / "a" / "b"
    assert (
        ThesisExport(directory).figure(figure, label="fig-pocty", font="DejaVu Serif")
        is figure
    )
    assert (directory / "fig-pocty.pdf").read_bytes().startswith(b"%PDF-")


@pytest.mark.parametrize(
    "label", ["tbl-a", "fig-", "../fig-a", "fig-A", "fig-a\n", "fig-a/b", "", 1]
)
def test_figure_label_is_validated_before_io(tmp_path: Path, label: str) -> None:
    """Neplatné id grafu nemůže vytvořit ani adresář."""
    directory = tmp_path / "export"
    with pytest.raises(InvalidInputError, match="label"):
        ThesisExport(directory).figure(Figure(), label=label)
    assert not directory.exists()


def _parameters(*, prevalence: float = 0.2) -> SyntheticParameters:
    return SyntheticParameters.model_validate(
        export_payload(("a", "b"), prevalence=prevalence), strict=False
    )


def test_json_roundtrip_and_overwrite(tmp_path: Path) -> None:
    """JSON parametrů načte existující čtenář a stejný label přepíše obsah."""
    directory = tmp_path / "a" / "b"
    model = _parameters()
    exporter = ThesisExport(directory)
    path = exporter.json(model, label="parametry-syntetiky")
    assert path == directory / "parametry-syntetiky.json"
    assert SyntheticParametersFile(path).load() == model
    updated = _parameters(prevalence=0.25)
    exporter.json(updated, label="parametry-syntetiky")
    assert SyntheticParametersFile(path).load() == updated


@pytest.mark.parametrize("label", ["../a", "a/b", "A", "a.json", "a\n", "", 1])
def test_json_label_is_validated_before_io(tmp_path: Path, label: str) -> None:
    """Neplatné jméno JSONu nevytvoří adresář ani soubor mimo něj."""
    directory = tmp_path / "export"
    model = _parameters()
    with pytest.raises(InvalidInputError, match="label"):
        ThesisExport(directory).json(model, label=label)
    assert not directory.exists()
