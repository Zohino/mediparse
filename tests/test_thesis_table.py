"""Bajtová parita tabulek pro práci s R kable a validace vstupů."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mediparse.domain.dua import OTHER_LABEL
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.thesis_table import ThesisTable
from mediparse.infrastructure.thesis_table_format import (
    latex_escape,
    latex_tabular,
    pipe_table,
    table_markdown,
)
from tests.thesis_table_support import CASES, FIXTURES, golden_case

if TYPE_CHECKING:
    from collections.abc import Mapping


def _table(case: Mapping) -> ThesisTable:
    return ThesisTable(
        header=tuple(case["header"]),
        rows=tuple(tuple(row) for row in case["rows"]),
        align=case["align"],
        widths=tuple(case["widths"]),
        code=frozenset(case["code"]),
    )


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_golden_table(case: Mapping) -> None:
    """LaTeX včetně koncového newline i markdown mají přesné bajty R exportu."""
    table = _table(case)
    assert (latex_tabular(table) + "\n").encode() == (
        FIXTURES / f"{case['name']}.tex"
    ).read_bytes()
    assert pipe_table(table).encode() == (FIXTURES / f"{case['name']}.md").read_bytes()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
        ("<", r"\textless{}"),
        (">", r"\textgreater{}"),
        ("\u00a0*[…]", "\u00a0*[…]"),
        (r"\&", r"\textbackslash{}\&"),
    ],
)
def test_latex_escape(text: str, expected: str) -> None:
    """Escapování projde původní text jednou a zachová NBSP i ostatní znaky."""
    assert latex_escape(text) == expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"align": ""},
        {"align": "ll"},
        {"align": "x"},
        {"widths": (50, 50)},
        {"widths": (0,)},
        {"widths": (-1,)},
        {"widths": (float("nan"),)},
        {"widths": (float("inf"),)},
        {"widths": (True,)},
        {"widths": ("auto",)},
        {"rows": ((1,),)},
        {"rows": ((None,),)},
        {"rows": (("x", "y"),)},
        {"rows": (("x\ny",),)},
        {"rows": (("x\ry",),)},
        {"header": ("A\nB",)},
        {"header": ("A\rB",)},
        {"header": (None,)},
        {"header": (1,)},
        {"code": frozenset({"B"})},
    ],
)
def test_invalid_table(overrides: dict) -> None:
    """Nesoulad rozvržení a netextové nebo víceřádkové buňky se odmítnou."""
    arguments: dict = {"header": ("A",), "rows": (("x",),), "align": "l", **overrides}
    with pytest.raises(InvalidInputError):
        ThesisTable(**arguments)


def test_invalid_cell_message_contains_position_without_value() -> None:
    """Chyba ukáže souřadnice, ale neodhalí obsah buňky."""
    with pytest.raises(InvalidInputError, match=r"řádek 1.*sloupec 1") as error:
        ThesisTable(header=("A",), rows=(("tajná\nhodnota",),), align="l")
    assert "tajná" not in str(error.value)


@pytest.mark.parametrize(
    ("widths", "attribute"),
    [
        ((), ""),
        ((22, 60, 18), ' tbl-colwidths="[22,60,18]"'),
        ((12.5, 70, 17.5), ' tbl-colwidths="[12.5,70,17.5]"'),
        ((70.0, 30.0), ' tbl-colwidths="[70,30]"'),
    ],
)
def test_table_caption(widths: tuple[float, ...], attribute: str) -> None:
    """Popisek nese jediné id tabulky a volitelné šířky."""
    columns = len(widths) or 3
    table = ThesisTable(
        header=tuple("ABC"[:columns]), rows=(), align="lcr"[:columns], widths=widths
    )
    assert table_markdown(table, label="tbl-a", caption="Popisek.") == (
        pipe_table(table) + f"\n\n: Popisek. {{#tbl-a{attribute}}}"
    )


@pytest.mark.parametrize("label", ["Tbl_A", "a", "tbl-", "fig-a", "tbl-a_b", "tbl-a\n"])
def test_invalid_label(label: str) -> None:
    """Label je bezpečné jméno souboru i id křížového odkazu."""
    with pytest.raises(InvalidInputError, match="label"):
        table_markdown(_table(golden_case("kody-dua")), label=label, caption="Popisek.")


@pytest.mark.parametrize("caption", ["", " ", "\n", "Více\nřádků", "Více\rřádků"])
def test_invalid_caption(caption: str) -> None:
    """Popisek musí být neprázdný na jednom řádku."""
    with pytest.raises(InvalidInputError, match="popisek"):
        table_markdown(_table(golden_case("kody-dua")), label="tbl-a", caption=caption)


def test_other_label() -> None:
    """Popisek sloučených kategorií odpovídá R DUA_OTHER."""
    assert OTHER_LABEL == "ostatní (n < 10)"


@pytest.mark.parametrize(
    ("align", "header"),
    [
        (("l", "r"), ("A", "B")),
        (("",), ("A",)),
        (("lc",), ("A",)),
        (None, ("A",)),
    ],
)
def test_alignment_must_be_a_string(align: object, header: tuple[str, ...]) -> None:
    """Rozvržení nelze předat jako kolekci dílčích řetězců nebo None."""
    arguments: dict = {"header": header, "rows": (), "align": align}
    with pytest.raises(InvalidInputError, match="align"):
        ThesisTable(**arguments)


def test_header_must_not_be_empty() -> None:
    """Tabulka bez sloupců nesmí vytvořit neplatný tabular."""
    with pytest.raises(InvalidInputError, match="hlavička"):
        ThesisTable(header=(), rows=(), align="")


@pytest.mark.parametrize(
    "overrides", [{"label": None}, {"label": 1}, {"caption": None}, {"caption": 1}]
)
def test_label_and_caption_must_be_strings(overrides: dict) -> None:
    """Neplatný runtime typ se přeloží na doménovou chybu."""
    arguments: dict = {"label": "tbl-a", "caption": "Popisek.", **overrides}
    with pytest.raises(InvalidInputError):
        table_markdown(_table(golden_case("kody-dua")), **arguments)
