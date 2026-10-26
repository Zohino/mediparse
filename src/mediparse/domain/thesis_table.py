"""Textové tabulky pro práci a Quarto s bajtovým formátem knitr::kable."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Final

from mediparse.domain.dua import OTHER_LABEL
from mediparse.domain.inputs import InvalidInputError

TABLE_LABEL: Final = re.compile(r"^tbl-[a-z0-9-]+$")
KABLE_WHITESPACE: Final = re.compile(r"^\s+|(?<!\\)\s+$", re.ASCII)
LATEX_SPECIALS: Final = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
}
LATEX_ALIGNMENT: Final = {
    "l": r"\raggedright",
    "c": r"\centering",
    "r": r"\raggedleft",
}


@dataclass(frozen=True)
class ThesisTable:
    """Hlavička, textové řádky a rozvržení tabulky pro oba výstupy."""

    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    align: str
    widths: tuple[float, ...] = ()
    code: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        """Odmítne neplatné buňky, rozvržení a neznámé kódové sloupce.

        Raises:
            InvalidInputError: Tabulka obsahuje neplatný vstup.
        """
        if not self.header:
            msg = "hlavička musí obsahovat alespoň jeden sloupec."
            raise InvalidInputError(msg)
        _check_cells(self.header, 0)
        for index, row in enumerate(self.rows, start=1):
            if len(row) != len(self.header):
                msg = f"řádek {index}: počet buněk neodpovídá hlavičce."
                raise InvalidInputError(msg)
            _check_cells(row, index)
        if (
            not isinstance(self.align, str)
            or not re.fullmatch(r"[lcr]+", self.align)
            or len(self.align) != len(self.header)
        ):
            msg = "align musí mít jeden znak l, c nebo r na sloupec."
            raise InvalidInputError(msg)
        _check_widths(self.widths, len(self.header))
        if not self.code.issubset(self.header):
            msg = "code musí jmenovat sloupce hlavičky."
            raise InvalidInputError(msg)


def _check_cells(cells: tuple[str, ...], row: int) -> None:
    for column, cell in enumerate(cells, start=1):
        if not isinstance(cell, str) or "\n" in cell or "\r" in cell:
            msg = f"řádek {row}, sloupec {column}: očekáván text bez konce řádku."
            raise InvalidInputError(msg)


def _check_widths(widths: tuple[float, ...], columns: int) -> None:
    if not widths:
        return
    if len(widths) != columns or any(
        isinstance(width, bool)
        or not isinstance(width, (int, float))
        or not math.isfinite(width)
        or width <= 0
        for width in widths
    ):
        msg = "widths musí být kladná konečná čísla, jedno na sloupec."
        raise InvalidInputError(msg)


def latex_escape(text: str) -> str:
    """Escapuje speciální znaky LaTeXu jediným průchodem.

    Returns:
        Text s escapovanými znaky; NBSP zůstává.
    """
    return "".join(LATEX_SPECIALS.get(char, char) for char in text)


def _column_spec(table: ThesisTable) -> str:
    if not table.widths:
        return table.align
    # R format() používá sedm platných číslic a píše 1, nikoli 1.0.
    return "".join(
        f">{{{LATEX_ALIGNMENT[side]}\\arraybackslash}}p{{\\dimexpr {width / 100:.7g}\\linewidth-2\\tabcolsep\\relax}}"
        for side, width in zip(table.align, table.widths, strict=True)
    )


def _latex_row(cells: tuple[str, ...]) -> str:
    # Kable s booktabs chrání počáteční [ před výkladem jako argument \\.
    protected = (re.sub(r"^([ \t\r\n]*)(\[)", r"\1{}\2", cell) for cell in cells)
    return " & ".join(protected) + r"\\"


def _latex_cells(table: ThesisTable, row: tuple[str, ...]) -> tuple[str, ...]:
    cells = []
    for header, raw in zip(table.header, row, strict=True):
        text = latex_escape(raw)
        if header in table.code and raw != OTHER_LABEL:
            text = f"\\texttt{{{text}}}"
        cells.append(_trim_cell(text))
    return tuple(cells)


def _trim_cell(text: str) -> str:
    # Kable ořezává PCRE \s (ASCII včetně \v a \f), koncovou mezeru za \ nechá; NBSP je obsah.
    return KABLE_WHITESPACE.sub("", text)


def latex_tabular(table: ThesisTable) -> str:
    """Sestaví booktabs tabular bez floatu a koncového newline.

    Returns:
        Fragment bajtově odpovídající R exportu před writeLines.
    """
    lines = [
        f"\\begin{{tabular}}{{{_column_spec(table)}}}",
        r"\toprule",
        _latex_row(tuple(latex_escape(cell) for cell in table.header)),
    ]
    if table.rows:
        lines.extend([
            r"\midrule",
            *(_latex_row(_latex_cells(table, row)) for row in table.rows),
        ])
    else:
        # Kable pro prázdnou tabulku vynechá midrule a vloží dva prázdné řádky.
        # Tento tvar zachovává paritu exportů syntetického sešitu 7.
        lines.extend(["", ""])
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _markdown_cells(table: ThesisTable, row: tuple[str, ...]) -> tuple[str, ...]:
    cells = []
    for header, raw in zip(table.header, row, strict=True):
        text = f"`{raw}`" if header in table.code and raw != OTHER_LABEL else raw
        cells.append(_trim_cell(text).replace("|", "&#124;"))
    return tuple(cells)


def _pad_cell(text: str, width: int, side: str) -> str:
    if side == "l":
        return text.ljust(width)
    if side == "r":
        return text.rjust(width)
    left = (width - len(text)) // 2
    return " " * left + text + " " * (width - len(text) - left)


def _pipe_row(cells: tuple[str, ...], widths: tuple[int, ...], align: str) -> str:
    return (
        "|"
        + "|".join(
            _pad_cell(cell, width, side)
            for cell, width, side in zip(cells, widths, align, strict=True)
        )
        + "|"
    )


def pipe_table(table: ThesisTable) -> str:
    """Sestaví pipe tabulku se zarovnáním a paddingem kable.

    Returns:
        Řádky markdownové tabulky bez koncového newline.
    """
    header = tuple(cell.replace("|", "&#124;") for cell in table.header)
    rows = tuple(_markdown_cells(table, row) for row in table.rows)
    # Bez tbl-colwidths Pandoc u dlouhých tabulek odvozuje šířky z pomlček;
    # padding proto musí odpovídat kable i kvůli vzhledu PDF.
    widths = tuple(
        max(
            max(len(row[index]) for row in (header, *rows)) + (2 if side == "c" else 1),
            3,
        )
        for index, side in enumerate(table.align)
    )
    separator = tuple(
        (":" if side in "lc" else "-")
        + "-" * (width - 2)
        + (":" if side in "rc" else "-")
        for width, side in zip(widths, table.align, strict=True)
    )
    return "\n".join([
        _pipe_row(header, widths, table.align),
        "|" + "|".join(separator) + "|",
        *(_pipe_row(row, widths, table.align) for row in rows),
    ])


def table_markdown(table: ThesisTable, *, label: str, caption: str) -> str:
    """Doplní tabulku o popisek, id a volitelné šířky pro Quarto.

    Returns:
        Markdown tabulky včetně řádku popisku.

    Raises:
        InvalidInputError: Label nebo popisek není platný.
    """
    if not isinstance(label, str) or not TABLE_LABEL.fullmatch(label):
        msg = "label musí odpovídat ^tbl-[a-z0-9-]+$."
        raise InvalidInputError(msg)
    if (
        not isinstance(caption, str)
        or not caption.strip()
        or "\n" in caption
        or "\r" in caption
    ):
        msg = "popisek musí být neprázdný text bez konce řádku."
        raise InvalidInputError(msg)
    attribute = ""
    if table.widths:
        widths = ",".join(f"{width:g}" for width in table.widths)
        attribute = f' tbl-colwidths="[{widths}]"'
    return pipe_table(table) + f"\n\n: {caption} {{#{label}{attribute}}}"
