"""Zápis tabulky pro práci a markdownový výstup pro Jupyter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.thesis_table import ThesisTable
from mediparse.infrastructure.thesis_table_format import latex_tabular, table_markdown

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence
    from pathlib import Path


@dataclass(frozen=True)
class MarkdownOutput:
    """Markdown zobrazitelný v sešitu bez závislosti na knihovně jádra."""

    text: str

    def _repr_markdown_(self) -> str:  # ruff: ignore[bad-dunder-method-name] - zobrazovací protokol Jupyteru
        """Vrátí obsah pro zobrazovací protokol Jupyteru.

        Returns:
            Markdown určený k vykreslení v buňce.
        """
        return self.text


@dataclass(frozen=True)
class ThesisExport:
    """Exportér fragmentů pro práci a stejného obsahu pro Quarto."""

    directory: Path

    def table(  # ruff: ignore[too-many-arguments] - explicitní volby jediné tabulky v sešitu
        self,
        rows: Sequence[Sequence[str]],
        *,
        header: Sequence[str],
        align: str,
        label: str,
        caption: str,
        code: Collection[str] = (),
        widths: Sequence[float] = (),
    ) -> MarkdownOutput:
        """Zapíše tabulku do UTF-8 LaTeXu a vrátí markdown s popiskem.

        Args:
            rows: Textové buňky v pořadí řádků.
            header: Hlavička; její názvy používá i code.
            align: Jeden znak l, c nebo r na sloupec.
            label: Id Quarta a jméno souboru bez přípony, začíná tbl-.
            caption: Neprázdný jednořádkový popisek.
            code: Hlavičky sloupců s kódovými hodnotami.
            widths: Volitelné kladné šířky v procentech.

        Returns:
            Zobrazitelný markdown stejné tabulky pro sešit.

        Raises:
            InvalidInputError: Vstup není platná tabulka, label nebo popisek.
        """
        if isinstance(header, str) or isinstance(code, str):
            msg = "header a code musí být kolekce, nikoli samotný text."
            raise InvalidInputError(msg)
        if any(isinstance(row, str) for row in rows):
            msg = "Řádek tabulky musí být kolekce buněk, nikoli samotný text."
            raise InvalidInputError(msg)
        table = ThesisTable(
            header=tuple(header),
            rows=tuple(tuple(row) for row in rows),
            align=align,
            widths=tuple(widths),
            code=frozenset(code),
        )
        markdown = table_markdown(table, label=label, caption=caption)
        latex = latex_tabular(table) + "\n"
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / f"{label}.tex").write_bytes(latex.encode("utf-8"))
        return MarkdownOutput(markdown)
