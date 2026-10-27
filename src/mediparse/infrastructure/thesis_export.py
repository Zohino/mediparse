"""Export tabulek, grafů a parametrů sešitu do adresáře výstupů práce."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.thesis_table import ThesisTable
from mediparse.infrastructure.json_file import write_model
from mediparse.infrastructure.thesis_figure import save_pdf
from mediparse.infrastructure.thesis_table_format import latex_tabular, table_markdown

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence
    from pathlib import Path

    from matplotlib.figure import Figure
    from pydantic import BaseModel


FIGURE_LABEL: Final = re.compile(r"^fig-[a-z0-9-]+$")
JSON_LABEL: Final = re.compile(r"^[a-z0-9-]+$")


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
        return self.write(table, label=label, caption=caption)

    def write(self, table: ThesisTable, *, label: str, caption: str) -> MarkdownOutput:
        """Zapíše hotovou doménovou tabulku pro práci a vrátí markdown pro sešit.

        Args:
            table: Textové buňky a jejich společné rozvržení.
            label: Id Quarta a jméno souboru bez přípony, začíná tbl-.
            caption: Neprázdný jednořádkový popisek.

        Returns:
            Zobrazitelný markdown stejné tabulky.

        Raises:
            InvalidInputError: Label nebo popisek není platný.
        """
        markdown = table_markdown(table, label=label, caption=caption)
        latex = latex_tabular(table) + "\n"
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / f"{label}.tex").write_bytes(latex.encode("utf-8"))
        return MarkdownOutput(markdown)

    def figure(self, figure: Figure, *, label: str) -> Figure:
        """Zapíše PDF grafu a vrátí stejnou figuru pro výstup buňky Quarta.

        Args:
            figure: Hotový graf se stylem a písmem práce.
            label: Id Quarta a jméno souboru bez přípony, začíná fig-.

        Returns:
            Tatáž figura, kterou Jupyter zobrazí v sešitu.

        Raises:
            InvalidInputError: Label není platný nebo graf nemá písmo práce.
        """
        if not isinstance(label, str) or not FIGURE_LABEL.fullmatch(label):
            msg = "label musí odpovídat ^fig-[a-z0-9-]+$."
            raise InvalidInputError(msg)
        self.directory.mkdir(parents=True, exist_ok=True)
        save_pdf(figure, self.directory / f"{label}.pdf")
        return figure

    def json(self, model: BaseModel, *, label: str) -> Path:
        """Zapíše model ve sdíleném formátu JSON a vrátí cestu k exportu.

        Args:
            model: Parametry určené k předání dalšímu kroku pipeline.
            label: Jméno souboru bez přípony, malá písmena, číslice a pomlčky.

        Returns:
            Cesta k zapsanému souboru JSON.

        Raises:
            InvalidInputError: Label není platný.
        """
        if not isinstance(label, str) or not JSON_LABEL.fullmatch(label):
            msg = "label musí odpovídat ^[a-z0-9-]+$."
            raise InvalidInputError(msg)
        path = self.directory / f"{label}.json"
        write_model(path, model)
        return path
