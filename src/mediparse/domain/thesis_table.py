"""Validovaná textová tabulka a její rozvržení."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from mediparse.domain.inputs import InvalidInputError


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
