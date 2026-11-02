"""Sdílené porty proudového čtení korpusu."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class NoteTexts(Protocol):
    """Zdroj textů celé tabulky v pořadí souboru."""

    def batches(self) -> Iterable[Sequence[tuple[str, str]]]:
        """Vrátí dávky dvojic identifikátor a text."""
        ...
