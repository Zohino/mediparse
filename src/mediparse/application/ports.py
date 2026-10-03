"""Porty, které používá víc use casů."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mediparse.domain.note_plan import NotePlan


class PlanSource(Protocol):
    """Plány zpráv korpusu."""

    def load(self) -> tuple[NotePlan, ...]:
        """Načte plány.

        Returns:
            Plány v pořadí zdroje.
        """


class NoteSource(Protocol):
    """Texty zpráv korpusu."""

    def notes(self) -> Mapping[str, str]:
        """Texty zpráv.

        Returns:
            Slovník relativní cesta ``<jazyk>/<note_id>.txt`` → text zprávy.
        """
