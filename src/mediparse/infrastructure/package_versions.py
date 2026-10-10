"""Verze nainstalovaných knihoven pro run manifest."""

from __future__ import annotations

import platform
from importlib.metadata import version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


def installed_versions(names: Iterable[str]) -> dict[str, str]:
    """Zjistí verze pojmenovaných balíčků; ``python`` je verze interpretu.

    Neznámý balíček je chyba programu a projde výjimkou ``PackageNotFoundError``.

    Returns:
        Název balíčku a jeho verze ve vstupním pořadí.
    """
    return {
        name: platform.python_version() if name == "python" else version(name)
        for name in names
    }
