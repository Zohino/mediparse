"""Společné minimum obou tříd pro stabilitu kohorty a metrik."""

from typing import Final

# Dostatek pozitivních i negativních zpráv zajišťuje stabilitu splitu a metrik.
MIN_CELL: Final = 10


def below_min_cell(positives: int, total: int) -> bool:
    """Určí, zda některá třída nemá dost zpráv pro stabilní metriky.

    Args:
        positives: Počet pozitivních zpráv.
        total: Počet všech zpráv v obou třídách.

    Returns:
        Zda počet pozitivních nebo negativních zpráv nedosahuje minima.
    """
    return min(positives, total - positives) < MIN_CELL
