"""Český zápis čísel v textu práce."""

from typing import Final

NBSP: Final = "\u00a0"


def format_number(value: int) -> str:
    """Oddělí tisíce nezlomitelnou mezerou.

    Returns:
        Celé číslo připravené pro text práce.
    """
    return f"{value:,}".replace(",", NBSP)
