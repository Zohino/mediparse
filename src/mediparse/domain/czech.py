"""Český zápis čísel v textu práce."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.domain.dua import CategoryCount

NBSP: Final = "\u00a0"
MISSING: Final = "\u2013"
_R_MAX_DIGITS: Final = 15


def format_number(value: int) -> str:
    """Oddělí tisíce nezlomitelnou mezerou.

    Returns:
        Celé číslo připravené pro text práce.
    """
    return f"{value:,}".replace(",", NBSP)


def decimal_text(value: float) -> str:
    """Zapíše jedno desetinné místo s čárkou a bez oddělovače tisíců.

    Returns:
        Desetinný zápis zaokrouhlený jako round z R.
    """
    rounded = _round_like_r(value)
    return f"{0.0 if rounded == 0 else rounded:.1f}".replace(".", ",")


# Parita výstupů EDA během migrace vyžaduje round(x, 1) z R >= 4.0.
# Nativní formát i Decimal zaokrouhlují přesnou binární hodnotu jinak (0.05 -> 0.1).
# Po odstranění R kopií se k nativnímu zaokrouhlení vrátí S20h13.
def _round_like_r(value: float) -> float:
    """Vybere bližší desetinný kandidát v aritmetice double jako R.

    Returns:
        Zaokrouhlená hodnota, při stejných vzdálenostech se sudou poslední číslicí.
    """
    if value == 0 or not math.isfinite(value):
        return value
    magnitude = abs(value)
    if math.log10(magnitude) + 1 > _R_MAX_DIGITS:
        return value
    lower_index = math.floor(magnitude * 10)
    lower = lower_index / 10
    upper = math.ceil(magnitude * 10) / 10
    distance_down = magnitude - lower
    distance_up = upper - magnitude
    choose_upper = distance_up < distance_down or (
        distance_up == distance_down and lower_index % 2 == 1
    )
    return math.copysign(upper if choose_upper else lower, value)


def number_text(value: int | None) -> str:
    """Zapíše počet nebo pomlčku pro chybějící hodnotu.

    Returns:
        Počet s nezlomitelnými mezerami mezi tisíci, nebo MISSING.
    """
    return MISSING if value is None else format_number(value)


def percent_text(n: int, total: int) -> str:
    """Zapíše podíl z celku v procentech bez znaku procenta.

    Returns:
        Podíl s jedním desetinným místem, mez pro malý kladný podíl nebo pomlčka.
    """
    if total <= 0:
        return MISSING
    share = decimal_text(100 * n / total)
    if n > 0 and share == "0,0":
        return f"<{NBSP}0,1"
    return share


def percent_label(n: int, total: int) -> str:
    """Přidá k podílu nezlomitelnou mezeru a znak procenta.

    Returns:
        Podíl se znakem procenta, nebo samostatná pomlčka.
    """
    share = percent_text(n, total)
    return share if share == MISSING else f"{share}{NBSP}%"


def share_rows(rows: Sequence[CategoryCount]) -> tuple[tuple[str, ...], ...]:
    """Připraví všechny kategorie jako textové buňky bez potlačení.

    Returns:
        Popisky, počet a podíl z celkového počtu pro každý řádek.
    """
    total = sum(row.n for row in rows)
    return tuple(
        (*row.labels, number_text(row.n), percent_text(row.n, total)) for row in rows
    )
