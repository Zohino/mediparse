"""Pravidlo přístupu k datům MIMIC: v jakém prostředí je smí krok číst."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Mapping

# Pořadí určuje, který důvod se nahlásí první, když platí obě proměnné.
BLOCKING_VARIABLES: Final[Mapping[str, str]] = MappingProxyType({
    "CI": "Veřejné CI nesmí číst data MIMIC.",
    "CLAUDECODE": (
        "Relace Claude Code nesmí číst data MIMIC: výstup by odešel hostovanému modelu."
    ),
})


class DataAccessError(InvalidInputError):
    """Prostředí nesmí číst skutečná data MIMIC."""


def blocked_reason(environ: Mapping[str, str]) -> str | None:
    """Najde první neprázdnou blokující proměnnou.

    Args:
        environ: Proměnné prostředí.

    Returns:
        Důvod zákazu čtení MIMIC, nebo ``None``, když žádná blokující proměnná neplatí.
    """
    return next(
        (reason for name, reason in BLOCKING_VARIABLES.items() if environ.get(name)),
        None,
    )


def check_data_access(
    environ: Mapping[str, str], *, synthetic: bool, location: str, marker: str
) -> None:
    """Odmítne čtení skutečných dat v prostředí, které je nesmí číst.

    Args:
        environ: Proměnné prostředí.
        synthetic: Zda data nesou platnou značku syntetiky.
        location: Místo dat pro hlášku.
        marker: Jméno souboru se značkou syntetiky pro hlášku.

    Raises:
        DataAccessError: Data nejsou syntetická a prostředí je blokuje.
    """
    if synthetic:
        return
    reason = blocked_reason(environ)
    if reason is not None:
        msg = (
            f"Adresář {location} neobsahuje {marker}, "
            f"takže jde o skutečná data. {reason}"
        )
        raise DataAccessError(msg)
