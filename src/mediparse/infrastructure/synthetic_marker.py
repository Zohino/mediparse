"""Značka adresáře s daty, která nepocházejí z MIMIC.

Potvrzuje syntetické tabulky i artefakty z nich odvozené; ``rows`` udává
počet syntetických zpráv zdroje.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict

from mediparse.infrastructure.input_file import parse_file
from mediparse.infrastructure.json_file import write_model

if TYPE_CHECKING:
    from pathlib import Path

MARKER_NAME: Final = "synthetic.json"


class SyntheticMarker(BaseModel):
    """Syntetický původ adresáře tabulek nebo odvozených artefaktů.

    ``prefix`` je prefix buněk, nebo ``None``, pokud zdroj prefix nepoužívá.
    ``rows`` je počet syntetických zpráv zdroje.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    prefix: str | None
    rows: int


def write_marker(directory: Path, marker: SyntheticMarker) -> None:
    """Zapíše značku původu do adresáře syntetických dat."""
    write_model(directory / MARKER_NAME, marker)


def read_marker(directory: Path) -> SyntheticMarker | None:
    """Přečte značku adresáře.

    Returns:
        Značka, nebo ``None``, pokud soubor značky chybí.
    """
    path = directory / MARKER_NAME
    if not path.exists():
        return None
    return parse_file(path, SyntheticMarker.model_validate_json)
