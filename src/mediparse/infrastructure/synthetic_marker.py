"""Značka ``synthetic.json`` v adresáři syntetických tabulek: podle ní se pozná syntetika."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict

from mediparse.infrastructure.input_file import parse_file

if TYPE_CHECKING:
    from pathlib import Path

MARKER_NAME: Final = "synthetic.json"


class SyntheticMarker(BaseModel):
    """Prefix buněk a počet řádků syntetických tabulek."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prefix: str
    rows: int


def write_marker(directory: Path, marker: SyntheticMarker) -> None:
    """Zapíše značku do adresáře syntetických tabulek."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / MARKER_NAME).write_text(marker.model_dump_json(), encoding="utf-8")


def read_marker(directory: Path) -> SyntheticMarker | None:
    """Přečte značku adresáře.

    Returns:
        Značka, nebo ``None``, když adresář syntetické tabulky neobsahuje.
    """
    path = directory / MARKER_NAME
    if not path.exists():
        return None
    return parse_file(path, SyntheticMarker.model_validate_json)
