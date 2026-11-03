"""Párování a čtení veřejných paralelních syntetických textů."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from pathlib import Path


def read_parallel(english: Path, czech: Path) -> list[tuple[str, str, str]]:
    """Spáruje veřejné texty podle jména a zachová jejich konce řádků.

    Returns:
        Identifikátor páru a oba texty v pořadí jmen souborů.

    Raises:
        InvalidInputError: Chybí zprávy, protějšek nebo platný UTF-8 obsah.
    """
    en = {path.name: path for path in english.glob("*.txt")}
    cs = {path.name: path for path in czech.glob("*.txt")}
    if not en and not cs:
        msg = f"V adresářích {english} a {czech} nejsou žádné zprávy (*.txt)."
        raise InvalidInputError(msg)
    missing = sorted(en.keys() ^ cs.keys())
    if missing:
        msg = f"Zpráva nemá protějšek: {', '.join(missing)}."
        raise InvalidInputError(msg)
    return [
        (en[name].stem, _read_text(en[name]), _read_text(cs[name]))
        for name in sorted(en)
    ]


def _read_text(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8")
    except (OSError, UnicodeError) as error:
        msg = f"Syntetický pár {path.name} nelze přečíst: {type(error).__name__}."
        raise InvalidInputError(msg) from None
