"""Čtení řádků General Equivalence Mappings; jen pro testy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, NamedTuple

from mediparse.domain.icd import IcdCode, IcdVersion

if TYPE_CHECKING:
    from pathlib import Path

NO_DIAGNOSIS: Final = "NoDx"
_COMBINATION_FLAG: Final = 2
_FIELDS: Final = 3


class GemRow(NamedTuple):
    """Jeden řádek GEMs: zdroj, cíl (None pro NoDx) a příznak kombinačního záznamu."""

    source: IcdCode
    target: IcdCode | None
    combination: bool


def read_gem_rows(
    path: Path, source_version: IcdVersion, target_version: IcdVersion
) -> list[GemRow]:
    """Přečte soubor GEMs ve tvaru `zdroj cíl příznaky` oddělený mezerami.

    Returns:
        Řádky souboru v pořadí, v jakém jsou v něm.

    Raises:
        ValueError: Řádek nemá tři pole.
    """
    rows = []
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) != _FIELDS:
            msg = f"Řádek GEMs nemá {_FIELDS} pole: {line!r}."
            raise ValueError(msg)
        source, target, flags = fields
        rows.append(
            GemRow(
                source=IcdCode(version=source_version, code=source),
                target=None
                if target == NO_DIAGNOSIS
                else IcdCode(version=target_version, code=target),
                combination=flags[_COMBINATION_FLAG] != "0",
            )
        )
    return rows
