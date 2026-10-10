"""Syntetické řádky tabulky ve tvaru MIMIC: každá buňka je jedinečný řetězec s prefixem."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from mediparse.domain.mimic_table import TableShape


def synthetic_rows(
    table: str, shape: TableShape, prefix: str, rows: int
) -> list[dict[str, str]]:
    """Vytvoří řádky se sloupci v pořadí tabulky a buňkami ``prefix-tabulka-sloupec-řádek``.

    Každá buňka je jedinečná, takže každá kategorie v syntetické tabulce má n = 1
    a únik jakékoli hodnoty do výstupu je nalezitelný hledáním prefixu.

    Args:
        table: Jméno tabulky bez přípony.
        shape: Sloupce tabulky v pořadí souboru.
        prefix: Řetězec, kterým začíná každá buňka.
        rows: Počet řádků.

    Returns:
        Řádky jako slovníky sloupec na buňku.

    Raises:
        InvalidInputError: Prefix je prázdný, počet řádků není kladný nebo tabulka nemá sloupce.
    """
    if not prefix:
        msg = "Prefix syntetických buněk nesmí být prázdný."
        raise InvalidInputError(msg)
    if rows < 1:
        msg = (
            f"Syntetická tabulka {table} potřebuje aspoň jeden řádek, ne {rows} řádků."
        )
        raise InvalidInputError(msg)
    if not shape.columns:
        msg = f"Tabulka {table} nemá žádné sloupce."
        raise InvalidInputError(msg)
    return [
        {column: f"{prefix}-{table}-{column}-{row}" for column in shape.columns}
        for row in range(rows)
    ]
