"""Use case převodu tabulky MIMIC na parquet: výsledek musí mít tvar inventáře validace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from mediparse.domain.mimic_table import TableShape


class ShapeSource(Protocol):
    """Zdroj tvaru tabulky zjištěného validací CSV."""

    def read(self) -> TableShape:
        """Přečte tvar tabulky."""


class TableConverter(Protocol):
    """Převod tabulky, který vrací tvar toho, co zapsal."""

    def convert(self) -> TableShape:
        """Převede tabulku a vrátí tvar výsledku."""


@dataclass(frozen=True)
class MimicParquetConversion:
    """Převod tabulky ověřený proti inventáři: počet záznamů i sloupce se musí shodovat."""

    inventory: ShapeSource
    converter: TableConverter

    def run(self) -> TableShape:
        """Převede tabulku a porovná výsledek s inventářem.

        Returns:
            Tvar převedené tabulky.

        Raises:
            InvalidInputError: Počet záznamů nebo sloupce se liší od inventáře.
        """
        expected = self.inventory.read()
        actual = self.converter.convert()
        if actual.records != expected.records:
            msg = (
                f"Převod zapsal {actual.records} záznamů, inventář uvádí "
                f"{expected.records}."
            )
            raise InvalidInputError(msg)
        if actual.columns != expected.columns:
            msg = (
                f"Hlavička souboru má sloupce {', '.join(actual.columns)}, "
                f"inventář uvádí {', '.join(expected.columns)}."
            )
            raise InvalidInputError(msg)
        return actual
