"""Syntetické řádky tabulky ve tvaru MIMIC: buňky jsou jedinečné řetězce s prefixem."""

from __future__ import annotations

import math
from string import ascii_lowercase
from typing import TYPE_CHECKING, Final

from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from mediparse.domain.mimic_table import TableShape

TEXT_COLUMN: Final = "text"

# Klíč patří tabulce, která entitu v MIMIC zakládá. Řádek r každé tabulky tak patří pacientovi
# a hospitalizaci r a spojení přes klíč dá neprázdný vstupní korpus. Klíč není kategorie
# (v tabulce je jedinečný), takže každá kategorie má dál n = 1 a buňka klíče nese prefix
# s pomlčkou, který kanárek najde.
_KEY_OWNERS: Final = {"subject_id": "patients", "hadm_id": "admissions"}
# note_id je v každé tabulce zpráv vlastní (propouštěcí a radiologické se v MIMIC nepotkají),
# jen MIMIC-IV-Ext-BHC odkazuje na propouštěcí zprávu.
_REFERENCES: Final = {("mimic-iv-bhc", "note_id"): "discharge"}


def synthetic_rows(
    table: str, shape: TableShape, prefix: str, rows: int, *, text_chars: int
) -> list[dict[str, str]]:
    """Vytvoří řádky se sloupci v pořadí tabulky a buňkami ``prefix-vlastník-sloupec-řádek``.

    Vlastníkem je tabulka sama, u klíčů pacienta a hospitalizace tabulka, která je zakládá,
    a u ``note_id`` z MIMIC-IV-Ext-BHC tabulka ``discharge``. V tabulce je každá buňka
    jedinečná, takže každá kategorie má n = 1 a únik jakékoli hodnoty do výstupu je
    nalezitelný hledáním prefixu. Sloupec ``text`` za buňkou opakuje token zprávy, dokud
    nemá aspoň ``text_chars`` znaků (viz ``_text_token``).

    Args:
        table: Jméno tabulky bez přípony.
        shape: Sloupce tabulky v pořadí souboru.
        prefix: Řetězec, kterým začíná každá buňka.
        rows: Počet řádků.
        text_chars: Nejmenší délka textu, aby zpráva prošla filtrem délky vstupního korpusu.

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
        {
            column: _cell(table, column, prefix, row, text_chars)
            for column in shape.columns
        }
        for row in range(rows)
    ]


def _text_token(prefix: str, row: int) -> str:
    """Vrátí token zprávy: písmena prefixu a číslo řádku zapsané písmeny.

    Tokenizace zahodí pomlčky a normalizace přepíše číslice, takže buňku s pomlčkami
    ve výstupech po tokenech (n-gramy, slovník) kanárek nenajde. Token ze samých písmen
    oběma kroky projde beze změny a v každé zprávě je jiný (DF = 1). Výstup, ve kterém se
    objeví, proto pustil token jediné zprávy.

    Returns:
        Token ze samých písmen.
    """
    stem = "".join(character for character in prefix if character.isalpha())
    return stem + _letters(row)


def _cell(table: str, column: str, prefix: str, row: int, text_chars: int) -> str:
    owner = _REFERENCES.get((table, column)) or _KEY_OWNERS.get(column, table)
    cell = f"{prefix}-{owner}-{column}-{row}"
    if column != TEXT_COLUMN:
        return cell
    token = _text_token(prefix, row)
    copies = max(1, math.ceil((text_chars - len(cell)) / (len(token) + 1)))
    return " ".join([cell, *[token] * copies])


def _letters(number: int) -> str:
    """Zapíše nezáporné číslo písmeny jako číslice soustavy o 26 znacích.

    Returns:
        Číslo zapsané písmeny, například 0 → a, 25 → z, 26 → ba.
    """
    digits = ""
    while True:
        number, digit = divmod(number, len(ascii_lowercase))
        digits = ascii_lowercase[digit] + digits
        if number == 0:
            return digits
