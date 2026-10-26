"""Text syntetické propouštěcí zprávy: token zprávy a (dál) sekce a target BHC."""

from __future__ import annotations

from string import ascii_lowercase


def text_token(prefix: str, row: int) -> str:
    """Vrátí token zprávy: písmena prefixu a číslo řádku zapsané písmeny.

    Tokenizace zahodí pomlčky a n-gramy EDA berou jen písmena (``[a-z]{2,}``), takže buňku
    s pomlčkami a číslem řádku ve výstupech po tokenech (n-gramy, slovník) kanárek nenajde.
    Token ze samých písmen projde beze změny a v každé zprávě je jiný (DF = 1); s číslicí by
    se v n-gramech zkrátil na kmen, který mají všechny zprávy. Výstup, ve kterém se token
    objeví, proto pustil token jediné zprávy.

    Returns:
        Token ze samých písmen.
    """
    stem = "".join(character for character in prefix if character.isalpha())
    return stem + _letters(row)


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
