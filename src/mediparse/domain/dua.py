"""Projektové pravidlo potlačení malých buněk skupin pacientů a kategorií z dat."""

from typing import Final

# Práh n < 10 platí pro buňky skupin pacientů vymezených kvaziidentifikátorem
# a pro kategorie, jejichž popisek je hodnota z dat. Zrcadlo DUA_THRESHOLD
# v eda/R/dua.R; změna se dělá v obou souborech.
MIN_CELL: Final = 10
