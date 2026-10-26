"""Projektové pravidlo malých buněk: výstup nesmí zveřejnit skupinu pacientů pod prahem."""

from typing import Final

# Práh n < 10 platí pro buňky skupin pacientů vymezených kvaziidentifikátorem (věk, pohlaví,
# diagnóza, období...) a pro kategorie, jejichž popisek je hodnota z dat. R má vlastní kopii
# (DUA_THRESHOLD v eda/R/dua.R).
MIN_CELL: Final = 10
