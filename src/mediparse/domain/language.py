"""Jazyk vstupních zpráv."""

from enum import StrEnum


class Language(StrEnum):
    """Jazyk vstupních zpráv; čeština vzniká výhradně strojovým překladem."""

    EN = "en"
    CS = "cs"
