"""Vstupy kroků: chyba, kvůli které krok odmítá běžet."""


class InvalidInputError(ValueError):
    """Krok odmítá běžet: vstup, config nebo data nesplňují jeho předpoklad.

    Zpráva jmenuje, co předpoklad porušuje. Na rozdíl od neplatného záznamu
    auditu nebo provenance, který use case hlásí jako porušení, je to důvod,
    proč krok vůbec nemůže běžet.
    """
