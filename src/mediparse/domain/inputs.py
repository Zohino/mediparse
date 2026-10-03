"""Vstupy kroků: chyba vstupu, který chybí nebo neodpovídá schématu."""


class InvalidInputError(ValueError):
    """Vstup kroku chybí nebo neodpovídá schématu; zpráva jmenuje, který.

    Na rozdíl od neplatného záznamu auditu nebo provenance, který use case hlásí
    jako porušení, je chybný vstup důvod, proč krok vůbec nemůže běžet.
    """
