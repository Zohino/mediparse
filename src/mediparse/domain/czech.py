"""Český zápis čísel v textu práce."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.domain.dua import CategoryCount

NBSP: Final = "\u00a0"
MISSING: Final = "\u2013"


def format_number(value: int) -> str:
    """Oddělí tisíce nezlomitelnou mezerou.

    Returns:
        Celé číslo připravené pro text práce.
    """
    return f"{value:,}".replace(",", NBSP)


def decimal_text(value: float) -> str:
    """Zapíše jedno desetinné místo s čárkou a NBSP mezi tisíci.

    Returns:
        Nativně zaokrouhlený desetinný zápis bez záporné nuly.
    """
    formatted = f"{value:,.1f}".replace(",", NBSP).replace(".", ",")
    return "0,0" if formatted == "-0,0" else formatted


def number_text(value: int | None) -> str:
    """Zapíše počet nebo pomlčku pro chybějící hodnotu.

    Returns:
        Počet s nezlomitelnými mezerami mezi tisíci, nebo MISSING.
    """
    return MISSING if value is None else format_number(value)


def percent_text(n: int, total: int) -> str:
    """Zapíše podíl z celku v procentech bez znaku procenta.

    Returns:
        Podíl s jedním desetinným místem, mez pro malý kladný podíl nebo pomlčka.
    """
    if total <= 0:
        return MISSING
    share = decimal_text(100 * n / total)
    if n > 0 and share == "0,0":
        return f"<{NBSP}0,1"
    return share


def percent_label(n: int, total: int) -> str:
    """Přidá k podílu nezlomitelnou mezeru a znak procenta.

    Returns:
        Podíl se znakem procenta, nebo samostatná pomlčka.
    """
    share = percent_text(n, total)
    return share if share == MISSING else f"{share}{NBSP}%"


def share_rows(rows: Sequence[CategoryCount]) -> tuple[tuple[str, ...], ...]:
    """Připraví všechny kategorie jako textové buňky bez potlačení.

    Returns:
        Popisky, počet a podíl z celkového počtu pro každý řádek.
    """
    total = sum(row.n for row in rows)
    return tuple(
        (*row.labels, number_text(row.n), percent_text(row.n, total)) for row in rows
    )


@dataclass(frozen=True)
class PluralForms:
    """České tvary po jedné, několika, mnoha a případně necelých jednotkách."""

    one: str
    few: str
    many: str
    fraction: str | None = None


def plural_form(n: float, forms: PluralForms) -> str:
    """Vybere český tvar podstatného jména podle číslovky.

    Returns:
        Tvar pro jednu, dvě až čtyři, ostatní celé nebo necelé jednotky.
    """
    if n != int(n):
        return forms.few if forms.fraction is None else forms.fraction
    if n == 1:
        return forms.one
    if n in {2, 3, 4}:
        return forms.few
    return forms.many


def number_phrase(n: float) -> str:
    """Zapíše číslo česky bez exponentu a s NBSP mezi tisíci.

    Returns:
        Celé číslo přesně, necelé alespoň se sedmi platnými číslicemi jako format z R.
        Celé číslice se nikdy nekrátí.
    """
    if n == int(n):
        return format_number(int(n))
    # R používá sedm platných číslic, celé číslice ale nikdy nekrátí.
    # Decimal z float by místo tohoto zápisu přidal binární artefakty.
    precision = max(7, len(str(int(abs(n)))))
    return (
        format(Decimal(f"{n:.{precision}g}"), ",f").replace(",", NBSP).replace(".", ",")
    )


def count_phrase(n: float, forms: PluralForms, *, zero: str | None = None) -> str:
    """Spojí počet s příslušným českým tvarem nezlomitelnou mezerou.

    Returns:
        Číslo a skloňovaný tvar, případně vlastní text pro nulu.
    """
    if n == 0 and zero is not None:
        return zero
    return f"{number_phrase(n)}{NBSP}{plural_form(n, forms)}"


def join_labels(labels: Sequence[str]) -> str:
    """Spojí výčet čárkami a závěrečnou spojkou a.

    Returns:
        Český výčet, nebo žádný pro prázdný seznam.
    """
    if not labels:
        return "žádný"
    if len(labels) == 1:
        return labels[0]
    return f"{', '.join(labels[:-1])} a {labels[-1]}"
