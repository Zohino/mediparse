"""Projektové pravidlo malých buněk: výstup nesmí zveřejnit skupinu pacientů pod prahem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Sequence

# Práh n < 10 platí pro buňky skupin pacientů vymezených kvaziidentifikátorem (věk, pohlaví,
# diagnóza, období...) a pro kategorie, jejichž popisek je hodnota z dat.
# eda/R/dua.R a eda/R/czech.R mají přechodné kopie celé sady pomocníků do S20h12.
MIN_CELL: Final = 10

OTHER_LABEL: Final[str] = f"ostatní (n < {MIN_CELL})"
UNKNOWN_LABEL: Final = "neuvedeno"


@dataclass(frozen=True)
class CategoryCount:
    """Agregovaný počet s textovými popisky kategorie."""

    labels: tuple[str, ...]
    n: int

    def __post_init__(self) -> None:
        """Odmítne neplatný počet nebo popisky.

        Raises:
            InvalidInputError: Počet není nezáporné int nebo chybí textové popisky.
        """
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 0:
            msg = "n musí být nezáporné celé číslo typu int."
            raise InvalidInputError(msg)
        if not self.labels or any(not isinstance(label, str) for label in self.labels):
            msg = "Popisky musí být neprázdná n-tice textů."
            raise InvalidInputError(msg)


def hidden_cells(counts: Sequence[int]) -> tuple[bool, ...]:
    """Určí skryté buňky včetně ochrany dopočítatelného součtu.

    Returns:
        Maska skrytých buněk v původním pořadí.
    """
    hidden = [n < MIN_CELL for n in counts]
    total = sum(n for n, mask in zip(counts, hidden, strict=True) if mask)
    if total == 0 or total >= MIN_CELL:
        return tuple(hidden)
    kept = [index for index, mask in enumerate(hidden) if not mask]
    for index in sorted(kept, key=lambda index: counts[index]):
        hidden[index] = True
        total += counts[index]
        if total >= MIN_CELL:
            break
    return tuple(hidden)


def suppress_small_cells(
    rows: Sequence[CategoryCount], *, group: int | None = None
) -> tuple[CategoryCount, ...]:
    """Sloučí skryté kategorie uvnitř každé skupiny.

    Args:
        rows: Agregované řádky se stejným počtem popisků.
        group: Index popisku skupiny, nebo None pro celou tabulku.

    Returns:
        Skupiny podle prvního výskytu, v každé zachované řádky a případně ostatní.

    Raises:
        InvalidInputError: Počty popisků se liší nebo group neleží v jejich rozsahu.
    """
    if not rows:
        return ()
    width = len(rows[0].labels)
    if any(len(row.labels) != width for row in rows):
        msg = "Řádky musí mít stejný počet popisků."
        raise InvalidInputError(msg)
    if group is not None and not 0 <= group < width:
        msg = "group musí být index popisku tabulky."
        raise InvalidInputError(msg)
    groups: dict[str | None, list[CategoryCount]] = {}
    for row in rows:
        key = None if group is None else row.labels[group]
        groups.setdefault(key, []).append(row)
    return tuple(
        row for members in groups.values() for row in _suppress_group(members, group)
    )


def _suppress_group(
    rows: Sequence[CategoryCount], group: int | None
) -> tuple[CategoryCount, ...]:
    """Potlačí jednu skupinu bez změny jejího popisku.

    Returns:
        Zachované řádky a případně řádek ostatní.
    """
    hidden = hidden_cells([row.n for row in rows])
    kept = tuple(row for row, mask in zip(rows, hidden, strict=True) if not mask)
    if not any(hidden):
        return kept
    labels = tuple(
        label if index == group else OTHER_LABEL
        for index, label in enumerate(rows[0].labels)
    )
    total = sum(row.n for row, mask in zip(rows, hidden, strict=True) if mask)
    return (*kept, CategoryCount(labels, total))


def binary_count(n: int | None, total: int | None) -> int | None:
    """Ochrání malý počet i jeho dopočítatelný doplněk.

    Returns:
        Zveřejnitelný počet, jinak None.
    """
    if n is None or total is None or min(n, total - n) < MIN_CELL:
        return None
    return n
