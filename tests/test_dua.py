"""Potlačení malých buněk nad syntetickými počty kategorií."""

from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mediparse.domain import dua
from mediparse.domain.inputs import InvalidInputError


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ([100, 200, 50], (False, False, False)),
        ([0, 100, 200], (True, False, False)),
        ([100, 4, 50], (False, True, True)),
        ([100, 3, 4], (True, True, True)),
        ([5, 5, 100], (True, True, False)),
        ([4, 3], (True, True)),
        ([], ()),
        ([100, 5, 20, 20], (False, True, True, False)),
    ],
)
def test_hidden_cells(counts: list[int], expected: tuple[bool, ...]) -> None:
    """Malé buňky doplní nejmenší zachovaná, při shodě dřívější."""
    assert dua.hidden_cells(counts) == expected


@pytest.mark.parametrize(
    ("labels", "counts", "expected"),
    [
        (
            ("velká", "malá A", "malá B", "střední"),
            (100, 4, 3, 20),
            (("velká", 100), ("ostatní (n < 10)", 27)),
        ),
        (
            ("velká", "střední", "malá"),
            (100, 40, 4),
            (("velká", 100), ("ostatní (n < 10)", 44)),
        ),
        (("velká", "a", "b"), (100, 6, 5), (("velká", 100), ("ostatní (n < 10)", 11))),
        (("a", "b"), (10, 25), (("a", 10), ("b", 25))),
        (("a", "b"), (50, 60), (("a", 50), ("b", 60))),
        ((), (), ()),
    ],
)
def test_suppress_small_cells(
    labels: tuple[str, ...],
    counts: tuple[int, ...],
    expected: tuple[tuple[str, int], ...],
) -> None:
    """Sloučení odstraní původní popisky a zachová pořadí i součty."""
    rows = tuple(
        dua.CategoryCount((label,), n) for label, n in zip(labels, counts, strict=True)
    )
    assert dua.suppress_small_cells(rows) == tuple(
        dua.CategoryCount((label,), n) for label, n in expected
    )


def test_suppression_with_groups() -> None:
    """Každá skupina se potlačí samostatně a její popisek zůstane."""
    rows = tuple(
        dua.CategoryCount(labels, n)
        for labels, n in [
            (("A", "x"), 100),
            (("A", "y"), 4),
            (("A", "z"), 50),
            (("B", "x"), 60),
            (("B", "y"), 70),
            (("B", "z"), 80),
        ]
    )
    assert dua.suppress_small_cells(rows, group=0) == (
        dua.CategoryCount(("A", "x"), 100),
        dua.CategoryCount(("A", "ostatní (n < 10)"), 54),
        *rows[3:],
    )


def test_nonadjacent_groups_and_multiple_labels() -> None:
    """Nesouvislé skupiny se seřadí podle prvního výskytu i při group=1."""
    rows = (
        dua.CategoryCount(("x", "A", "první"), 100),
        dua.CategoryCount(("x", "B", "druhý"), 60),
        dua.CategoryCount(("y", "A", "třetí"), 4),
    )
    assert dua.suppress_small_cells(rows, group=1) == (
        dua.CategoryCount(("ostatní (n < 10)", "A", "ostatní (n < 10)"), 104),
        rows[1],
    )


@given(st.lists(st.sampled_from((*range(13), 50, 400)), max_size=8))
def test_hidden_counts_cannot_disclose_small_sum(counts: list[int]) -> None:
    """Skryté buňky splňují práh, ledaže nezbývá žádná zachovaná."""
    hidden = dua.hidden_cells(counts)
    assert len(hidden) == len(counts)
    assert all(mask for n, mask in zip(counts, hidden, strict=True) if n < dua.MIN_CELL)
    total = sum(n for n, mask in zip(counts, hidden, strict=True) if mask)
    assert total == 0 or total >= dua.MIN_CELL or all(hidden)


@given(
    st.lists(st.tuples(st.sampled_from(("A", "B")), st.integers(0, 400)), max_size=8)
)
def test_suppression_preserves_group_totals_and_removes_labels(
    entries: list[tuple[str, int]],
) -> None:
    """Součty zůstanou v každé skupině a skryté kategorie zmizí."""
    rows = tuple(
        dua.CategoryCount((group, f"kategorie {i}"), n)
        for i, (group, n) in enumerate(entries)
    )
    output = dua.suppress_small_cells(rows, group=0)
    assert all(
        row.n >= dua.MIN_CELL for row in output if row.labels[1] != dua.OTHER_LABEL
    )
    for group in ("A", "B"):
        members = [row for row in rows if row.labels[0] == group]
        assert sum(row.n for row in output if row.labels[0] == group) == sum(
            row.n for row in members
        )
        hidden = dua.hidden_cells([row.n for row in members])
        hidden_labels = {
            row.labels[1] for row, mask in zip(members, hidden, strict=True) if mask
        }
        assert hidden_labels.isdisjoint(row.labels[1] for row in output)


@pytest.mark.parametrize("n", [1.5, 42.0, True, -1])
def test_category_count_rejects_invalid_count(n: object) -> None:
    """Počet musí být nezáporné int bez bool a float."""
    with pytest.raises(
        InvalidInputError, match=r"^n musí být nezáporné celé číslo typu int\.$"
    ):
        dua.CategoryCount(("a",), cast("int", n))


@pytest.mark.parametrize("labels", [(1,), (), "velká"])
def test_category_count_rejects_invalid_labels(labels: object) -> None:
    """Popisky jsou neprázdná n-tice textů."""
    with pytest.raises(
        InvalidInputError, match=r"^Popisky musí být neprázdná n-tice textů\.$"
    ):
        dua.CategoryCount(cast("tuple[str, ...]", labels), 10)


def test_suppression_rejects_different_label_counts() -> None:
    """Tabulka musí mít stejný počet popisků v každém řádku."""
    rows = (dua.CategoryCount(("a",), 10), dua.CategoryCount(("b", "c"), 20))
    with pytest.raises(InvalidInputError, match="popisk"):
        dua.suppress_small_cells(rows)


@pytest.mark.parametrize("group", [-1, 1, 2])
def test_suppression_rejects_invalid_group(group: int) -> None:
    """Skupina musí odkazovat na existující popisek."""
    with pytest.raises(InvalidInputError, match="group"):
        dua.suppress_small_cells((dua.CategoryCount(("a",), 50),), group=group)


def test_empty_grouped_table() -> None:
    """Prázdný vstup se skupinou zůstane prázdný."""
    assert dua.suppress_small_cells((), group=0) == ()


@pytest.mark.parametrize(
    ("n", "total", "expected"),
    [
        (9, 100, None),
        (10, 100, 10),
        (50, 100, 50),
        (91, 100, None),
        (0, 100, None),
        (100, 100, None),
        (90, 100, 90),
        (10, 25, 10),
        (10, 19, None),
        (None, 100, None),
        (10, None, None),
    ],
)
def test_binary_count(n: int | None, total: int | None, expected: int | None) -> None:
    """Binární počet chrání i dopočítatelný doplněk a chybějící hodnoty."""
    assert dua.binary_count(n, total) == expected


def test_unknown_label() -> None:
    """Neznámá kategorie má jednotný popisek."""
    assert dua.UNKNOWN_LABEL == "neuvedeno"
