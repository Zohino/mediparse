"""Předpony pětice jsou mezi ICD-9 a ICD-10 konzistentní podle GEMs (CMS 2018)."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Final

import pytest

from mediparse.domain.icd import DiagnosisCodes, IcdCode, IcdVersion
from mediparse.infrastructure.diagnosis_codes_file import load_diagnosis_codes
from tests.gems_support import GemRow, read_gem_rows
from tests.support import REPOSITORY, REPOSITORY_DIAGNOSIS_CANDIDATES

if TYPE_CHECKING:
    from pathlib import Path

GEMS: Final = REPOSITORY / "resources" / "icd"
I9_GEM: Final = GEMS / "2018_I9gem.txt"
I10_GEM: Final = GEMS / "2018_I10gem.txt"
SHA256: Final = {
    I9_GEM: "44f4079c698efa2c0a8cc159f70075cf6cf23277fdd35ae3416b2bc516652ce3",
    I10_GEM: "fe679ddbdc2d3e2275574299ca8c2a5e21e0143136a6ae6290cc7dcec1042f2b",
}
I9_ROWS: Final = 24860
I10_ROWS: Final = 81593
EXCEPTION_ROWS: Final = 38
MANIFESTATION_SOURCES: Final = frozenset({
    "3572",
    "36201",
    "36202",
    "36203",
    "36204",
    "36205",
    "36206",
    "36207",
    "36641",
})
EXCEPTION_PAIRS: Final = frozenset({
    ("4870", "J129"),
    ("J129", "4870"),
    ("514", "J182"),
    ("J182", "514"),
    ("5171", "J17"),
    ("J17", "5171"),
})


@pytest.fixture(scope="module")
def codes() -> DiagnosisCodes:
    """Načte config v repu.

    Returns:
        Předpony pětice.
    """
    return load_diagnosis_codes(REPOSITORY_DIAGNOSIS_CANDIDATES)


@pytest.fixture(scope="module")
def rows() -> list[GemRow]:
    """Přečte oba soubory GEMs.

    Returns:
        Řádky obou směrů.
    """
    rows = [
        *read_gem_rows(I9_GEM, IcdVersion.NINE, IcdVersion.TEN),
        *read_gem_rows(I10_GEM, IcdVersion.TEN, IcdVersion.NINE),
    ]
    assert len(rows) == I9_ROWS + I10_ROWS
    return rows


@pytest.fixture(scope="module")
def universe(rows: list[GemRow]) -> set[IcdCode]:
    """Sloučí zdroje i cíle obou směrů.

    Returns:
        Všechny kódy GEMs.
    """
    sources = {row.source for row in rows}
    targets = {row.target for row in rows if row.target is not None}
    return sources | targets


def _is_exception(row: GemRow) -> bool:
    assert row.target is not None
    if row.source.version is IcdVersion.NINE and row.source.code in (
        MANIFESTATION_SOURCES
    ):
        return True
    return (row.source.code, row.target.code) in EXCEPTION_PAIRS


def _inconsistent(rows: list[GemRow], codes: DiagnosisCodes) -> list[GemRow]:
    return [
        row
        for row in rows
        if row.target is not None
        and not row.combination
        and codes.diagnosis_of(row.source) is not codes.diagnosis_of(row.target)
    ]


@pytest.mark.parametrize("path", [I9_GEM, I10_GEM])
def test_gems_files_are_unchanged(path: Path) -> None:
    """Soubory GEMs z balíku CMS 2018 nešly nepozorovaně upravit."""
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SHA256[path]


def test_every_prefix_has_a_code_in_the_universe(
    codes: DiagnosisCodes, universe: set[IcdCode]
) -> None:
    """Mrtvá předpona nebo překlep nemá v GEMs jediný kód své verze."""
    dead = [
        (diagnosis.value, version.value, prefix)
        for diagnosis, by_version in codes.prefixes.items()
        for version, prefixes in by_version.items()
        for prefix in prefixes
        if not any(
            code.version is version and code.code.startswith(prefix)
            for code in universe
        )
    ]

    assert dead == []


def test_no_code_falls_into_two_groups(
    codes: DiagnosisCodes, universe: set[IcdCode]
) -> None:
    """Žádný kód univerza GEMs nezačíná předponami dvou diagnóz."""
    for code in universe:
        matching = {
            diagnosis
            for diagnosis, by_version in codes.prefixes.items()
            if code.code.startswith(by_version.get(code.version, ()))
        }
        assert len(matching) <= 1, code
        assert codes.diagnosis_of(code) is (next(iter(matching), None))


def test_equivalent_codes_share_a_diagnosis(
    rows: list[GemRow], codes: DiagnosisCodes
) -> None:
    """Zdroj i cíl nekombinačního řádku mají stejnou diagnózu, kromě vyjmenovaných výjimek."""
    unexplained = [
        (row.source.code, row.target.code)
        for row in _inconsistent(rows, codes)
        if row.target is not None and not _is_exception(row)
    ]

    assert unexplained == []


def test_every_exception_occurs(rows: list[GemRow], codes: DiagnosisCodes) -> None:
    """Žádná vyjmenovaná výjimka není mrtvá."""
    inconsistent = _inconsistent(rows, codes)
    sources = {
        row.source.code for row in inconsistent if row.source.version is IcdVersion.NINE
    }
    pairs = {
        (row.source.code, row.target.code)
        for row in inconsistent
        if row.target is not None
    }

    assert sources >= MANIFESTATION_SOURCES
    assert pairs >= EXCEPTION_PAIRS
