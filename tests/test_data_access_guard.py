"""Stráž adresářů datových vstupů pipeline a EDA."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mediparse.domain.data_access import DataAccessError
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.data_access_guard import check_tables_access
from tests.support import mark_synthetic

if TYPE_CHECKING:
    from pathlib import Path


def test_guard_checks_every_input_directory(tmp_path: Path) -> None:
    """Značka prvního adresáře nepovolí neoznačený druhý adresář."""
    first, second = tmp_path / "first", tmp_path / "second"
    mark_synthetic(first)

    with pytest.raises(DataAccessError) as refusal:
        check_tables_access(
            {"CODEX_THREAD_ID": "x"}, [first / "a.parquet", second / "b.parquet"]
        )

    assert str(second) in str(refusal.value)
    assert "Agent Codex nesmí číst data MIMIC" in str(refusal.value)


def test_guard_allows_marked_directories(tmp_path: Path) -> None:
    """Platná značka každého adresáře dovolí čtení v prostředí Codexu."""
    first, second = tmp_path / "first", tmp_path / "second"
    for directory in (first, second):
        mark_synthetic(directory)

    check_tables_access(
        {"CODEX_THREAD_ID": "x"}, [first / "a.parquet", second / "b.parquet"]
    )


def test_guard_allows_unmarked_directories_without_blocking_variable(
    tmp_path: Path,
) -> None:
    """Prázdné prostředí dovolí vstupy bez značky."""
    check_tables_access(
        {}, [tmp_path / "first/a.parquet", tmp_path / "second/b.parquet"]
    )


def test_guard_rejects_invalid_marker_without_blocking_variable(tmp_path: Path) -> None:
    """Neplatná značka je chyba vstupu i bez blokující proměnné."""
    (tmp_path / "synthetic.json").write_text("{}", encoding="utf-8")

    with pytest.raises(InvalidInputError, match=r"synthetic\.json"):
        check_tables_access({}, [tmp_path / "a.parquet"])
