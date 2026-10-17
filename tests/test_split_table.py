"""Parquet adaptér sdíleného splitu: řádek na pacienta, parametry v metadatech, odmítnutí soubor."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.split import SharedSplit, Split, SplitConfig
from mediparse.infrastructure.split_table import METADATA_KEY, SCHEMA, ParquetSplitTable

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

LEAKING: Final = 987654
CONFIG: Final = SplitConfig(seed=7, folds=3)
SPLIT: Final = Split((frozenset({2, 40}), frozenset({10}), frozenset({1})))
SHARED: Final = SharedSplit(CONFIG, SPLIT)


def _metadata(split: Split = SPLIT, **overrides: object) -> dict[str, object]:
    return {"folds": 3, "seed": 7, "sha256": split.sha256()} | overrides


def _write_raw(
    path: Path,
    rows: Sequence[tuple[int, int]],
    metadata: dict[str, object] | None,
) -> None:
    """Zapíše soubor mimo adaptér, aby šlo vyrobit stav, který adaptér nezapíše."""
    schema = SCHEMA
    if metadata is not None:
        schema = SCHEMA.with_metadata({METADATA_KEY: json.dumps(metadata).encode()})
    columns = {
        "subject_id": [subject for subject, _ in rows],
        "fold": [fold for _, fold in rows],
    }
    pq.write_table(pa.Table.from_pydict(columns, schema=schema), path)


def test_read_returns_written_split(tmp_path: Path) -> None:
    """Zápis a čtení vrátí týž SharedSplit."""
    table = ParquetSplitTable(tmp_path / "out" / "splits.parquet")

    table.write(SHARED)

    assert table.read() == SHARED


@pytest.mark.parametrize(
    "folds",
    [
        (frozenset({1, 2}), frozenset(), frozenset({3})),
        (frozenset({1, 2}), frozenset({3}), frozenset()),
    ],
    ids=["prazdny-uprostred", "prazdny-posledni"],
)
def test_empty_fold_survives_round_trip(
    tmp_path: Path, folds: tuple[frozenset[int], ...]
) -> None:
    """K je z metadat, takže prázdný fold, i poslední, zůstane prázdným foldem."""
    shared = SharedSplit(CONFIG, Split(folds))
    table = ParquetSplitTable(tmp_path / "splits.parquet")

    table.write(shared)

    assert table.read() == shared


def test_rows_are_sorted_numerically_by_subject(tmp_path: Path) -> None:
    """Řádky jsou seřazené podle subject_id jako čísla (2 před 10)."""
    path = tmp_path / "splits.parquet"

    ParquetSplitTable(path).write(SHARED)

    table = pq.read_table(path)
    assert table.schema.equals(SCHEMA)
    assert table.column("subject_id").to_pylist() == [1, 2, 10, 40]
    assert table.column("fold").to_pylist() == [2, 0, 1, 0]


def test_same_split_gives_identical_bytes(tmp_path: Path) -> None:
    """Dva zápisy téhož splitu dají bajtově shodný soubor."""
    ParquetSplitTable(tmp_path / "a.parquet").write(SHARED)
    ParquetSplitTable(tmp_path / "b.parquet").write(SHARED)

    assert (tmp_path / "a.parquet").read_bytes() == (
        tmp_path / "b.parquet"
    ).read_bytes()


def test_metadata_carry_seed_folds_and_fingerprint(tmp_path: Path) -> None:
    """Metadata souboru nesou seed, K a otisk přiřazení."""
    path = tmp_path / "splits.parquet"

    ParquetSplitTable(path).write(SHARED)

    stored = pq.read_table(path).schema.metadata[METADATA_KEY]
    assert json.loads(stored) == {
        "folds": 3,
        "seed": 7,
        "sha256": SPLIT.sha256(),
    }


def test_read_missing_file_is_invalid_input(tmp_path: Path) -> None:
    """Chybějící soubor je InvalidInputError s cestou."""
    path = tmp_path / "missing.parquet"

    with pytest.raises(InvalidInputError, match=str(path)):
        ParquetSplitTable(path).read()


@pytest.mark.parametrize(
    ("rows", "metadata", "reason"),
    [
        ([(1, 0)], None, "metadata"),
        ([(1, 0)], _metadata(folds=1), "metadata"),
        ([(1, 0)], _metadata() | {"extra": 1}, "metadata"),
        ([(1, -1)], _metadata(), "mimo"),
        ([(1, 3)], _metadata(), "mimo"),
        ([(1, 0), (2, 1), (2, 2)], _metadata(), "ve více foldech: 1"),
        ([(1, 0), (2, 1), (2, 1), (3, 2)], _metadata(), "řádků"),
        ([(1, 0), (2, 1), (3, 2)], _metadata(sha256="0" * 64), "otisk"),
    ],
    ids=[
        "bez-metadat",
        "jeden-fold",
        "pole-navic",
        "fold-zaporny",
        "fold-K",
        "pacient-ve-dvou-foldech",
        "duplicitni-radek",
        "otisk-nesedi",
    ],
)
def test_read_invalid_file_is_invalid_input(
    tmp_path: Path,
    rows: Sequence[tuple[int, int]],
    metadata: dict[str, object] | None,
    reason: str,
) -> None:
    """Každý nepřípustný obsah souboru je InvalidInputError s cestou a bez identifikátorů."""
    path = tmp_path / "splits.parquet"
    _write_raw(path, rows, metadata)

    with pytest.raises(InvalidInputError, match=reason) as raised:
        ParquetSplitTable(path).read()

    assert str(path) in str(raised.value)


def test_read_subject_in_two_folds_hides_identifier(tmp_path: Path) -> None:
    """Pacient ve dvou foldech: hláška nese počet, ne subject_id, a výjimka bez příčiny."""
    path = tmp_path / "splits.parquet"
    _write_raw(path, [(LEAKING, 0), (LEAKING, 1)], _metadata())

    with pytest.raises(InvalidInputError, match="ve více foldech: 1") as raised:
        ParquetSplitTable(path).read()

    assert str(LEAKING) not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_read_other_schema_is_invalid_input(tmp_path: Path) -> None:
    """Soubor s jiným schématem je InvalidInputError."""
    path = tmp_path / "other.parquet"
    pq.write_table(pa.table({"subject_id": [1]}), path)

    with pytest.raises(InvalidInputError, match="neodpovídá schématu"):
        ParquetSplitTable(path).read()
