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
from tests.model_support import write_raw_split

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

LEAKING: Final = 987654
CONFIG: Final = SplitConfig(seed=7, folds=3)
SPLIT: Final = Split(
    (frozenset({2, 40}), frozenset({10}), frozenset({1})), development=frozenset({5})
)
SHARED: Final = SharedSplit(CONFIG, SPLIT)


def _metadata(split: Split = SPLIT, **overrides: object) -> dict[str, object]:
    return {"folds": 3, "seed": 7, "sha256": split.sha256()} | overrides


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
    shared = SharedSplit(CONFIG, Split(folds, development=frozenset()))
    table = ParquetSplitTable(tmp_path / "splits.parquet")

    table.write(shared)

    assert table.read() == shared


def test_rows_are_sorted_numerically_by_subject(tmp_path: Path) -> None:
    """Řádky jsou seřazené podle subject_id jako čísla (2 před 10)."""
    path = tmp_path / "splits.parquet"

    ParquetSplitTable(path).write(SHARED)

    table = pq.read_table(path)
    assert table.schema.equals(SCHEMA)
    assert table.column("subject_id").to_pylist() == [1, 2, 5, 10, 40]
    assert table.column("fold").to_pylist() == [2, 0, None, 1, 0]


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


def test_metadata_bytes_are_canonical(tmp_path: Path) -> None:
    """Doslovné bajty metadat mají abecední pořadí klíčů a známý otisk splitu."""
    path = tmp_path / "splits.parquet"
    shared = SharedSplit(
        SplitConfig(seed=0, folds=2),
        Split((frozenset({2}), frozenset({10})), development=frozenset({5})),
    )

    ParquetSplitTable(path).write(shared)

    assert pq.read_schema(path).metadata[METADATA_KEY] == (
        b'{"folds": 2, "seed": 0, "sha256": '
        b'"0f4a94aad52205bf10af72b289f926bffcbdce3fc141500daf9762d7777ce608"}'
    )


def test_read_missing_file_is_invalid_input(tmp_path: Path) -> None:
    """Chybějící soubor je InvalidInputError s cestou."""
    path = tmp_path / "missing.parquet"

    with pytest.raises(InvalidInputError, match=str(path)):
        ParquetSplitTable(path).read()


@pytest.mark.parametrize(
    ("rows", "metadata", "reason"),
    [
        ([(1, 0)], None, "nemá metadata splitu"),
        ([(1, 0)], _metadata(folds=1), "neplatná metadata splitu"),
        ([(1, 0)], _metadata() | {"extra": 1}, "neplatná metadata splitu"),
        ([(1, -1)], _metadata(), r"mimo 0 až 2: 1\."),
        ([(1, 3)], _metadata(), r"mimo 0 až 2: 1\."),
        ([(1, 0), (2, 1), (2, 2)], _metadata(), "ve více částech splitu: 1"),
        ([(1, 0), (2, 1), (2, 1), (3, 2)], _metadata(), "řádků"),
        ([(1, None), (1, None)], _metadata(), "řádků"),
        ([(1, 0), (1, None)], _metadata(), "ve více částech splitu: 1"),
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
        "duplicitni-vyvojovy-pacient",
        "vyvojovy-pacient-ve-foldu",
        "otisk-nesedi",
    ],
)
def test_read_invalid_file_is_invalid_input(
    tmp_path: Path,
    rows: Sequence[tuple[int, int | None]],
    metadata: dict[str, object] | None,
    reason: str,
) -> None:
    """Každý nepřípustný obsah souboru je InvalidInputError s cestou."""
    path = tmp_path / "splits.parquet"
    write_raw_split(path, rows, metadata)

    with pytest.raises(InvalidInputError, match=reason) as raised:
        ParquetSplitTable(path).read()

    assert str(path) in str(raised.value)


def test_read_subject_in_two_folds_hides_identifier(tmp_path: Path) -> None:
    """Pacient ve dvou foldech: hláška nese počet, ne subject_id, a výjimka bez řetězu příčin."""
    path = tmp_path / "splits.parquet"
    write_raw_split(path, [(LEAKING, 0), (LEAKING, 1)], _metadata())

    with pytest.raises(InvalidInputError, match="ve více částech splitu: 1") as raised:
        ParquetSplitTable(path).read()

    assert str(LEAKING) not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True


def test_read_other_schema_is_invalid_input(tmp_path: Path) -> None:
    """Soubor s jiným schématem je InvalidInputError."""
    path = tmp_path / "other.parquet"
    pq.write_table(pa.table({"subject_id": [1]}), path)

    with pytest.raises(InvalidInputError, match="neodpovídá schématu"):
        ParquetSplitTable(path).read()


def test_null_fold_is_development(tmp_path: Path) -> None:
    """NULL ve foldu označuje vývojového pacienta i při čtení nezávislého souboru."""
    path = tmp_path / "splits.parquet"
    write_raw_split(
        path,
        [(2, 0), (5, None), (10, 1)],
        {
            "seed": 0,
            "folds": 2,
            "sha256": "0f4a94aad52205bf10af72b289f926bffcbdce3fc141500daf9762d7777ce608",
        },
    )

    assert ParquetSplitTable(path).read().split == Split(
        (frozenset({2}), frozenset({10})), development=frozenset({5})
    )


def test_old_nonnullable_fold_schema_is_rejected(tmp_path: Path) -> None:
    """Původní schéma s povinným foldem se odmítne i s platnými metadaty a otiskem."""
    path = tmp_path / "old.parquet"
    schema = pa.schema(
        [
            pa.field("subject_id", pa.int64(), nullable=False),
            pa.field("fold", pa.int64(), nullable=False),
        ],
        metadata={
            METADATA_KEY: json.dumps({
                "seed": 0,
                "folds": 2,
                "sha256": "91f9e8d3b27133625651802f23ad0fbbdb182b788cbf7058427385a493b5c85d",
            }).encode()
        },
    )
    pq.write_table(
        pa.Table.from_pydict({"subject_id": [2, 10], "fold": [0, 1]}, schema=schema),
        path,
    )

    with pytest.raises(InvalidInputError, match="neodpovídá schématu"):
        ParquetSplitTable(path).read()
