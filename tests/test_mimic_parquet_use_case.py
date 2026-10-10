"""Use case převodu tabulky MIMIC: výsledek se porovná s inventářem."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from mediparse.application.mimic_parquet import MimicParquetConversion
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.mimic_table import TableShape

SHAPE = TableShape(records=3, columns=("a", "b"))


@dataclass(frozen=True)
class _Fixed:
    shape: TableShape

    def read(self) -> TableShape:
        return self.shape

    def convert(self) -> TableShape:
        return self.shape


def _conversion(inventory: TableShape, converted: TableShape) -> MimicParquetConversion:
    return MimicParquetConversion(_Fixed(inventory), _Fixed(converted))


def test_matching_shape_is_returned() -> None:
    """Shoda s inventářem vrací tvar převedené tabulky."""
    assert _conversion(SHAPE, SHAPE).run() == SHAPE


def test_different_record_count_is_refused() -> None:
    """Jiný počet záznamů je chyba vstupu a hláška nese obě čísla."""
    converted = TableShape(records=2, columns=SHAPE.columns)
    with pytest.raises(InvalidInputError, match=r"2 záznamů.*3"):
        _conversion(SHAPE, converted).run()


@pytest.mark.parametrize("columns", [("a", "c"), ("b", "a"), ("a",)])
def test_different_columns_are_refused(columns: tuple[str, ...]) -> None:
    """Jiné sloupce i jiné pořadí jsou chyba vstupu."""
    converted = TableShape(records=SHAPE.records, columns=columns)
    with pytest.raises(InvalidInputError, match="sloupce"):
        _conversion(SHAPE, converted).run()
