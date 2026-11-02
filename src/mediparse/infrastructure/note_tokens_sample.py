"""Deterministický vzorek zpráv pro přesné dělení překladu."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import duckdb

from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Generator

    from mediparse.infrastructure.eda_duckdb import EdaDatabase

SAMPLE_LIMIT: Final = 1000


@dataclass(frozen=True)
class SampleNoteTexts:
    """Neprázdné zprávy vybrané stejným hash pořadím jako v R."""

    database: EdaDatabase
    table: str

    def batches(self) -> Generator[list[tuple[str, str]]]:
        """Vybere nejvýš tisíc identifikátorů a teprve k nim dočte texty.

        Yields:
            Jeden vzorek v deterministickém pořadí.

        Raises:
            InvalidInputError: Vzorek nelze číst.
        """
        if self.table not in {"discharge", "radiology"}:
            msg = "Vzorek musí pocházet z discharge nebo radiology."
            raise InvalidInputError(msg)
        # Řazení celých textů by při radiologii spillovalo gigabajty na disk.
        query = f'''
            WITH chosen AS MATERIALIZED (
                SELECT note_id FROM "{self.table}"
                WHERE coalesce(length(text), 0) > 0
                ORDER BY hash(note_id), note_id LIMIT {SAMPLE_LIMIT}
            )
            SELECT chosen.note_id, notes.text FROM chosen
            JOIN "{self.table}" notes USING (note_id)
            ORDER BY hash(chosen.note_id), chosen.note_id
        '''  # ruff: ignore[hardcoded-sql-expression] — jméno z uzavřeného seznamu výše
        try:
            with self.database.connect() as con:
                yield con.execute(query).fetchall()
        except duckdb.Error as error:
            msg = f"Vzorek {self.table} nelze číst: {type(error).__name__}."
            raise InvalidInputError(msg) from None
