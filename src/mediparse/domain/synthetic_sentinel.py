"""Dekorace korpusových tabulek buňkami nalezitelnými kanárkem EDA."""

from __future__ import annotations

import math
from string import ascii_lowercase
from typing import TYPE_CHECKING, Final

from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mediparse.domain.mimic_table import TableShape

RARE_ROWS: Final = 2
_KEY_COLUMNS: Final = frozenset({"subject_id", "hadm_id", "note_id"})
_KEY_OWNERS: Final = {"subject_id": "patients", "hadm_id": "admissions"}
_REFERENCES: Final = {("mimic-iv-bhc", "note_id"): "discharge"}
# Vzácná verze i kód musí nést prefix: platná verze by kód zařadila do tříznakové
# skupiny a prefix by se v žebříčku ztratil. Ostatní typové hodnoty nepřidávají kategorii.
_RARE_WITNESSES: Final = frozenset({
    ("patients", "anchor_year_group"),
    ("diagnoses_icd", "icd_version"),
    ("diagnoses_icd", "icd_code"),
})


def sentinel_tables(
    corpus: Mapping[str, Sequence[Mapping[str, str]]],
    shapes: Mapping[str, TableShape],
    prefix: str,
    text_chars: int,
) -> dict[str, list[dict[str, str]]]:
    """Doplní tabulky manifestu, přeznačí klíče a vloží svědky úniku a potlačení.

    Returns:
        Nové řádky všech tabulek se sloupci v pořadí manifestu.

    Raises:
        InvalidInputError: Prefix je prázdný, tabulka nemá sloupce nebo manifest
            neobsahuje korpusovou tabulku či její sloupec.
    """
    if not prefix:
        msg = "Prefix syntetických buněk nesmí být prázdný."
        raise InvalidInputError(msg)
    _validate_corpus(corpus, shapes)
    tables = {}
    for table, shape in shapes.items():
        if not shape.columns:
            msg = f"Tabulka {table} nemá žádné sloupce."
            raise InvalidInputError(msg)
        rows = _source_rows(table, corpus)
        tables[table] = [
            _decorate_row(
                table,
                {column: row.get(column) for column in shape.columns},
                index,
                prefix,
                text_chars,
            )
            for index, row in enumerate(rows)
        ]
    return tables


def _validate_corpus(
    corpus: Mapping[str, Sequence[Mapping[str, str]]],
    shapes: Mapping[str, TableShape],
) -> None:
    if missing := corpus.keys() - shapes.keys():
        msg = f"V manifestu chybí korpusové tabulky: {', '.join(sorted(missing))}."
        raise InvalidInputError(msg)
    for table, rows in corpus.items():
        columns = set(shapes[table].columns)
        unknown = {column for row in rows for column in row if column not in columns}
        if unknown:
            msg = f"Tabulka {table} má sloupce mimo manifest: {', '.join(sorted(unknown))}."
            raise InvalidInputError(msg)


def _source_rows(
    table: str, corpus: Mapping[str, Sequence[Mapping[str, str]]]
) -> list[dict[str, str]]:
    if table not in corpus:
        return [{key: row[key] for key in _KEY_COLUMNS} for row in corpus["discharge"]]
    rows = [dict(row) for row in corpus[table]]
    if table in {"patients", "admissions", "diagnoses_icd"} and rows:
        for index in range(RARE_ROWS):
            rare = {
                column: value
                for column, value in rows[0].items()
                if (table, column) not in _RARE_WITNESSES
            }
            rare["subject_id"] = f"rare-subject-{index}"
            if table != "patients":
                rare["hadm_id"] = f"rare-admission-{index}"
            rows.append(rare)
    return rows


def _decorate_row(
    table: str,
    row: Mapping[str, str | None],
    index: int,
    prefix: str,
    text_chars: int,
) -> dict[str, str]:
    decorated = {}
    for column, original in row.items():
        key = (
            original if column in _KEY_COLUMNS and original is not None else str(index)
        )
        cell = _cell(table, column, prefix, key)
        if column in _KEY_COLUMNS:
            decorated[column] = cell
        elif column == "text":
            decorated[column] = _text_value(
                original,
                cell,
                _text_token(prefix, index),
                text_chars,
            )
        else:
            decorated[column] = original if original is not None else cell
    return decorated


def _text_value(original: str | None, cell: str, token: str, text_chars: int) -> str:
    if original is not None:
        # První řádek zachová korpus i target BHC; token s DF=1 projde tokenizací.
        return f"{cell} {token}\n{original}"
    copies = max(1, math.ceil((text_chars - len(cell)) / (len(token) + 1)))
    return " ".join([cell, *[token] * copies])


def _cell(table: str, column: str, prefix: str, key: str) -> str:
    owner = _REFERENCES.get((table, column)) or _KEY_OWNERS.get(column, table)
    return f"{prefix}-{owner}-{column}-{key}"


def _text_token(prefix: str, row: int) -> str:
    """Složí token ze samých písmen, který tokenizace nezkrátí na sdílený kmen.

    Returns:
        Písmena prefixu a číslo řádku zapsané písmeny; DF v tabulce je jedna.
    """
    stem = "".join(character for character in prefix if character.isalpha())
    return stem + _letters(row)


def _letters(number: int) -> str:
    digits = ""
    while True:
        number, digit = divmod(number, len(ascii_lowercase))
        digits = ascii_lowercase[digit] + digits
        if number == 0:
            return digits
