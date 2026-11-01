"""Dekorace korpusových tabulek buňkami nalezitelnými kanárkem EDA."""

from __future__ import annotations

import math
from string import ascii_lowercase
from typing import TYPE_CHECKING, Final

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.section_headers import split_sections

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mediparse.domain.mimic_table import TableShape
    from mediparse.domain.section_headers import SectionLexicon

_KEY_COLUMNS: Final = frozenset({"subject_id", "hadm_id", "note_id"})
_KEY_OWNERS: Final = {"subject_id": "patients", "hadm_id": "admissions"}
_REFERENCES: Final = {("mimic-iv-bhc", "note_id"): "discharge"}


def section_tokens(
    discharge: Sequence[Mapping[str, str]], lexicon: SectionLexicon, prefix: str
) -> list[dict[str, str]]:
    """Vloží do neprázdného těla každé části zprávy vlastní token kanárku.

    Returns:
        Nové řádky s dekorovaným textem a ostatními sloupci beze změny.

    Raises:
        InvalidInputError: Prefix syntetických buněk je prázdný.
    """
    _require_prefix(prefix)
    # _decorate_row rezervuje pro první řádky čísla 0 až n-1; těla musí mít jiné tokeny.
    number = len(discharge)
    decorated = []
    for row in discharge:
        parts = []
        for part in split_sections(row["text"], lexicon):
            body = part.body
            if body.strip():
                body = _body_token(body, _text_token(prefix, number))
                number += 1
            parts.append(part.header + body)
        decorated.append({**row, "text": "".join(parts)})
    return decorated


def _body_token(body: str, token: str) -> str:
    lines = body.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.strip():
            # Vlastní řádek by z řezu BHC „první řádek“ udělal jen token.
            content = line.rstrip("\r\n")
            lines[index] = f"{content} {token}{line[len(content) :]}"
            break
    return "".join(lines)


def sentinel_tables(
    corpus: Mapping[str, Sequence[Mapping[str, str]]],
    shapes: Mapping[str, TableShape],
    prefix: str,
    text_chars: int,
) -> dict[str, list[dict[str, str]]]:
    """Doplní tabulky manifestu, přeznačí klíče a vloží svědky úniku textu a popisků.

    Returns:
        Nové řádky všech tabulek se sloupci v pořadí manifestu.

    Raises:
        InvalidInputError: Prefix je prázdný, tabulka nemá sloupce nebo manifest
            neobsahuje korpusovou tabulku či její sloupec.
    """
    _require_prefix(prefix)
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


def _require_prefix(prefix: str) -> None:
    if not prefix:
        msg = "Prefix syntetických buněk nesmí být prázdný."
        raise InvalidInputError(msg)


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
    return [dict(row) for row in corpus[table]]


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
                header_witness=table == "discharge" and index == 0,
            )
        else:
            decorated[column] = original if original is not None else cell
    return decorated


def _text_value(
    original: str | None,
    cell: str,
    token: str,
    text_chars: int,
    *,
    header_witness: bool,
) -> str:
    if original is not None:
        # Samostatný první řádek nemění hlavičky; token s DF=1 projde tokenizací.
        # Svědek n=1 hlídá publishable_floor, až žebříček nevyčerpá limit 40 hlaviček
        # (méně hlaviček nad prahem nebo větší limit). Dnes ho skryje už limit;
        # samotný práh hlídá testthat.
        header = f"{token.capitalize()}:\n" if header_witness else ""
        return f"{cell} {token}\n{header}{original}"
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
