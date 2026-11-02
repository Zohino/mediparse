"""Vstupy klíče tokenové cache, paralelní texty a deterministický vzorek."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Final

import duckdb

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.hf_tokenizers import library_versions
from mediparse.infrastructure.note_features_cache import cache_key

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence
    from pathlib import Path

    from mediparse.domain.note_tokens import TokenizerInfo
    from mediparse.infrastructure.eda_duckdb import EdaDatabase


SAMPLE_LIMIT: Final = 1000


def read_parallel(english: Path, czech: Path) -> list[tuple[str, str, str]]:
    """Spáruje veřejné texty podle jména a zachová jejich konce řádků.

    Returns:
        Identifikátor páru a oba texty v pořadí jmen souborů.

    Raises:
        InvalidInputError: Chybí zprávy, protějšek nebo platný UTF-8 obsah.
    """
    en = {path.name: path for path in english.glob("*.txt")}
    cs = {path.name: path for path in czech.glob("*.txt")}
    if not en and not cs:
        msg = f"V adresářích {english} a {czech} nejsou žádné zprávy (*.txt)."
        raise InvalidInputError(msg)
    missing = sorted(en.keys() ^ cs.keys())
    if missing:
        msg = f"Zpráva nemá protějšek: {', '.join(missing)}."
        raise InvalidInputError(msg)
    return [
        (en[name].stem, _read_text(en[name]), _read_text(cs[name]))
        for name in sorted(en)
    ]


def _read_text(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8")
    except (OSError, UnicodeError) as error:
        msg = f"Syntetický pár {path.name} nelze přečíst: {type(error).__name__}."
        raise InvalidInputError(msg) from None


def tokens_key(
    views: Mapping[str, Path],
    translation: Path,
    english: Path,
    czech: Path,
    infos: Sequence[TokenizerInfo],
) -> str:
    """Otiskne tabulky, skripty, páry, revize a verze tokenizace.

    Returns:
        Klíč skupiny tokens se stejnou konstrukcí jako cache sešitu 9.

    Raises:
        InvalidInputError: Vstup klíče nelze přečíst.
    """
    try:
        inputs = [
            (name, file_sha256(views[name]).encode())
            for name in ("discharge", "radiology")
        ]
        inputs.extend(
            (f"translation/{path.name}", path.read_bytes())
            for path in sorted(translation.glob("*.py"))
        )
        for label, directory in (("english", english), ("czech", czech)):
            inputs.extend(
                (f"{label}/{path.name}", path.read_bytes())
                for path in sorted(directory.glob("*.txt"))
            )
        inputs.append((
            "tokenizers",
            json.dumps([asdict(info) for info in infos], sort_keys=True).encode(),
        ))
        inputs.extend(library_versions())
        return cache_key(inputs)
    except OSError as error:
        msg = f"Vstup klíče cache nelze přečíst: {type(error).__name__}."
        raise InvalidInputError(msg) from None


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
