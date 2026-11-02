"""Referenční soubor MIMIC-IV-Note stažený jako ``.csv.gz``, čtený proudově."""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from mediparse.domain.corpus_audit import ReferenceNote
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.file_digest import file_sha256

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

_COLUMNS: Final = frozenset({"subject_id", "text"})


@dataclass(frozen=True)
class MimicReference:
    """Tabulka discharge nebo radiology z MIMIC-IV-Note na lokálním disku."""

    path: Path

    @property
    def name(self) -> str:
        """Jméno souboru, pod kterým se reference uvede v záznamu auditu."""
        return self.path.name

    def notes(self) -> Iterator[ReferenceNote]:
        """Zprávy po jedné, bez načtení celého souboru do paměti.

        Chyba čtení vznikne až při iteraci, protože soubor se čte proudově.

        Yields:
            Pacient a text každé zprávy.

        Raises:
            InvalidInputError: Soubor nejde otevřít, je poškozený, nemá sloupce
                ``subject_id`` a ``text``, má krátký řádek nebo ``subject_id`` není číslo.
        """
        line = 0
        try:
            with gzip.open(
                self.path, mode="rt", encoding="utf-8", newline=""
            ) as stream:
                reader = csv.DictReader(stream)
                _require_columns(reader, self.path.name)
                for row in reader:
                    line = reader.line_num
                    yield _note(row)
        except InvalidInputError:
            raise
        except (OSError, EOFError, csv.Error, KeyError, ValueError) as error:
            msg = f"Reference {self.path.name} nejde přečíst u řádku {line}: {type(error).__name__}."
            # Text původní výjimky (u int() i hodnota subject_id z MIMIC) nesmí do hlášky
            # ani do logu, a proto ``from None`` místo ``from error``.
            raise InvalidInputError(msg) from None

    def sha256(self) -> str:
        """SHA-256 souboru, srovnatelný s ``SHA256SUMS.txt`` na PhysioNetu.

        Returns:
            Hexadecimální otisk.
        """
        return file_sha256(self.path)


def _require_columns(reader: csv.DictReader[str], name: str) -> None:
    missing = sorted(_COLUMNS - set(reader.fieldnames or ()))
    if missing:
        msg = f"Reference {name} nemá sloupce: {', '.join(missing)}."
        raise InvalidInputError(msg)


def _note(row: dict[str, str | None]) -> ReferenceNote:
    subject_id, text = row["subject_id"], row["text"]
    if subject_id is None or text is None:
        msg = "krátký řádek"
        raise ValueError(msg)
    return ReferenceNote(subject_id=int(subject_id), text=text)
