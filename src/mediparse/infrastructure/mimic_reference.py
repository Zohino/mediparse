"""Proudové čtení volného textu MIMIC-IV-Note ze staženého souboru ``.csv.gz``."""

from __future__ import annotations

import csv
import gzip
import hashlib
from typing import TYPE_CHECKING

from mediparse.domain.corpus_audit import ReferenceNote

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def reference_notes(path: Path) -> Iterator[ReferenceNote]:
    """Zprávy tabulky discharge nebo radiology po jedné, bez načtení celého souboru do paměti.

    Yields:
        Pacient a text každé zprávy.
    """
    with gzip.open(path, mode="rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            yield ReferenceNote(subject_id=row["subject_id"], text=row["text"])


def file_sha256(path: Path) -> str:
    """SHA-256 souboru, srovnatelný s ``SHA256SUMS.txt`` na PhysioNetu.

    Returns:
        Hexadecimální otisk.
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
