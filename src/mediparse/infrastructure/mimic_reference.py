"""Referenční soubor MIMIC-IV-Note stažený jako ``.csv.gz``, čtený proudově."""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mediparse.domain.corpus_audit import ReferenceNote
from mediparse.infrastructure.file_digest import file_sha256

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


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

        Yields:
            Pacient a text každé zprávy.
        """
        with gzip.open(self.path, mode="rt", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                yield ReferenceNote(subject_id=row["subject_id"], text=row["text"])

    def sha256(self) -> str:
        """SHA-256 souboru, srovnatelný s ``SHA256SUMS.txt`` na PhysioNetu.

        Returns:
            Hexadecimální otisk.
        """
        return file_sha256(self.path)
