"""Syntetický korpus na disku: zprávy ve tvaru ``<jazyk>/<note_id>.txt``, záznam auditu a provenance v kořeni."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ValidationError

from mediparse.domain.corpus_audit import (
    AuditRecord,
    InvalidAuditRecordError,
    fingerprint,
)
from mediparse.domain.corpus_provenance import (
    InvalidProvenanceRecordError,
    ProvenanceRecord,
)
from mediparse.infrastructure.file_digest import file_sha256

CORPUS_ROOT: Final = Path("resources/synthetic")
RECORD_NAME: Final = "audit.json"
PROVENANCE_NAME: Final = "provenance.json"
_NOTES: Final = "*/*.txt"


@dataclass(frozen=True)
class CorpusDirectory:
    """Korpus v adresáři; neexistující adresář je prázdný korpus."""

    root: Path

    def notes(self) -> dict[str, str]:
        """Texty zpráv klíčované cestou relativní ke kořeni korpusu.

        Returns:
            Slovník relativní cesta → text zprávy.
        """
        return {
            self._relative(path): path.read_text(encoding="utf-8")
            for path in self._paths()
        }

    def note_ids(self) -> list[str]:
        """Note_id zpráv podle jmen souborů.

        Returns:
            Jména souborů bez přípony, seřazená podle cesty.
        """
        return [path.stem for path in self._paths()]

    def fingerprint(self) -> str | None:
        """Otisk zpráv tak, jak leží na disku.

        Returns:
            SHA-256 otisk, nebo None, když korpus neobsahuje žádnou zprávu.
        """
        paths = self._paths()
        if not paths:
            return None
        return fingerprint((self._relative(path), path.read_bytes()) for path in paths)

    def audit_record(self) -> AuditRecord | None:
        """Záznam posledního auditu.

        Returns:
            Záznam auditu, nebo None, když chybí.

        Raises:
            InvalidAuditRecordError: Obsah záznamu neodpovídá schématu.
        """
        path = self.root / RECORD_NAME
        if not path.exists():
            return None
        try:
            return AuditRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as error:
            raise InvalidAuditRecordError from error

    def audit_sha256(self) -> str:
        """Otisk souboru se záznamem auditu tak, jak leží na disku.

        Returns:
            SHA-256 otisk.
        """
        return file_sha256(self.root / RECORD_NAME)

    def save_record(self, record: AuditRecord) -> None:
        """Zapíše záznam auditu do kořene korpusu."""
        self._write(RECORD_NAME, record)

    def provenance_record(self) -> ProvenanceRecord | None:
        """Záznam provenance korpusu.

        Returns:
            Záznam provenance, nebo None, když chybí.

        Raises:
            InvalidProvenanceRecordError: Obsah záznamu neodpovídá schématu.
        """
        path = self.root / PROVENANCE_NAME
        if not path.exists():
            return None
        try:
            return ProvenanceRecord.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except ValidationError as error:
            raise InvalidProvenanceRecordError from error

    def save_provenance(self, record: ProvenanceRecord) -> None:
        """Zapíše záznam provenance do kořene korpusu."""
        self._write(PROVENANCE_NAME, record)

    def _write(self, name: str, record: BaseModel) -> None:
        (self.root / name).write_text(
            f"{record.model_dump_json(indent=2)}\n", encoding="utf-8"
        )

    def _paths(self) -> list[Path]:
        return sorted(self.root.glob(_NOTES))

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()
