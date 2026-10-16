"""Syntetický korpus na disku: zprávy ve tvaru ``<jazyk>/<note_id>.txt``, záznam auditu a provenance v kořeni."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ValidationError

from mediparse.domain.corpus_audit import (
    AuditRecord,
    InvalidAuditRecordError,
)
from mediparse.domain.corpus_provenance import (
    InvalidProvenanceRecordError,
    ProvenanceRecord,
)
from mediparse.domain.digest import fingerprint
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.language import Language
from mediparse.infrastructure.file_digest import file_sha256

CORPUS_ROOT: Final = Path("resources/synthetic")
RECORD_NAME: Final = "audit.json"
PROVENANCE_NAME: Final = "provenance.json"
_NOTES: Final = "*/*.txt"
_LANGUAGES: Final = frozenset(language.value for language in Language)


@dataclass(frozen=True)
class CorpusDirectory:
    """Korpus v adresáři; neexistující adresář je prázdný korpus."""

    root: Path

    def notes(self, language: Language) -> dict[str, str]:
        """Texty zpráv jednoho jazyka.

        Adresář jazyka mimo Language a zprávu, která nejde přečíst nebo není v UTF-8,
        odmítne jako InvalidInputError.

        Returns:
            Slovník note_id → text zprávy; jazyk bez zpráv dá prázdný slovník.
        """
        return {
            path.stem: self._read(path)
            for path in self._paths()
            if path.parent.name == language.value
        }

    def fingerprint(self) -> str | None:
        """Otisk zpráv tak, jak leží na disku; adresář jazyka mimo Language odmítne jako InvalidInputError.

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
        # Otisk pokrývá všechny */*.txt, audit jen jazyky z Language. Neznámý adresář by tak
        # byl otisknutý, ale neauditovaný, a proto se odmítne zde, v jediném místě pro obojí.
        paths = sorted(self.root.glob(_NOTES))
        unknown = sorted({
            path.parent.name for path in paths if path.parent.name not in _LANGUAGES
        })
        if unknown:
            msg = f"Korpus {self.root} obsahuje adresář neznámého jazyka: {', '.join(unknown)}."
            raise InvalidInputError(msg)
        return paths

    def _read(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as error:
            msg = (
                f"Zpráva {self._relative(path)} nejde přečíst: {type(error).__name__}."
            )
            raise InvalidInputError(msg) from error

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()
