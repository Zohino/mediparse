"""Syntetický korpus na disku: zprávy ve tvaru ``<jazyk>/<note_id>.txt``, záznam auditu a provenance v kořeni."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ValidationError

from mediparse.domain.digest import fingerprint
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.language import Language
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.json_file import write_model
from mediparse.synthetic.domain.corpus_audit import (
    AuditRecord,
    InvalidAuditRecordError,
)
from mediparse.synthetic.domain.corpus_provenance import (
    InvalidProvenanceRecordError,
    ProvenanceRecord,
)

CORPUS_ROOT: Final = Path("resources/synthetic")
RECORD_NAME: Final = "audit.json"
PROVENANCE_NAME: Final = "provenance.json"
PLANS_NAME: Final = "plans.jsonl"
LABELS_NAME: Final = "labels.csv"
_ROOT_FILES: Final = frozenset({RECORD_NAME, PROVENANCE_NAME, PLANS_NAME, LABELS_NAME})
_SUFFIX: Final = ".txt"
_LANGUAGES: Final = frozenset(language.value for language in Language)


@dataclass(frozen=True)
class CorpusDirectory:
    """Korpus v adresáři; neexistující adresář je prázdný korpus."""

    root: Path

    def notes(self, language: Language) -> dict[str, str]:
        """Texty zpráv jednoho jazyka.

        Nepovolenou cestu v korpusu a zprávu, která nejde přečíst nebo není v UTF-8,
        odmítne jako InvalidInputError.

        Returns:
            Slovník note_id → text zprávy; jazyk bez zpráv dá prázdný slovník.
        """
        return {
            path.stem: self._read(path)
            for path in self._paths()
            if path.parent.name == language.value
        }

    def note_path(self, language: Language, note_id: str) -> Path:
        """Cesta ke zprávě; disk nečte.

        Returns:
            ``<kořen>/<jazyk>/<note_id>.txt``, tedy soubor, který čte ``notes``.
        """
        return self.root / language.value / f"{note_id}{_SUFFIX}"

    def fingerprint(self) -> str | None:
        """Otisk zpráv tak, jak leží na disku; nepovolenou cestu v korpusu odmítne jako InvalidInputError.

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
        write_model(self.root / name, record)

    def _paths(self) -> list[Path]:
        # Brána, audit i otisk musí odmítnout stejný strom; cizí soubory by jinak
        # odešly do image bez auditu.
        paths = sorted(self.root.rglob("*"))
        unexpected = [
            self._relative(path) for path in paths if not self._is_expected_path(path)
        ]
        if unexpected:
            msg = f"Korpus {self.root} obsahuje nepovolené cesty: {', '.join(unexpected)}."
            raise InvalidInputError(msg)
        return [path for path in paths if self._is_note(path)]

    def _is_expected_path(self, path: Path) -> bool:
        if path.parent == self.root:
            return (path.is_file() and path.name in _ROOT_FILES) or (
                path.is_dir() and path.name in _LANGUAGES
            )
        return self._is_note(path)

    def _is_note(self, path: Path) -> bool:
        return (
            path.parent.parent == self.root
            and path.parent.name in _LANGUAGES
            and path.is_file()
            and path.suffix == _SUFFIX
        )

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
