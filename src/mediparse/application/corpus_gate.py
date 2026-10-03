"""Use case brány: korpus smí do repozitáře, jen když jeho otisk sedí se záznamem auditu proti připnuté referenci a provenance ukazuje na tento audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.domain.corpus_audit import InvalidAuditRecordError, gate_violations
from mediparse.domain.corpus_provenance import (
    InvalidProvenanceRecordError,
    provenance_violations,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mediparse.domain.corpus_audit import AuditRecord
    from mediparse.domain.corpus_provenance import ProvenanceRecord


class AuditedCorpus(Protocol):
    """Korpus, jak ho vidí brána: otisk zpráv, záznam posledního auditu a provenance."""

    def fingerprint(self) -> str | None:
        """Otisk zpráv korpusu.

        Returns:
            SHA-256 otisk, nebo None, když korpus neobsahuje žádnou zprávu.
        """

    def audit_record(self) -> AuditRecord | None:
        """Záznam posledního auditu.

        Returns:
            Záznam, nebo None, když chybí.

        Raises:
            InvalidAuditRecordError: Záznam neodpovídá schématu.
        """

    def audit_sha256(self) -> str:
        """Otisk souboru se záznamem auditu; záznam musí existovat.

        Returns:
            SHA-256 otisk.
        """

    def provenance_record(self) -> ProvenanceRecord | None:
        """Záznam provenance korpusu.

        Returns:
            Záznam, nebo None, když chybí.

        Raises:
            InvalidProvenanceRecordError: Záznam neodpovídá schématu.
        """


@dataclass(frozen=True)
class CorpusGate:
    """Brána syntetického korpusu; s daty MIMIC nepracuje, smí proto běžet i ve veřejném CI."""

    corpus: AuditedCorpus

    def run(self, reference_sha256: Mapping[str, str]) -> tuple[str, ...]:
        """Důvody, proč korpus nesmí do repozitáře; prázdný výsledek znamená, že smí.

        Provenance se kontroluje až u korpusu, který odpovídá svému auditu.

        Returns:
            Popisy porušení; neexistující korpus nic neporušuje.
        """
        try:
            record = self.corpus.audit_record()
        except InvalidAuditRecordError:
            return ("Záznam auditu neodpovídá schématu.",)
        fingerprint = self.corpus.fingerprint()
        violations = gate_violations(fingerprint, record, reference_sha256)
        if violations or fingerprint is None:
            return violations
        try:
            provenance = self.corpus.provenance_record()
        except InvalidProvenanceRecordError:
            return ("Záznam provenance neodpovídá schématu.",)
        return provenance_violations(provenance, self.corpus.audit_sha256())
