"""Use case provenance: k auditovanému korpusu zapíše, čím a podle čeho vznikl.

Provenance vzniká až po čistém auditu a váže se na něj otiskem záznamu auditu.
Korpus, který by neprošel bránou, provenance nedostane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.domain.corpus_audit import InvalidAuditRecordError, gate_violations
from mediparse.domain.corpus_provenance import ProvenanceRecord

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mediparse.domain.corpus_audit import AuditRecord
    from mediparse.domain.corpus_provenance import Generation


class ProvenanceCorpus(Protocol):
    """Korpus, ke kterému se zapisuje provenance: otisk zpráv, záznam auditu a jeho otisk."""

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

    def save_provenance(self, record: ProvenanceRecord) -> None:
        """Uloží záznam provenance ke korpusu."""


@dataclass(frozen=True)
class ProvenanceWritten:
    """Záznam provenance je uložený."""


@dataclass(frozen=True)
class ProvenanceRefused:
    """Provenance nevznikla; důvod je určený člověku."""

    reason: str


type ProvenanceOutcome = ProvenanceWritten | ProvenanceRefused


@dataclass(frozen=True)
class CorpusProvenance:
    """Provenance syntetického korpusu, který prošel auditem proti připnuté referenci."""

    corpus: ProvenanceCorpus

    def run(
        self, generation: Generation, reference_sha256: Mapping[str, str]
    ) -> ProvenanceOutcome:
        """Ověří, že korpus odpovídá svému auditu, a zapíše provenance s otiskem záznamu auditu.

        Returns:
            Zápis záznamu, nebo odmítnutí s důvodem.
        """
        fingerprint = self.corpus.fingerprint()
        if fingerprint is None:
            return ProvenanceRefused("Korpus neobsahuje žádnou zprávu.")
        try:
            audit = self.corpus.audit_record()
        except InvalidAuditRecordError:
            return ProvenanceRefused("Záznam auditu neodpovídá schématu.")
        violations = gate_violations(fingerprint, audit, reference_sha256)
        if violations:
            return ProvenanceRefused(" ".join(violations))
        self.corpus.save_provenance(
            ProvenanceRecord(
                generation=generation, audit_sha256=self.corpus.audit_sha256()
            )
        )
        return ProvenanceWritten()
