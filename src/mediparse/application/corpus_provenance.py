"""Use case provenance: k auditovanému korpusu zapíše, čím a podle čeho vznikl.

Provenance vzniká až po čistém auditu a váže se na něj otiskem záznamu auditu.
Soulad korpusu s auditem posuzuje brána; korpus, který jí neprojde, provenance
nedostane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mediparse.application.corpus_gate import audit_violations
from mediparse.domain.corpus_provenance import ProvenanceRecord

if TYPE_CHECKING:
    from mediparse.application.ports.corpus import AuditedCorpus, ProvenanceCorpus
    from mediparse.domain.corpus_audit import AuditPin
    from mediparse.domain.corpus_provenance import Generation


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

    def run(self, generation: Generation, pin: AuditPin) -> ProvenanceOutcome:
        """Ověří, že korpus odpovídá svému auditu, a zapíše provenance s otiskem záznamu auditu.

        Returns:
            Zápis záznamu, nebo odmítnutí s důvodem.
        """
        refusal = audit_refusal(self.corpus, pin)
        if refusal is not None:
            return refusal
        self.corpus.save_provenance(
            ProvenanceRecord(
                generation=generation, audit_sha256=self.corpus.audit_sha256()
            )
        )
        return ProvenanceWritten()


def audit_refusal(corpus: AuditedCorpus, pin: AuditPin) -> ProvenanceRefused | None:
    """Odmítnutí provenance pro korpus bez zpráv nebo bez čistého auditu.

    Returns:
        Odmítnutí s důvodem, nebo None, když korpus odpovídá svému auditu.
    """
    fingerprint = corpus.fingerprint()
    if fingerprint is None:
        return ProvenanceRefused("Korpus neobsahuje žádnou zprávu.")
    violations = audit_violations(corpus, fingerprint, pin)
    if violations:
        return ProvenanceRefused(" ".join(violations))
    return None
