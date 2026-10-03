"""Use case provenance: k auditovanému korpusu zapíše, čím a podle čeho vznikl.

Provenance vzniká až po čistém auditu a váže se na něj otiskem záznamu auditu.
Shodu korpusu s auditem posuzuje brána; korpus, který jí neprojde, provenance
nedostane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.application.corpus_gate import audit_violations
from mediparse.application.ports import AuditedCorpus
from mediparse.domain.corpus_provenance import ProvenanceRecord

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from mediparse.domain.corpus_provenance import Generation


class ProvenanceCorpus(AuditedCorpus, Protocol):
    """Korpus se záznamem auditu, ke kterému se zapisuje provenance."""

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
        self,
        generation: Generation,
        reference_sha256: Mapping[str, str],
        structure_labels: Collection[str],
    ) -> ProvenanceOutcome:
        """Ověří, že korpus odpovídá svému auditu, a zapíše provenance s otiskem záznamu auditu.

        Returns:
            Zápis záznamu, nebo odmítnutí s důvodem.
        """
        fingerprint = self.corpus.fingerprint()
        if fingerprint is None:
            return ProvenanceRefused("Korpus neobsahuje žádnou zprávu.")
        violations = audit_violations(
            self.corpus, fingerprint, reference_sha256, structure_labels
        )
        if violations:
            return ProvenanceRefused(" ".join(violations))
        self.corpus.save_provenance(
            ProvenanceRecord(
                generation=generation, audit_sha256=self.corpus.audit_sha256()
            )
        )
        return ProvenanceWritten()
