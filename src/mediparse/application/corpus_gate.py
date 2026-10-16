"""Use case brány: korpus smí do repozitáře, jen když jeho otisk sedí se záznamem auditu proti připnuté referenci a provenance ukazuje na tento audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mediparse.domain.corpus_audit import InvalidAuditRecordError
from mediparse.domain.corpus_provenance import (
    InvalidProvenanceRecordError,
    provenance_violations,
)
from mediparse.domain.language import Language

if TYPE_CHECKING:
    from mediparse.application.ports.corpus import AuditedCorpus, GatedCorpus
    from mediparse.domain.corpus_audit import AuditPin


@dataclass(frozen=True)
class CorpusGate:
    """Brána syntetického korpusu; s daty MIMIC nepracuje, smí proto běžet i ve veřejném CI."""

    corpus: GatedCorpus

    def run(self, pin: AuditPin) -> tuple[str, ...]:
        """Důvody, proč korpus nesmí do repozitáře; prázdný výsledek znamená, že smí.

        Provenance se kontroluje až u korpusu, který odpovídá svému auditu.

        Returns:
            Popisy porušení; neexistující korpus nic neporušuje.
        """
        fingerprint = self.corpus.fingerprint()
        if fingerprint is None:
            return ()
        violations = audit_violations(self.corpus, fingerprint, pin)
        if violations:
            return violations
        try:
            provenance = self.corpus.provenance_record()
        except InvalidProvenanceRecordError:
            return ("Záznam provenance neodpovídá schématu.",)
        if provenance is None:
            return ("Korpus nemá záznam provenance.",)
        translated = bool(self.corpus.notes(Language.CS))
        return provenance_violations(
            provenance, self.corpus.audit_sha256(), translated=translated
        )


def audit_violations(
    corpus: AuditedCorpus,
    fingerprint: str,
    pin: AuditPin,
) -> tuple[str, ...]:
    """Důvody, proč korpus s daným otiskem neodpovídá svému záznamu auditu; provenance nekontroluje.

    Returns:
        Popisy porušení auditu.
    """
    try:
        record = corpus.audit_record()
    except InvalidAuditRecordError:
        return ("Záznam auditu neodpovídá schématu.",)
    if record is None:
        return ("Korpus nemá záznam auditu.",)
    return pin.violations(fingerprint, record)
