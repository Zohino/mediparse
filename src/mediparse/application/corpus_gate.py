"""Use case brány: korpus smí do repozitáře, jen když jeho otisk sedí se záznamem auditu proti připnuté referenci a provenance ukazuje na tento audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.application.corpus_texts import TRANSLATION_LANGUAGE, texts_in_language
from mediparse.application.ports import AuditedCorpus, NoteSource
from mediparse.domain.corpus_audit import (
    InvalidAuditRecordError,
    audit_record_violations,
)
from mediparse.domain.corpus_provenance import (
    InvalidProvenanceRecordError,
    provenance_violations,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from mediparse.domain.corpus_provenance import ProvenanceRecord


class GatedCorpus(AuditedCorpus, NoteSource, Protocol):
    """Korpus, jak ho vidí brána: korpus se záznamem auditu a provenance a s texty zpráv."""

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

    corpus: GatedCorpus

    def run(
        self, reference_sha256: Mapping[str, str], structure_labels: Collection[str]
    ) -> tuple[str, ...]:
        """Důvody, proč korpus nesmí do repozitáře; prázdný výsledek znamená, že smí.

        Provenance se kontroluje až u korpusu, který odpovídá svému auditu.

        Returns:
            Popisy porušení; neexistující korpus nic neporušuje.
        """
        fingerprint = self.corpus.fingerprint()
        if fingerprint is None:
            return ()
        violations = audit_violations(
            self.corpus, fingerprint, reference_sha256, structure_labels
        )
        if violations:
            return violations
        try:
            provenance = self.corpus.provenance_record()
        except InvalidProvenanceRecordError:
            return ("Záznam provenance neodpovídá schématu.",)
        translated = bool(texts_in_language(self.corpus.notes(), TRANSLATION_LANGUAGE))
        return provenance_violations(
            provenance, self.corpus.audit_sha256(), translated=translated
        )


def audit_violations(
    corpus: AuditedCorpus,
    fingerprint: str,
    reference_sha256: Mapping[str, str],
    structure_labels: Collection[str],
) -> tuple[str, ...]:
    """Důvody, proč korpus s daným otiskem neodpovídá svému záznamu auditu; provenance nekontroluje.

    Returns:
        Popisy porušení auditu.
    """
    try:
        record = corpus.audit_record()
    except InvalidAuditRecordError:
        return ("Záznam auditu neodpovídá schématu.",)
    return audit_record_violations(
        fingerprint, record, reference_sha256, structure_labels
    )
