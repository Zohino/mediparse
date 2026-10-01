"""Use case brány: korpus smí do repozitáře, jen když jeho otisk sedí se záznamem auditu a audit běžel proti připnuté referenci."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from mediparse.domain.corpus_audit import InvalidAuditRecordError, gate_violations

if TYPE_CHECKING:
    from collections.abc import Mapping

    from mediparse.domain.corpus_audit import AuditRecord


class AuditedCorpus(Protocol):
    """Korpus, jak ho vidí brána: otisk zpráv a záznam posledního auditu."""

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


@dataclass(frozen=True)
class CorpusGate:
    """Brána syntetického korpusu; s daty MIMIC nepracuje, smí proto běžet i ve veřejném CI."""

    corpus: AuditedCorpus

    def run(self, reference_sha256: Mapping[str, str]) -> tuple[str, ...]:
        """Důvody, proč korpus nesmí do repozitáře; prázdný výsledek znamená, že smí.

        Returns:
            Popisy porušení; neexistující korpus nic neporušuje.
        """
        try:
            record = self.corpus.audit_record()
        except InvalidAuditRecordError:
            return ("Záznam auditu neodpovídá schématu.",)
        return gate_violations(self.corpus.fingerprint(), record, reference_sha256)
