"""Use case brány nad fakem korpusu: pořadí kontrol, neplatné záznamy a pravidla provenance."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from mediparse.application.corpus_gate import CorpusGate
from mediparse.domain.corpus_audit import (
    AuditRecord,
    InvalidAuditRecordError,
)
from mediparse.domain.corpus_provenance import (
    Generation,
    InvalidProvenanceRecordError,
    ProvenanceRecord,
)
from tests.support import CORPUS_SHA, PINNED, audit_record

AUDIT_FILE = "9" * 64


def _provenance(audit_sha256: str) -> ProvenanceRecord:
    return ProvenanceRecord(
        generation=Generation(
            seed=0,
            sampler_config_sha256="1" * 64,
            prompts_sha256="3" * 64,
            model="claude-opus-5-5",
            claude_code_version="2.1.5",
            generated_on=date(2026, 10, 10),
            specification_commit="d" * 40,
        ),
        audit_sha256=audit_sha256,
    )


PROVENANCE = _provenance(AUDIT_FILE)


@dataclass(frozen=True)
class _Corpus:
    sha256: str | None
    record: AuditRecord | None = None
    invalid_record: bool = False
    provenance: ProvenanceRecord | None = PROVENANCE
    invalid_provenance: bool = False
    audit_file: str = AUDIT_FILE

    def fingerprint(self) -> str | None:
        return self.sha256

    def audit_record(self) -> AuditRecord | None:
        if self.invalid_record:
            raise InvalidAuditRecordError
        return self.record

    def audit_sha256(self) -> str:
        return self.audit_file

    def provenance_record(self) -> ProvenanceRecord | None:
        if self.invalid_provenance:
            raise InvalidProvenanceRecordError
        return self.provenance


def test_audited_corpus_passes() -> None:
    """Korpus, jehož otisk sedí se záznamem, projde bez porušení."""
    assert CorpusGate(_Corpus(CORPUS_SHA, audit_record())).run(PINNED) == ()


def test_missing_corpus_passes_whatever_the_records() -> None:
    """Neexistující korpus nic neporušuje, ani když vedle leží neplatný záznam auditu."""
    assert CorpusGate(_Corpus(None, invalid_record=True)).run(PINNED) == ()


def test_changed_corpus_is_blocked() -> None:
    """Korpus změněný po auditu neprojde."""
    assert CorpusGate(_Corpus("d" * 64, audit_record())).run(PINNED)


def test_invalid_record_is_blocked() -> None:
    """Záznam, který neodpovídá schématu, korpus neatestuje."""
    violations = CorpusGate(_Corpus(CORPUS_SHA, invalid_record=True)).run(PINNED)

    assert violations == ("Záznam auditu neodpovídá schématu.",)


def test_missing_provenance_is_blocked() -> None:
    """Auditovaný korpus bez provenance neprojde."""
    corpus = _Corpus(CORPUS_SHA, audit_record(), provenance=None)

    assert CorpusGate(corpus).run(PINNED) == ("Korpus nemá záznam provenance.",)


def test_provenance_of_other_audit_is_blocked() -> None:
    """Provenance s otiskem jiného záznamu auditu neprojde."""
    corpus = _Corpus(CORPUS_SHA, audit_record(), provenance=_provenance("8" * 64))

    assert CorpusGate(corpus).run(PINNED)


def test_invalid_provenance_is_blocked() -> None:
    """Záznam provenance, který neodpovídá schématu, neprojde."""
    corpus = _Corpus(CORPUS_SHA, audit_record(), invalid_provenance=True)

    assert CorpusGate(corpus).run(PINNED) == ("Záznam provenance neodpovídá schématu.",)
