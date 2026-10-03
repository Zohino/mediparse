"""Use case brány nad fakem korpusu: porušení, neplatné záznamy a korpus beze změny."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

from mediparse.application.corpus_gate import CorpusGate
from mediparse.domain.corpus_audit import (
    NGRAM_SIZE,
    NORMALIZATION,
    AuditRecord,
    InvalidAuditRecordError,
    ReferenceFile,
)
from mediparse.domain.corpus_provenance import (
    Generation,
    InvalidProvenanceRecordError,
    ProvenanceRecord,
)

AUDITED = "a" * 64
PINNED = {"discharge.csv.gz": "b" * 64, "radiology.csv.gz": "e" * 64}
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


def _record(corpus_sha256: str) -> AuditRecord:
    return AuditRecord(
        corpus_sha256=corpus_sha256,
        corpus_files=1,
        ngram_size=NGRAM_SIZE,
        normalization=NORMALIZATION,
        synthetic_ngrams=0,
        reference=tuple(
            ReferenceFile(name=name, sha256=sha256, rows=1)
            for name, sha256 in PINNED.items()
        ),
        shared_ngrams=0,
        colliding_subjects=0,
        tool_commit="c" * 40,
        created_at=datetime(2026, 9, 29, tzinfo=UTC),
    )


def test_audited_corpus_passes() -> None:
    """Korpus, jehož otisk sedí se záznamem, projde bez porušení."""
    assert CorpusGate(_Corpus(AUDITED, _record(AUDITED))).run(PINNED) == ()


def test_missing_corpus_passes_whatever_the_records() -> None:
    """Neexistující korpus nic neporušuje, ani když vedle leží neplatný záznam auditu."""
    assert CorpusGate(_Corpus(None, invalid_record=True)).run(PINNED) == ()


def test_changed_corpus_is_blocked() -> None:
    """Korpus změněný po auditu neprojde."""
    assert CorpusGate(_Corpus("d" * 64, _record(AUDITED))).run(PINNED)


def test_invalid_record_is_blocked() -> None:
    """Záznam, který neodpovídá schématu, korpus neatestuje."""
    violations = CorpusGate(_Corpus(AUDITED, invalid_record=True)).run(PINNED)

    assert violations == ("Záznam auditu neodpovídá schématu.",)


def test_record_against_other_reference_is_blocked() -> None:
    """Otisky připnuté zvenku rozhodují, proti čemu musel audit běžet."""
    pinned = PINNED | {"radiology.csv.gz": "f" * 64}

    assert CorpusGate(_Corpus(AUDITED, _record(AUDITED))).run(pinned)


def test_missing_provenance_is_blocked() -> None:
    """Auditovaný korpus bez provenance neprojde."""
    corpus = _Corpus(AUDITED, _record(AUDITED), provenance=None)

    assert CorpusGate(corpus).run(PINNED) == ("Korpus nemá záznam provenance.",)


def test_provenance_of_other_audit_is_blocked() -> None:
    """Provenance s otiskem jiného záznamu auditu neprojde."""
    corpus = _Corpus(AUDITED, _record(AUDITED), provenance=_provenance("8" * 64))

    assert CorpusGate(corpus).run(PINNED)


def test_invalid_provenance_is_blocked() -> None:
    """Záznam provenance, který neodpovídá schématu, neprojde."""
    corpus = _Corpus(AUDITED, _record(AUDITED), invalid_provenance=True)

    assert CorpusGate(corpus).run(PINNED) == ("Záznam provenance neodpovídá schématu.",)
