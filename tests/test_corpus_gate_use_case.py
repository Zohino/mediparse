"""Use case brány nad fakem korpusu: porušení, neplatný záznam a korpus beze změny."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from mediparse.application.corpus_gate import CorpusGate
from mediparse.domain.corpus_audit import (
    NGRAM_SIZE,
    NORMALIZATION,
    AuditRecord,
    InvalidAuditRecordError,
    ReferenceFile,
)

AUDITED = "a" * 64
PINNED = {"discharge.csv.gz": "b" * 64, "radiology.csv.gz": "e" * 64}


@dataclass(frozen=True)
class _Corpus:
    sha256: str | None
    record: AuditRecord | None = None
    invalid_record: bool = False

    def fingerprint(self) -> str | None:
        return self.sha256

    def audit_record(self) -> AuditRecord | None:
        if self.invalid_record:
            raise InvalidAuditRecordError
        return self.record


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
