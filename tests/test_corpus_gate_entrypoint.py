"""Vstupní bod brány nad skutečnými soubory: otisk korpusu, záznamy auditu a provenance a korpus v repu."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mediparse.domain.corpus_audit import InvalidAuditRecordError
from mediparse.domain.corpus_provenance import InvalidProvenanceRecordError
from mediparse.entrypoints.corpus_gate import run
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.synthetic_corpus import (
    PROVENANCE_NAME,
    RECORD_NAME,
    CorpusDirectory,
)
from tests.support import (
    NOTE,
    REPOSITORY_CORPUS,
    REPOSITORY_TABLES,
    SHORT_NOTE,
    translation_record,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tests.conftest import AuditFiles


def _gate(root: Path, tables: Path) -> int:
    return run(["--corpus", str(root), "--tables", str(tables)])


def test_gate_passes_without_corpus(tmp_path: Path) -> None:
    """Dokud korpus neexistuje, brána commit ani push neblokuje."""
    assert _gate(tmp_path / "missing", REPOSITORY_TABLES) == ExitCode.OK


def test_gate_passes_audited_corpus_with_provenance(audit_files: AuditFiles) -> None:
    """Korpus beze změny od auditu s provenance, která na audit ukazuje, projde."""
    root = audit_files.released_corpus(SHORT_NOTE)

    assert _gate(root, audit_files.tables) == ExitCode.OK


def test_gate_requires_translation_provenance_for_czech_corpus(
    audit_files: AuditFiles,
) -> None:
    """Český korpus s provenance bez části translation neprojde, s ní projde."""
    root = audit_files.released_corpus(SHORT_NOTE | {"cs/90000001-DS-1.txt": "text"})
    assert _gate(root, audit_files.tables) == ExitCode.BLOCKED

    corpus = CorpusDirectory(root)
    record = corpus.provenance_record()
    assert record is not None
    corpus.save_provenance(
        record.model_copy(update={"translation": translation_record()})
    )

    assert _gate(root, audit_files.tables) == ExitCode.OK


def test_gate_rejects_corpus_changed_after_audit(audit_files: AuditFiles) -> None:
    """Přegenerovaná zpráva bez nového auditu neprojde."""
    root = audit_files.audited_corpus(SHORT_NOTE)
    (root / NOTE).write_text("a regenerated synthetic note", encoding="utf-8")

    assert _gate(root, audit_files.tables) == ExitCode.BLOCKED


def test_gate_rejects_added_note(audit_files: AuditFiles) -> None:
    """Nová zpráva přidaná po auditu otisk také změní."""
    root = audit_files.audited_corpus(SHORT_NOTE)
    (root / "en" / "90000002-DS-1.txt").write_text("another note", encoding="utf-8")

    assert _gate(root, audit_files.tables) == ExitCode.BLOCKED


@pytest.mark.parametrize(
    ("name", "read", "error"),
    [
        (RECORD_NAME, CorpusDirectory.audit_record, InvalidAuditRecordError),
        (
            PROVENANCE_NAME,
            CorpusDirectory.provenance_record,
            InvalidProvenanceRecordError,
        ),
    ],
    ids=["audit", "provenance"],
)
def test_invalid_records_surface_as_domain_errors(
    tmp_path: Path,
    name: str,
    read: Callable[[CorpusDirectory], object],
    error: type[ValueError],
) -> None:
    """Adaptér přeloží chybu schématu záznamu na doménovou výjimku, pydantic za něj neprosákne."""
    (tmp_path / name).write_text("{}", encoding="utf-8")

    with pytest.raises(error):
        read(CorpusDirectory(tmp_path))


@pytest.mark.parametrize(
    "read",
    [CorpusDirectory.audit_record, CorpusDirectory.provenance_record],
    ids=["audit", "provenance"],
)
def test_missing_records_read_as_none(
    tmp_path: Path, read: Callable[[CorpusDirectory], object]
) -> None:
    """Chybějící záznam adaptér vrátí jako None, brána ho pak hlásí, místo aby spadla."""
    assert read(CorpusDirectory(tmp_path)) is None


def test_repository_corpus_passes_gate() -> None:
    """Korpus v repu odpovídá svému auditu, připnuté referenci i provenance — jinak neprojde CI."""
    assert _gate(REPOSITORY_CORPUS, REPOSITORY_TABLES) == ExitCode.OK
