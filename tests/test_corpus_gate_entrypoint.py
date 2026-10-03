"""Brána syntetického korpusu: commitnout ani pushnout jde jen korpus, který prošel auditem."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mediparse.domain.corpus_audit import InvalidAuditRecordError
from mediparse.entrypoints.corpus_gate import run
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.synthetic_corpus import (
    PROVENANCE_NAME,
    RECORD_NAME,
    CorpusDirectory,
)

if TYPE_CHECKING:
    from tests.conftest import AuditFiles

REPOSITORY = Path(__file__).parents[1]
REPOSITORY_CORPUS = REPOSITORY / "resources" / "synthetic"
REPOSITORY_TABLES = REPOSITORY / "config" / "mimic_tables.json"
NOTE = "en/90000001-DS-1.txt"


def _audited_corpus(files: AuditFiles) -> Path:
    return files.audited_corpus({NOTE: "a short synthetic note"})


def _gate(root: Path, tables: Path) -> int:
    return run(["--corpus", str(root), "--tables", str(tables)])


def test_gate_passes_without_corpus(tmp_path: Path) -> None:
    """Dokud korpus neexistuje, brána commit ani push neblokuje."""
    assert _gate(tmp_path / "missing", REPOSITORY_TABLES) == ExitCode.OK


def test_gate_rejects_unaudited_corpus(audit_files: AuditFiles) -> None:
    """Korpus bez záznamu auditu neprojde."""
    root = _audited_corpus(audit_files)
    (root / RECORD_NAME).unlink()

    assert _gate(root, audit_files.tables) == ExitCode.BLOCKED


def test_gate_passes_audited_corpus_with_provenance(audit_files: AuditFiles) -> None:
    """Korpus beze změny od auditu s provenance, která na audit ukazuje, projde."""
    root = audit_files.released_corpus({NOTE: "a short synthetic note"})

    assert _gate(root, audit_files.tables) == ExitCode.OK


def test_gate_rejects_audited_corpus_without_provenance(
    audit_files: AuditFiles,
) -> None:
    """Auditovaný korpus bez záznamu provenance neprojde."""
    assert _gate(_audited_corpus(audit_files), audit_files.tables) == ExitCode.BLOCKED


def test_gate_rejects_provenance_of_previous_audit(audit_files: AuditFiles) -> None:
    """Nový audit bez nově zapsané provenance neprojde."""
    root = audit_files.released_corpus({NOTE: "a short synthetic note"})
    audit_files.audited_corpus({NOTE: "a regenerated synthetic note"})

    assert _gate(root, audit_files.tables) == ExitCode.BLOCKED


def test_gate_rejects_invalid_provenance(audit_files: AuditFiles) -> None:
    """Záznam provenance, který neodpovídá schématu, neprojde."""
    root = audit_files.released_corpus({NOTE: "a short synthetic note"})
    (root / PROVENANCE_NAME).write_text("{}", encoding="utf-8")

    assert _gate(root, audit_files.tables) == ExitCode.BLOCKED


def test_gate_rejects_corpus_changed_after_audit(audit_files: AuditFiles) -> None:
    """Přegenerovaná zpráva bez nového auditu neprojde."""
    root = _audited_corpus(audit_files)
    (root / NOTE).write_text("a regenerated synthetic note", encoding="utf-8")

    assert _gate(root, audit_files.tables) == ExitCode.BLOCKED


def test_gate_rejects_added_note(audit_files: AuditFiles) -> None:
    """Nová zpráva přidaná po auditu otisk také změní."""
    root = _audited_corpus(audit_files)
    (root / "en" / "90000002-DS-1.txt").write_text("another note", encoding="utf-8")

    assert _gate(root, audit_files.tables) == ExitCode.BLOCKED


def test_gate_rejects_invalid_record(audit_files: AuditFiles) -> None:
    """Záznam, který neodpovídá schématu, korpus neatestuje."""
    root = _audited_corpus(audit_files)
    (root / RECORD_NAME).write_text("{}", encoding="utf-8")

    assert _gate(root, audit_files.tables) == ExitCode.BLOCKED


def test_gate_rejects_record_against_part_of_reference(audit_files: AuditFiles) -> None:
    """Záznam jen proti discharge (ručně upravený nebo ze starší verze auditu) korpus neatestuje."""
    root = _audited_corpus(audit_files)
    path = root / RECORD_NAME
    record = json.loads(path.read_text(encoding="utf-8"))
    record["reference"] = record["reference"][:1]
    path.write_text(json.dumps(record), encoding="utf-8")

    assert _gate(root, audit_files.tables) == ExitCode.BLOCKED


def test_invalid_record_surfaces_as_domain_error(tmp_path: Path) -> None:
    """Adaptér přeloží chybu schématu na doménovou výjimku, pydantic za něj neprosákne."""
    (tmp_path / RECORD_NAME).write_text("{}", encoding="utf-8")

    with pytest.raises(InvalidAuditRecordError):
        CorpusDirectory(tmp_path).audit_record()


def test_repository_corpus_passes_gate() -> None:
    """Korpus v repu odpovídá svému auditu, připnuté referenci i provenance — jinak neprojde CI."""
    assert _gate(REPOSITORY_CORPUS, REPOSITORY_TABLES) == ExitCode.OK
