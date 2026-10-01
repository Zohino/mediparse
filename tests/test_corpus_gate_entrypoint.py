"""Brána syntetického korpusu: commitnout ani pushnout jde jen korpus, který prošel auditem."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mediparse.domain.corpus_audit import InvalidAuditRecordError
from mediparse.entrypoints import corpus_audit
from mediparse.entrypoints.corpus_gate import run
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.mimic_reference import MimicReference
from mediparse.infrastructure.synthetic_corpus import RECORD_NAME, CorpusDirectory

if TYPE_CHECKING:
    from collections.abc import Sequence

REPOSITORY = Path(__file__).parents[1]
REPOSITORY_CORPUS = REPOSITORY / "resources" / "synthetic"
REPOSITORY_TABLES = REPOSITORY / "config" / "mimic_tables.json"
NOTE = "en/90000001-DS-1.txt"
TABLES = "mimic_tables.json"
REFERENCE_FILES = ("discharge.csv.gz", "radiology.csv.gz")
HOSP = {
    "url": "https://physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz",
    "sha256": "f" * 64,
}


def _reference(path: Path, subject_id: str) -> Path:
    with gzip.open(path, mode="wt", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows([
            ["note_id", "subject_id", "text"],
            [f"{subject_id}-DS-1", subject_id, f"reference text of {path.name}"],
        ])
    return path


def _pin(tables: Path, references: Sequence[Path]) -> None:
    entries = [
        {
            "url": f"https://physionet.org/files/mimic-iv-note/2.2/note/{path.name}",
            "sha256": MimicReference(path).sha256(),
        }
        for path in references
    ]
    entries.append(HOSP)
    tables.write_text(json.dumps({"mimic_tables": entries}), encoding="utf-8")


def _audited_corpus(tmp_path: Path) -> Path:
    root = tmp_path / "repo" / "resources" / "synthetic"
    (root / "en").mkdir(parents=True)
    (tmp_path / "repo" / ".git").mkdir()
    (root / NOTE).write_text("a short synthetic note", encoding="utf-8")
    references = [
        _reference(tmp_path / name, str(subject_id))
        for subject_id, name in enumerate(REFERENCE_FILES, start=1)
    ]
    _pin(tmp_path / TABLES, references)
    argv = [
        "--corpus",
        str(root),
        *(f"--reference={path}" for path in references),
        "--tables",
        str(tmp_path / TABLES),
        "--report",
        str(tmp_path / "report.json"),
        "--commit",
        "c" * 40,
    ]
    assert corpus_audit.run(argv, {}) == ExitCode.OK
    return root


def _gate(root: Path, tables: Path) -> int:
    return run(["--corpus", str(root), "--tables", str(tables)])


def test_gate_passes_without_corpus(tmp_path: Path) -> None:
    """Dokud korpus neexistuje, brána commit ani push neblokuje."""
    assert _gate(tmp_path / "missing", REPOSITORY_TABLES) == ExitCode.OK


def test_gate_rejects_unaudited_corpus(tmp_path: Path) -> None:
    """Korpus bez záznamu auditu neprojde."""
    root = _audited_corpus(tmp_path)
    (root / RECORD_NAME).unlink()

    assert _gate(root, tmp_path / TABLES) == ExitCode.BLOCKED


def test_gate_passes_audited_corpus(tmp_path: Path) -> None:
    """Korpus beze změny od auditu projde."""
    assert _gate(_audited_corpus(tmp_path), tmp_path / TABLES) == ExitCode.OK


def test_gate_rejects_corpus_changed_after_audit(tmp_path: Path) -> None:
    """Přegenerovaná zpráva bez nového auditu neprojde."""
    root = _audited_corpus(tmp_path)
    (root / NOTE).write_text("a regenerated synthetic note", encoding="utf-8")

    assert _gate(root, tmp_path / TABLES) == ExitCode.BLOCKED


def test_gate_rejects_added_note(tmp_path: Path) -> None:
    """Nová zpráva přidaná po auditu otisk také změní."""
    root = _audited_corpus(tmp_path)
    (root / "en" / "90000002-DS-1.txt").write_text("another note", encoding="utf-8")

    assert _gate(root, tmp_path / TABLES) == ExitCode.BLOCKED


def test_gate_rejects_invalid_record(tmp_path: Path) -> None:
    """Záznam, který neodpovídá schématu, korpus neatestuje."""
    root = _audited_corpus(tmp_path)
    (root / RECORD_NAME).write_text("{}", encoding="utf-8")

    assert _gate(root, tmp_path / TABLES) == ExitCode.BLOCKED


def test_gate_rejects_record_against_part_of_reference(tmp_path: Path) -> None:
    """Záznam jen proti discharge (ručně upravený nebo ze starší verze auditu) korpus neatestuje."""
    root = _audited_corpus(tmp_path)
    path = root / RECORD_NAME
    record = json.loads(path.read_text(encoding="utf-8"))
    record["reference"] = record["reference"][:1]
    path.write_text(json.dumps(record), encoding="utf-8")

    assert _gate(root, tmp_path / TABLES) == ExitCode.BLOCKED


def test_invalid_record_surfaces_as_domain_error(tmp_path: Path) -> None:
    """Adaptér přeloží chybu schématu na doménovou výjimku, pydantic za něj neprosákne."""
    (tmp_path / RECORD_NAME).write_text("{}", encoding="utf-8")

    with pytest.raises(InvalidAuditRecordError):
        CorpusDirectory(tmp_path).audit_record()


def test_repository_corpus_passes_gate() -> None:
    """Korpus v repu odpovídá svému auditu i připnuté referenci — jinak neprojde CI."""
    assert _gate(REPOSITORY_CORPUS, REPOSITORY_TABLES) == ExitCode.OK
