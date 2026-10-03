"""Vstupní bod auditu korpusu: čtení MIMIC-IV-Note, výstupní kontrakt a odmítnutí mimo kontrolované prostředí."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from mediparse.domain.corpus_audit import ReferenceNote
from mediparse.entrypoints.corpus_audit import run
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.mimic_reference import MimicReference
from mediparse.infrastructure.synthetic_corpus import (
    RECORD_NAME,
    CorpusDirectory,
)
from tests.support import NOTE, SENTENCE, SHORT_NOTE

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import AuditFiles, Rows


def _audit(
    files: AuditFiles,
    notes: dict[str, str],
    rows: Rows,
    environ: dict[str, str] | None = None,
) -> tuple[ExitCode, Path]:
    corpus = files.corpus(notes)
    references = files.pinned_references(rows)
    argv = files.audit_argv(corpus, references, files.root / "report.json")
    return run(argv, environ or {}), corpus


def test_reference_reader_keeps_multiline_quoted_text(audit_files: AuditFiles) -> None:
    """Víceřádkový text v uvozovkách se přečte jako jedna zpráva beze změny."""
    text = 'first line\nsecond "quoted" line, with comma'

    path = audit_files.reference("discharge.csv.gz", [("10000032", text)])

    notes = list(MimicReference(path).notes())

    assert notes == [ReferenceNote(subject_id="10000032", text=text)]


def test_clean_corpus_gets_audit_record(audit_files: AuditFiles) -> None:
    """Čistý audit zapíše záznam s otiskem právě auditovaného korpusu."""
    code, corpus = _audit(audit_files, SHORT_NOTE, [("10000032", SENTENCE)])

    record = CorpusDirectory(corpus).audit_record()

    assert code == ExitCode.OK
    assert record is not None
    assert record.corpus_sha256 == CorpusDirectory(corpus).fingerprint()
    assert [file.name for file in record.reference] == [
        "discharge.csv.gz",
        "radiology.csv.gz",
    ]
    assert record.reference[0].rows == 1
    discharge = MimicReference(audit_files.root / "discharge.csv.gz")
    assert record.reference[0].sha256 == discharge.sha256()


def test_overlap_blocks_record_and_never_prints_text(
    audit_files: AuditFiles, capsys: pytest.CaptureFixture[str]
) -> None:
    """Shoda zablokuje záznam; výstup nese jen počty a note_id, report jen pozice."""
    code, corpus = _audit(audit_files, {NOTE: SENTENCE}, [("10000032", SENTENCE)])

    output = capsys.readouterr()
    report = (audit_files.root / "report.json").read_text(encoding="utf-8")

    assert code == ExitCode.BLOCKED
    assert not (corpus / RECORD_NAME).exists()
    assert NOTE in output.out
    assert "lighthouse" not in output.out + output.err + report
    assert json.loads(report)["positions"]


def test_subject_collision_blocks_record(audit_files: AuditFiles) -> None:
    """Syntetický subject_id, který existuje v referenci, audit neprojde."""
    code, corpus = _audit(audit_files, SHORT_NOTE, [("90000001", "unrelated text")])

    assert code == ExitCode.BLOCKED
    assert not (corpus / RECORD_NAME).exists()


def test_empty_corpus_is_refused(audit_files: AuditFiles) -> None:
    """Prázdný korpus nemá co auditovat a záznam nevznikne."""
    code, corpus = _audit(audit_files, {}, [("10000032", SENTENCE)])

    assert code == ExitCode.REFUSED
    assert not (corpus / RECORD_NAME).exists()


@pytest.mark.parametrize(
    "environ", [{"CI": "true"}, {"CLAUDECODE": "1"}], ids=["ci", "claude-code"]
)
def test_audit_refuses_ci_and_ai_session(
    audit_files: AuditFiles, environ: dict[str, str]
) -> None:
    """Audit čte MIMIC, proto odmítne veřejné CI i relaci hostovaného modelu."""
    code, corpus = _audit(
        audit_files, {NOTE: SENTENCE}, [("10000032", SENTENCE)], environ
    )

    assert code == ExitCode.REFUSED
    assert not (corpus / RECORD_NAME).exists()
    assert not (audit_files.root / "report.json").exists()


def test_invalid_note_id_is_refused(audit_files: AuditFiles) -> None:
    """Soubor, jehož jméno není note_id, audit odmítne kódem 2, ne tracebackem."""
    code, corpus = _audit(
        audit_files, {"en/not-a-note.txt": SENTENCE}, [("10000032", SENTENCE)]
    )

    assert code == ExitCode.REFUSED
    assert not (corpus / RECORD_NAME).exists()
    assert not (audit_files.root / "report.json").exists()


def test_missing_reference_is_refused(
    audit_files: AuditFiles, capsys: pytest.CaptureFixture[str]
) -> None:
    """Chybějící referenční soubor audit odmítne jako chybějící, ne jako neshodu otisku."""
    corpus = audit_files.corpus(SHORT_NOTE)
    _, radiology = audit_files.pinned_references([("10000032", SENTENCE)])
    missing = audit_files.root / "missing.csv.gz"
    argv = audit_files.audit_argv(
        corpus, (missing, radiology), audit_files.root / "r.json"
    )

    assert run(argv, {}) == ExitCode.REFUSED
    assert "neexistují" in capsys.readouterr().err


def test_reference_changed_after_pin_is_refused(
    audit_files: AuditFiles, capsys: pytest.CaptureFixture[str]
) -> None:
    """Soubor, který se liší od otisku v configu, audit odmítne dřív, než vznikne záznam či report."""
    corpus = audit_files.corpus(SHORT_NOTE)
    references = audit_files.pinned_references([("10000032", SENTENCE)])
    audit_files.reference("discharge.csv.gz", [("10000032", "a different text")])
    report = audit_files.root / "report.json"

    code = run(audit_files.audit_argv(corpus, references, report), {})

    assert code == ExitCode.REFUSED
    assert "discharge.csv.gz" in capsys.readouterr().err
    assert not (corpus / RECORD_NAME).exists()
    assert not report.exists()


def test_audit_against_part_of_reference_is_refused(
    audit_files: AuditFiles, capsys: pytest.CaptureFixture[str]
) -> None:
    """Audit jen proti discharge reference z configu nepokryje, a proto neproběhne."""
    corpus = audit_files.corpus(SHORT_NOTE)
    discharge, _ = audit_files.pinned_references([("10000032", SENTENCE)])
    report = audit_files.root / "report.json"

    code = run(audit_files.audit_argv(corpus, (discharge,), report), {})

    assert code == ExitCode.REFUSED
    assert "radiology.csv.gz" in capsys.readouterr().err
    assert not (corpus / RECORD_NAME).exists()


def test_empty_reference_is_refused(audit_files: AuditFiles) -> None:
    """Reference bez jediné zprávy by atestovala nic, záznam nevznikne."""
    code, corpus = _audit(audit_files, SHORT_NOTE, [])

    assert code == ExitCode.REFUSED
    assert not (corpus / RECORD_NAME).exists()


def test_corpus_outside_repository_is_refused(
    audit_files: AuditFiles, tmp_path: Path
) -> None:
    """Bez repozitáře nejde ověřit, že report leží mimo něj."""
    root = tmp_path / "loose"
    (root / "en").mkdir(parents=True)
    (root / NOTE).write_text(SHORT_NOTE[NOTE], encoding="utf-8")
    references = audit_files.pinned_references([("10000032", SENTENCE)])

    code = run(audit_files.audit_argv(root, references, tmp_path / "r.json"), {})

    assert code == ExitCode.REFUSED


def test_report_inside_repository_is_refused(audit_files: AuditFiles) -> None:
    """Report s pozicemi shod nesmí vzniknout uvnitř repozitáře."""
    corpus = audit_files.corpus(SHORT_NOTE)
    references = audit_files.pinned_references([])
    argv = audit_files.audit_argv(corpus, references, corpus / "r.json")

    assert run(argv, {}) == ExitCode.REFUSED


def test_malformed_commit_is_rejected(audit_files: AuditFiles) -> None:
    """Commit nástroje musí být celý hash, zkratka ani jiný text záznam nepodepíše."""
    corpus = audit_files.corpus(SHORT_NOTE)
    references = audit_files.pinned_references([])
    argv = [
        *audit_files.audit_argv(corpus, references, audit_files.root / "r.json")[:-1],
        "abc123",
    ]

    with pytest.raises(SystemExit) as raised:
        run(argv, {})

    assert raised.value.code == ExitCode.REFUSED
