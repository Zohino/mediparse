"""Lokální audit korpusu: čtení MIMIC-IV-Note, výstupní kontrakt a odmítnutí mimo kontrolované prostředí."""

from __future__ import annotations

import csv
import gzip
import json
from typing import TYPE_CHECKING

import pytest

from mediparse.domain.corpus_audit import ReferenceNote
from mediparse.infrastructure.corpus_audit_cli import Exit, main, run_audit
from mediparse.infrastructure.mimic_reference import reference_notes
from mediparse.infrastructure.synthetic_corpus import (
    RECORD_NAME,
    corpus_sha256,
    load_record,
)

if TYPE_CHECKING:
    from pathlib import Path

SENTENCE = (
    "the old lighthouse keeper counted seven gulls before the storm "
    "reached the northern harbor wall"
)
NOTE = "en/90000001-DS-1.txt"
COMMIT = "c" * 40
HEADER = [
    "note_id",
    "subject_id",
    "hadm_id",
    "note_type",
    "note_seq",
    "charttime",
    "storetime",
    "text",
]


def _corpus(tmp_path: Path, notes: dict[str, str]) -> Path:
    repository = tmp_path / "repo"
    (repository / ".git").mkdir(parents=True)
    root = repository / "resources" / "synthetic"
    for name, text in notes.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _reference(tmp_path: Path, rows: list[tuple[str, str]]) -> Path:
    path = tmp_path / "discharge.csv.gz"
    with gzip.open(path, mode="wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        writer.writerows(
            [f"{subject}-DS-1", subject, "1", "DS", "1", "", "", text]
            for subject, text in rows
        )
    return path


def _argv(corpus: Path, reference: Path, report: Path) -> list[str]:
    return [
        "--corpus",
        str(corpus),
        "--reference",
        str(reference),
        "--report",
        str(report),
        "--commit",
        COMMIT,
    ]


def _audit(
    tmp_path: Path,
    notes: dict[str, str],
    rows: list[tuple[str, str]],
    environ: dict[str, str] | None = None,
) -> tuple[Exit, Path]:
    corpus = _corpus(tmp_path, notes)
    reference = _reference(tmp_path, rows)
    code = run_audit(
        corpus, [reference], tmp_path / "report.json", COMMIT, environ or {}
    )
    return code, corpus


def test_reference_reader_keeps_multiline_quoted_text(tmp_path: Path) -> None:
    """Víceřádkový text v uvozovkách se přečte jako jedna zpráva beze změny."""
    text = 'first line\nsecond "quoted" line, with comma'

    notes = list(reference_notes(_reference(tmp_path, [("10000032", text)])))

    assert notes == [ReferenceNote(subject_id="10000032", text=text)]


def test_clean_corpus_gets_audit_record(tmp_path: Path) -> None:
    """Čistý audit zapíše záznam s otiskem právě auditovaného korpusu."""
    code, corpus = _audit(
        tmp_path, {NOTE: "a short synthetic note"}, [("10000032", SENTENCE)]
    )

    record = load_record(corpus)

    assert code == Exit.CLEAN
    assert record is not None
    assert record.corpus_sha256 == corpus_sha256(corpus)
    assert record.reference[0].rows == 1


def test_overlap_blocks_record_and_never_prints_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Shoda zablokuje záznam; výstup nese jen počty a note_id, report jen pozice."""
    code, corpus = _audit(tmp_path, {NOTE: SENTENCE}, [("10000032", SENTENCE)])

    output = capsys.readouterr()
    report = (tmp_path / "report.json").read_text(encoding="utf-8")

    assert code == Exit.FOUND
    assert not (corpus / RECORD_NAME).exists()
    assert NOTE in output.out
    assert "lighthouse" not in output.out + output.err + report
    assert json.loads(report)["positions"]


def test_subject_collision_blocks_record(tmp_path: Path) -> None:
    """Syntetický subject_id, který existuje v referenci, audit neprojde."""
    code, corpus = _audit(
        tmp_path, {NOTE: "a short synthetic note"}, [("90000001", "unrelated text")]
    )

    assert code == Exit.FOUND
    assert not (corpus / RECORD_NAME).exists()


def test_empty_corpus_is_refused(tmp_path: Path) -> None:
    """Prázdný korpus nemá co auditovat a záznam nevznikne."""
    code, corpus = _audit(tmp_path, {}, [("10000032", SENTENCE)])

    assert code == Exit.REFUSED
    assert not (corpus / RECORD_NAME).exists()


@pytest.mark.parametrize(
    "environ", [{"CI": "true"}, {"CLAUDECODE": "1"}], ids=["ci", "claude-code"]
)
def test_audit_refuses_ci_and_ai_session(
    tmp_path: Path, environ: dict[str, str]
) -> None:
    """Audit čte MIMIC, proto odmítne veřejné CI i relaci hostovaného modelu."""
    code, corpus = _audit(tmp_path, {NOTE: SENTENCE}, [("10000032", SENTENCE)], environ)

    assert code == Exit.REFUSED
    assert not (corpus / RECORD_NAME).exists()
    assert not (tmp_path / "report.json").exists()


def test_missing_reference_is_refused(tmp_path: Path) -> None:
    """Chybějící referenční soubor audit odmítne, místo aby spadl při čtení."""
    corpus = _corpus(tmp_path, {NOTE: "a short synthetic note"})

    code = run_audit(
        corpus, [tmp_path / "missing.csv.gz"], tmp_path / "r.json", COMMIT, {}
    )

    assert code == Exit.REFUSED


def test_empty_reference_is_refused(tmp_path: Path) -> None:
    """Reference bez jediné zprávy by atestovala nic, záznam nevznikne."""
    code, corpus = _audit(tmp_path, {NOTE: "a short synthetic note"}, [])

    assert code == Exit.REFUSED
    assert not (corpus / RECORD_NAME).exists()


def test_corpus_outside_repository_is_refused(tmp_path: Path) -> None:
    """Bez repozitáře nejde ověřit, že report leží mimo něj."""
    root = tmp_path / "loose"
    (root / "en").mkdir(parents=True)
    (root / NOTE).write_text("a short synthetic note", encoding="utf-8")
    reference = _reference(tmp_path, [("10000032", SENTENCE)])

    code = run_audit(root, [reference], tmp_path / "r.json", COMMIT, {})

    assert code == Exit.REFUSED


def test_report_inside_repository_is_refused(tmp_path: Path) -> None:
    """Report s pozicemi shod nesmí vzniknout uvnitř repozitáře."""
    corpus = _corpus(tmp_path, {NOTE: "a short synthetic note"})
    report = corpus / "r.json"

    assert main(_argv(corpus, _reference(tmp_path, []), report), {}) == Exit.REFUSED


def test_malformed_commit_is_rejected(tmp_path: Path) -> None:
    """Commit nástroje musí být celý hash, zkratka ani jiný text záznam nepodepíše."""
    corpus = _corpus(tmp_path, {NOTE: "a short synthetic note"})
    argv = [
        *_argv(corpus, _reference(tmp_path, []), tmp_path / "r.json")[:-1],
        "abc123",
    ]

    with pytest.raises(SystemExit) as raised:
        main(argv, {})

    assert raised.value.code == Exit.REFUSED
