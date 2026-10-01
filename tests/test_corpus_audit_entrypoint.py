"""Vstupní bod auditu korpusu: čtení MIMIC-IV-Note, výstupní kontrakt a odmítnutí mimo kontrolované prostředí."""

from __future__ import annotations

import csv
import gzip
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

if TYPE_CHECKING:
    from collections.abc import Sequence
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
RADIOLOGY_ROWS = [("10000033", "an unrelated radiology report")]


def _corpus(tmp_path: Path, notes: dict[str, str]) -> Path:
    repository = tmp_path / "repo"
    (repository / ".git").mkdir(parents=True)
    root = repository / "resources" / "synthetic"
    for name, text in notes.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _reference(path: Path, rows: list[tuple[str, str]]) -> Path:
    with gzip.open(path, mode="wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        writer.writerows(
            [f"{subject}-DS-1", subject, "1", "DS", "1", "", "", text]
            for subject, text in rows
        )
    return path


def _references(tmp_path: Path, rows: list[tuple[str, str]]) -> tuple[Path, Path]:
    return (
        _reference(tmp_path / "discharge.csv.gz", rows),
        _reference(tmp_path / "radiology.csv.gz", RADIOLOGY_ROWS),
    )


def _pin(tmp_path: Path, references: Sequence[Path]) -> Path:
    tables = tmp_path / "mimic_tables.json"
    entries = [
        {
            "url": f"https://physionet.org/files/mimic-iv-note/2.2/note/{path.name}",
            "sha256": MimicReference(path).sha256(),
        }
        for path in references
    ]
    tables.write_text(json.dumps({"mimic_tables": entries}), encoding="utf-8")
    return tables


def _argv(
    corpus: Path, references: Sequence[Path], report: Path, tables: Path
) -> list[str]:
    return [
        "--corpus",
        str(corpus),
        *(f"--reference={path}" for path in references),
        "--tables",
        str(tables),
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
) -> tuple[ExitCode, Path]:
    corpus = _corpus(tmp_path, notes)
    references = _references(tmp_path, rows)
    tables = _pin(tmp_path, references)
    argv = _argv(corpus, references, tmp_path / "report.json", tables)
    return run(argv, environ or {}), corpus


def test_reference_reader_keeps_multiline_quoted_text(tmp_path: Path) -> None:
    """Víceřádkový text v uvozovkách se přečte jako jedna zpráva beze změny."""
    text = 'first line\nsecond "quoted" line, with comma'

    path = _reference(tmp_path / "discharge.csv.gz", [("10000032", text)])

    notes = list(MimicReference(path).notes())

    assert notes == [ReferenceNote(subject_id="10000032", text=text)]


def test_clean_corpus_gets_audit_record(tmp_path: Path) -> None:
    """Čistý audit zapíše záznam s otiskem právě auditovaného korpusu."""
    code, corpus = _audit(
        tmp_path, {NOTE: "a short synthetic note"}, [("10000032", SENTENCE)]
    )

    record = CorpusDirectory(corpus).audit_record()

    assert code == ExitCode.OK
    assert record is not None
    assert record.corpus_sha256 == CorpusDirectory(corpus).fingerprint()
    assert [file.name for file in record.reference] == [
        "discharge.csv.gz",
        "radiology.csv.gz",
    ]
    assert record.reference[0].rows == 1
    assert (
        record.reference[0].sha256
        == MimicReference(tmp_path / "discharge.csv.gz").sha256()
    )


def test_overlap_blocks_record_and_never_prints_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Shoda zablokuje záznam; výstup nese jen počty a note_id, report jen pozice."""
    code, corpus = _audit(tmp_path, {NOTE: SENTENCE}, [("10000032", SENTENCE)])

    output = capsys.readouterr()
    report = (tmp_path / "report.json").read_text(encoding="utf-8")

    assert code == ExitCode.BLOCKED
    assert not (corpus / RECORD_NAME).exists()
    assert NOTE in output.out
    assert "lighthouse" not in output.out + output.err + report
    assert json.loads(report)["positions"]


def test_subject_collision_blocks_record(tmp_path: Path) -> None:
    """Syntetický subject_id, který existuje v referenci, audit neprojde."""
    code, corpus = _audit(
        tmp_path, {NOTE: "a short synthetic note"}, [("90000001", "unrelated text")]
    )

    assert code == ExitCode.BLOCKED
    assert not (corpus / RECORD_NAME).exists()


def test_empty_corpus_is_refused(tmp_path: Path) -> None:
    """Prázdný korpus nemá co auditovat a záznam nevznikne."""
    code, corpus = _audit(tmp_path, {}, [("10000032", SENTENCE)])

    assert code == ExitCode.REFUSED
    assert not (corpus / RECORD_NAME).exists()


@pytest.mark.parametrize(
    "environ", [{"CI": "true"}, {"CLAUDECODE": "1"}], ids=["ci", "claude-code"]
)
def test_audit_refuses_ci_and_ai_session(
    tmp_path: Path, environ: dict[str, str]
) -> None:
    """Audit čte MIMIC, proto odmítne veřejné CI i relaci hostovaného modelu."""
    code, corpus = _audit(tmp_path, {NOTE: SENTENCE}, [("10000032", SENTENCE)], environ)

    assert code == ExitCode.REFUSED
    assert not (corpus / RECORD_NAME).exists()
    assert not (tmp_path / "report.json").exists()


def test_invalid_note_id_is_refused(tmp_path: Path) -> None:
    """Soubor, jehož jméno není note_id, audit odmítne kódem 2, ne tracebackem."""
    code, corpus = _audit(
        tmp_path, {"en/not-a-note.txt": SENTENCE}, [("10000032", SENTENCE)]
    )

    assert code == ExitCode.REFUSED
    assert not (corpus / RECORD_NAME).exists()
    assert not (tmp_path / "report.json").exists()


def test_missing_reference_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Chybějící referenční soubor audit odmítne jako chybějící, ne jako neshodu otisku."""
    corpus = _corpus(tmp_path, {NOTE: "a short synthetic note"})
    references = _references(tmp_path, [("10000032", SENTENCE)])
    tables = _pin(tmp_path, references)
    argv = _argv(
        corpus,
        (tmp_path / "missing.csv.gz", references[1]),
        tmp_path / "r.json",
        tables,
    )

    assert run(argv, {}) == ExitCode.REFUSED
    assert "neexistují" in capsys.readouterr().err


def test_reference_changed_after_pin_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Soubor, který se liší od otisku v configu, audit odmítne dřív, než vznikne záznam či report."""
    corpus = _corpus(tmp_path, {NOTE: "a short synthetic note"})
    references = _references(tmp_path, [("10000032", SENTENCE)])
    tables = _pin(tmp_path, references)
    _reference(references[0], [("10000032", "a different reference text")])

    code = run(_argv(corpus, references, tmp_path / "report.json", tables), {})

    assert code == ExitCode.REFUSED
    assert "discharge.csv.gz" in capsys.readouterr().err
    assert not (corpus / RECORD_NAME).exists()
    assert not (tmp_path / "report.json").exists()


def test_audit_against_part_of_reference_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Audit jen proti discharge reference z configu nepokryje, a proto neproběhne."""
    corpus = _corpus(tmp_path, {NOTE: "a short synthetic note"})
    references = _references(tmp_path, [("10000032", SENTENCE)])
    tables = _pin(tmp_path, references)

    code = run(_argv(corpus, references[:1], tmp_path / "report.json", tables), {})

    assert code == ExitCode.REFUSED
    assert "radiology.csv.gz" in capsys.readouterr().err
    assert not (corpus / RECORD_NAME).exists()


def test_empty_reference_is_refused(tmp_path: Path) -> None:
    """Reference bez jediné zprávy by atestovala nic, záznam nevznikne."""
    code, corpus = _audit(tmp_path, {NOTE: "a short synthetic note"}, [])

    assert code == ExitCode.REFUSED
    assert not (corpus / RECORD_NAME).exists()


def test_corpus_outside_repository_is_refused(tmp_path: Path) -> None:
    """Bez repozitáře nejde ověřit, že report leží mimo něj."""
    root = tmp_path / "loose"
    (root / "en").mkdir(parents=True)
    (root / NOTE).write_text("a short synthetic note", encoding="utf-8")
    references = _references(tmp_path, [("10000032", SENTENCE)])
    tables = _pin(tmp_path, references)

    code = run(_argv(root, references, tmp_path / "r.json", tables), {})

    assert code == ExitCode.REFUSED


def test_report_inside_repository_is_refused(tmp_path: Path) -> None:
    """Report s pozicemi shod nesmí vzniknout uvnitř repozitáře."""
    corpus = _corpus(tmp_path, {NOTE: "a short synthetic note"})
    report = corpus / "r.json"
    references = _references(tmp_path, [])
    tables = _pin(tmp_path, references)

    assert run(_argv(corpus, references, report, tables), {}) == ExitCode.REFUSED


def test_malformed_commit_is_rejected(tmp_path: Path) -> None:
    """Commit nástroje musí být celý hash, zkratka ani jiný text záznam nepodepíše."""
    corpus = _corpus(tmp_path, {NOTE: "a short synthetic note"})
    references = _references(tmp_path, [])
    tables = _pin(tmp_path, references)
    argv = [
        *_argv(corpus, references, tmp_path / "r.json", tables)[:-1],
        "abc123",
    ]

    with pytest.raises(SystemExit) as raised:
        run(argv, {})

    assert raised.value.code == ExitCode.REFUSED
