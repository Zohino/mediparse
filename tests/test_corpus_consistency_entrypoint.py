"""Příkaz kontrol shody: vypíše zprávy k přegenerování a korpus v repu hlídá v CI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mediparse.entrypoints.corpus_consistency import run
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.plans_file import PlansFile

if TYPE_CHECKING:
    import pytest

    from tests.conftest import AuditFiles, PlannedNote

REPOSITORY = Path(__file__).parents[1]
REPOSITORY_CORPUS = REPOSITORY / "resources" / "synthetic"
REPOSITORY_CONFIG = REPOSITORY / "config" / "synthetic_plan.json"


def _run(corpus: Path, plans: Path) -> int:
    return run([
        "--corpus",
        str(corpus),
        "--plans",
        str(plans),
        "--config",
        str(REPOSITORY_CONFIG),
    ])


def _corpus(
    audit_files: AuditFiles, planned_note: PlannedNote, text: str
) -> tuple[Path, Path]:
    plans = audit_files.root / "plans.jsonl"
    PlansFile(plans).save((planned_note.plan,))
    root = audit_files.corpus({f"en/{planned_note.plan.note_id}.txt": text})
    return root, plans


def test_matching_notes_leave_only_corpus_findings(
    audit_files: AuditFiles,
    planned_note: PlannedNote,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Zpráva shodná s plánem se k přegenerování nehlásí; korpus z jediné zprávy model nesplní."""
    code = _run(*_corpus(audit_files, planned_note, planned_note.text))

    output = capsys.readouterr().out
    assert code == ExitCode.BLOCKED
    assert "Korpus: " in output
    assert "Zprávy k přegenerování" not in output


def test_violation_names_note_to_regenerate(
    audit_files: AuditFiles,
    planned_note: PlannedNote,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Porušení zablokuje a výstup jmenuje zprávu k přegenerování."""
    text = planned_note.text.replace("Sex: F", "Sex: M")

    code = _run(*_corpus(audit_files, planned_note, text))

    assert code == ExitCode.BLOCKED
    output = capsys.readouterr().out
    assert f"Zprávy k přegenerování: {planned_note.plan.note_id}" in output


def test_repository_corpus_matches_its_plans() -> None:
    """Zprávy v repu odpovídají plánům; dokud žádné nejsou, kontroly nemají co hlásit."""
    assert _run(REPOSITORY_CORPUS, REPOSITORY_CORPUS / "plans.jsonl") == ExitCode.OK
