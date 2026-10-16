"""Příkaz kontrol souladu: vypíše zprávy k přegenerování a korpus v repu hlídá v CI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.entrypoints.corpus_consistency import run
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.plans_file import PlansFile
from tests.support import REPOSITORY_CONFIG, REPOSITORY_CORPUS, REPOSITORY_PLANS

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from tests.conftest import AuditFiles, PlannedNote


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
    audit_files: AuditFiles,
    planned_note: PlannedNote,
    text: str,
    czech: str | None = None,
) -> tuple[Path, Path]:
    plans = audit_files.root / "plans.jsonl"
    PlansFile(plans).save((planned_note.plan,))
    name = f"{planned_note.plan.note_id}.txt"
    files = {f"en/{name}": text} | ({f"cs/{name}": czech} if czech else {})
    return audit_files.corpus(files), plans


def test_matching_notes_leave_only_corpus_findings(
    audit_files: AuditFiles,
    planned_note: PlannedNote,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Zpráva v souladu s plánem se k přegenerování nehlásí; korpus z jediné zprávy model nesplní."""
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


def test_repository_corpus_matches_its_plans(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Zprávy v repu odpovídají plánům a výstup to hlásí jako soulad, ne jako shodu auditu."""
    code = _run(REPOSITORY_CORPUS, REPOSITORY_PLANS)

    assert code == ExitCode.OK
    assert (
        "Kontroly souladu: korpus odpovídá plánům i modelu." in capsys.readouterr().out
    )


def test_notices_alone_keep_exit_ok(
    audit_files: AuditFiles,
    planned_note: PlannedNote,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nesoulad počtu ___ v překladu se vypíše jako upozornění a exit zůstane OK."""
    monkeypatch.setattr(
        "mediparse.application.corpus_consistency.corpus_violations",
        lambda *_: (),
    )
    czech = planned_note.text.replace("___", "x", 1)

    code = _run(*_corpus(audit_files, planned_note, planned_note.text, czech))

    output = capsys.readouterr().out
    assert code == ExitCode.OK
    assert f"Upozornění: cs/{planned_note.plan.note_id}" in output
