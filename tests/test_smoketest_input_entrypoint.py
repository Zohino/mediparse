"""Vstupní bod kroku prepare_smoketest_subset: tabulka z plánů a korpusu, odmítnutí nepárového korpusu."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pyarrow.parquet as pq

from mediparse.entrypoints import smoketest_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.plans_file import PlansFile

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from tests.conftest import PlannedNote


def _plans(root: Path, planned_note: PlannedNote) -> Path:
    plans = root / "plans.jsonl"
    PlansFile(plans).save([planned_note.plan])
    return plans


def test_writes_input_table(
    tmp_path: Path, planned_note: PlannedNote, caplog: pytest.LogCaptureFixture
) -> None:
    """Plány a zprávy z cest pravidla dají tabulku; zápis se potvrdí v logu."""
    plans = _plans(tmp_path, planned_note)
    (tmp_path / "en").mkdir()
    (tmp_path / "en" / f"{planned_note.plan.note_id}.txt").write_text(
        planned_note.text, encoding="utf-8"
    )
    output = tmp_path / "build" / "notes.parquet"

    with caplog.at_level(logging.INFO, logger="mediparse"):
        code = smoketest_input.run(plans=plans, corpus=tmp_path, output=output)

    assert code is ExitCode.OK
    assert pq.read_table(output).column("note_id").to_pylist() == [
        planned_note.plan.note_id
    ]
    assert f"1 zpráv v {output}" in caplog.text


def test_unpaired_corpus_is_refused(
    tmp_path: Path, planned_note: PlannedNote, caplog: pytest.LogCaptureFixture
) -> None:
    """Plán bez zprávy krok odmítne a tabulka nevznikne."""
    plans = _plans(tmp_path, planned_note)
    output = tmp_path / "notes.parquet"

    code = smoketest_input.run(plans=plans, corpus=tmp_path, output=output)

    assert code is ExitCode.REFUSED
    assert not output.exists()
    assert "Plány bez zprávy v jazyce en" in caplog.text


def test_missing_plans_are_refused(tmp_path: Path) -> None:
    """Chybějící soubor plánů je chyba vstupu, ne pád s tracebackem."""
    output = tmp_path / "notes.parquet"
    code = smoketest_input.run(
        plans=tmp_path / "missing.jsonl",
        corpus=tmp_path,
        output=output,
    )

    assert code is ExitCode.REFUSED
    assert not output.exists()
