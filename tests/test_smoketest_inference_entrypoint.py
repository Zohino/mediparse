"""Vstupní bod kroku infer_smoketest_model: uložený model nad odloženými zprávami."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Final, NamedTuple

from mediparse.domain.run_manifest import SourceRevision
from mediparse.entrypoints import smoketest_inference, smoketest_training
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.prediction_table import ParquetPredictionTable
from tests.support import separable_notes

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

PATIENTS: Final = 12


class Trained(NamedTuple):
    """Cesty k výstupům tréninku."""

    notes: Path
    model: Path
    predictions: Path


def _revision() -> SourceRevision:
    return SourceRevision(commit="a" * 40, dirty=False)


def _train(root: Path) -> Trained:
    notes = root / "notes.parquet"
    ParquetNoteTable(notes).write(separable_notes(PATIENTS))
    config = root / "config.json"
    config.write_text(
        json.dumps({
            "row_id": "row",
            "diagnosis": "diabetes",
            "seed": 0,
            "folds": 2,
            "regularization": 1.0,
        }),
        encoding="utf-8",
    )
    trained = Trained(notes, root / "model.skops", root / "predictions.parquet")
    code = smoketest_training.run(
        notes=notes,
        config=config,
        model=trained.model,
        predictions=trained.predictions,
        manifest=root / "run_manifest.json",
        revision=_revision,
    )
    assert code is ExitCode.OK
    return trained


def test_writes_positive_and_negative_demo(tmp_path: Path) -> None:
    """Uložený model dá shodné predikce a demo.parquet má pozitivní a negativní ukázku."""
    trained = _train(tmp_path)
    demo = tmp_path / "build" / "demo.parquet"

    code = smoketest_inference.run(
        notes=trained.notes,
        model=trained.model,
        predictions=trained.predictions,
        demo=demo,
    )

    rows = ParquetPredictionTable(demo).read()
    held_out = {
        item.note_id for item in ParquetPredictionTable(trained.predictions).read()
    }
    assert code is ExitCode.OK
    assert [row.truth for row in rows] == [True, False]
    assert {row.note_id for row in rows} <= held_out


def test_missing_model_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Chybějící model krok odmítne s hláškou a bez tracebacku."""
    trained = _train(tmp_path)
    demo = tmp_path / "demo.parquet"

    code = smoketest_inference.run(
        notes=trained.notes,
        model=tmp_path / "none.skops",
        predictions=trained.predictions,
        demo=demo,
    )

    assert code is ExitCode.REFUSED
    assert not demo.exists()
    assert "none.skops" in caplog.text
    assert "Traceback" not in caplog.text


def test_missing_predictions_are_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Chybějící predikce krok odmítnou s hláškou a bez tracebacku."""
    trained = _train(tmp_path)
    demo = tmp_path / "demo.parquet"

    code = smoketest_inference.run(
        notes=trained.notes,
        model=trained.model,
        predictions=tmp_path / "none.parquet",
        demo=demo,
    )

    assert code is ExitCode.REFUSED
    assert not demo.exists()
    assert "none.parquet" in caplog.text
    assert "Traceback" not in caplog.text


def test_tampered_predictions_are_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Predikce, které uložený model nereprodukuje, krok odmítne jako neshodu."""
    trained = _train(tmp_path)
    table = ParquetPredictionTable(trained.predictions)
    first, *rest = table.read()
    table.write([replace(first, predicted=not first.predicted), *rest])
    demo = tmp_path / "demo.parquet"

    code = smoketest_inference.run(
        notes=trained.notes,
        model=trained.model,
        predictions=trained.predictions,
        demo=demo,
    )

    assert code is ExitCode.REFUSED
    assert not demo.exists()
    assert first.note_id in caplog.text
    assert "Neshoda" in caplog.text
