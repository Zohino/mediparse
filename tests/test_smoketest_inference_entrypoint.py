"""Vstupní bod kroku infer_smoketest_model: uložený model nad odloženými zprávami."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING, Final, NamedTuple

from mediparse.entrypoints import smoketest_inference, smoketest_training
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.prediction_table import ParquetPredictionTable
from tests.model_support import (
    SEPARABLE_VOCABULARY,
    SPELLING_VOCABULARY,
    sectioned_notes,
    separable_notes,
    training_config,
    write_split,
)
from tests.support import REPOSITORY_SECTION_HEADERS, fixed_revision

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

PATIENTS: Final = 12


class Trained(NamedTuple):
    """Cesty k výstupům tréninku."""

    notes: Path
    config: Path
    model: Path
    predictions: Path


def _train(
    root: Path,
    *,
    strip: bool = False,
    sections: bool = False,
    normalize: bool = False,
) -> Trained:
    notes = root / "notes.parquet"
    corpus = (
        sectioned_notes(PATIENTS)
        if sections
        else separable_notes(
            PATIENTS, SPELLING_VOCABULARY if normalize else SEPARABLE_VOCABULARY
        )
    )
    ParquetNoteTable(notes).write(corpus)
    config = root / "config.json"
    config.write_text(
        json.dumps(
            training_config(
                row_id="row",
                diagnosis="diabetes",
                strip_diagnostic_sections=strip,
                normalize_clinical_text=normalize,
            )
        ),
        encoding="utf-8",
    )
    split = root / "splits.parquet"
    write_split(split, corpus)
    trained = Trained(notes, config, root / "model.skops", root / "predictions.parquet")
    code = smoketest_training.run(
        notes=notes,
        config=config,
        lexicon=REPOSITORY_SECTION_HEADERS,
        split=split,
        model=trained.model,
        predictions=trained.predictions,
        manifest=root / "run_manifest.json",
        revision=fixed_revision,
    )
    assert code is ExitCode.OK
    return trained


def test_writes_positive_and_negative_demo(tmp_path: Path) -> None:
    """Uložený model dá stejné predikce a demo.parquet má pozitivní a negativní ukázku."""
    trained = _train(tmp_path)
    demo = tmp_path / "build" / "demo.parquet"

    code = smoketest_inference.run(
        notes=trained.notes,
        config=trained.config,
        lexicon=REPOSITORY_SECTION_HEADERS,
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
        config=trained.config,
        lexicon=REPOSITORY_SECTION_HEADERS,
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
        config=trained.config,
        lexicon=REPOSITORY_SECTION_HEADERS,
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
    """Predikce, které uložený model nereprodukuje, krok odmítne jako nesoulad."""
    trained = _train(tmp_path)
    table = ParquetPredictionTable(trained.predictions)
    first, *rest = table.read()
    table.write([replace(first, predicted=not first.predicted), *rest])
    demo = tmp_path / "demo.parquet"

    code = smoketest_inference.run(
        notes=trained.notes,
        config=trained.config,
        lexicon=REPOSITORY_SECTION_HEADERS,
        model=trained.model,
        predictions=trained.predictions,
        demo=demo,
    )

    assert code is ExitCode.REFUSED
    assert not demo.exists()
    assert first.note_id in caplog.text
    assert "Nesoulad" in caplog.text


def test_stripped_training_and_inference_agree(tmp_path: Path) -> None:
    """Trénink se stripem a inference se stejným configem projdou a zapíšou demo."""
    trained = _train(tmp_path, strip=True, sections=True)
    demo = tmp_path / "demo.parquet"

    code = smoketest_inference.run(
        notes=trained.notes,
        config=trained.config,
        lexicon=REPOSITORY_SECTION_HEADERS,
        model=trained.model,
        predictions=trained.predictions,
        demo=demo,
    )

    assert code is ExitCode.OK
    assert demo.exists()


def test_model_trained_without_strip_mismatches_stripped_inference(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Inference čte přepínač: naučené slovo leží jen v odstraněné sekci, skóre se změní."""
    trained = _train(tmp_path, strip=False, sections=True)
    stripping = tmp_path / "strip.json"
    stripping.write_text(
        json.dumps(training_config(row_id="row", strip_diagnostic_sections=True)),
        encoding="utf-8",
    )
    demo = tmp_path / "demo.parquet"

    code = smoketest_inference.run(
        notes=trained.notes,
        config=stripping,
        lexicon=REPOSITORY_SECTION_HEADERS,
        model=trained.model,
        predictions=trained.predictions,
        demo=demo,
    )

    assert code is ExitCode.REFUSED
    assert not demo.exists()
    assert "Nesoulad" in caplog.text


def test_normalized_model_infers_from_raw_notes(tmp_path: Path) -> None:
    """Model s normalizací se načte a klasifikuje syrové zprávy; config inference ji nečte."""
    trained = _train(tmp_path, normalize=True)
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps(training_config(row_id="row")), encoding="utf-8")
    demo = tmp_path / "demo.parquet"

    code = smoketest_inference.run(
        notes=trained.notes,
        config=plain,
        lexicon=REPOSITORY_SECTION_HEADERS,
        model=trained.model,
        predictions=trained.predictions,
        demo=demo,
    )

    assert code is ExitCode.OK
    assert demo.exists()
