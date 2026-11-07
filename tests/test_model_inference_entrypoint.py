"""Vstupní bod kroku infer_model: uložený model nad odloženými zprávami."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from typing import TYPE_CHECKING, Final, NamedTuple

import pytest

from mediparse.domain.costs import InferenceCosts, LatencyPlan
from mediparse.entrypoints import model_inference, model_training
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.machine_hardware import machine_hardware
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.prediction_table import ParquetPredictionTable
from tests.model_support import (
    ACRONYM_VOCABULARY,
    SEPARABLE_VOCABULARY,
    SPELLING_VOCABULARY,
    sectioned_notes,
    separable_notes,
    training_config,
    write_split,
)
from tests.support import (
    REPOSITORY_DIAGNOSIS_CANDIDATES,
    REPOSITORY_SECTION_HEADERS,
    fixed_revision,
)

if TYPE_CHECKING:
    from pathlib import Path


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
    vocabulary: tuple[str, str] | None = None,
) -> Trained:
    notes = root / "notes.parquet"
    corpus = (
        sectioned_notes(PATIENTS)
        if sections
        else separable_notes(
            PATIENTS,
            vocabulary or (SPELLING_VOCABULARY if normalize else SEPARABLE_VOCABULARY),
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
    candidates = root / "candidates.json"
    shutil.copy(REPOSITORY_DIAGNOSIS_CANDIDATES, candidates)
    trained = Trained(notes, config, root / "model.skops", root / "predictions.parquet")
    code = model_training.run(
        cohort=notes,
        config=config,
        candidates=candidates,
        lexicon=REPOSITORY_SECTION_HEADERS,
        split=split,
        model=trained.model,
        predictions=trained.predictions,
        manifest=root / "run_manifest.json",
        revision=fixed_revision,
    )
    assert code is ExitCode.OK
    return trained


def _infer(trained: Trained, tmp_path: Path, **changes: Path) -> ExitCode:
    inference = changes.get("inference", tmp_path / "inference.parquet")
    paths = {
        "cohort": trained.notes,
        "config": trained.config,
        "lexicon": REPOSITORY_SECTION_HEADERS,
        "model": trained.model,
        "predictions": trained.predictions,
        "inference": inference,
        "costs": inference.with_suffix(".costs.json"),
    }
    return model_inference.run(**(paths | changes))


def test_writes_all_held_out_predictions(tmp_path: Path) -> None:
    """Uložený model reprodukuje predikce všech odložených zpráv."""
    trained = _train(tmp_path)
    inference = tmp_path / "build" / "inference.parquet"

    code = _infer(trained, tmp_path, inference=inference)

    rows = ParquetPredictionTable(inference).read()
    expected = ParquetPredictionTable(trained.predictions).read()
    assert code is ExitCode.OK
    assert [
        (row.row_id, row.fold, row.note_id, row.subject_id, row.diagnosis)
        for row in rows
    ] == [
        (row.row_id, row.fold, row.note_id, row.subject_id, row.diagnosis)
        for row in expected
    ]
    assert [row.predicted for row in rows] == [row.predicted for row in expected]
    assert [row.score for row in rows] == pytest.approx(
        [row.score for row in expected], rel=1e-9, abs=1e-9
    )


def test_missing_model_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Chybějící model krok odmítne s hláškou a bez tracebacku."""
    trained = _train(tmp_path)
    inference = tmp_path / "inference.parquet"

    code = _infer(trained, tmp_path, model=tmp_path / "none.skops")

    assert code is ExitCode.REFUSED
    assert not inference.exists()
    assert not inference.with_suffix(".costs.json").exists()
    assert "none.skops" in caplog.text
    assert "Traceback" not in caplog.text


def test_missing_predictions_are_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Chybějící predikce krok odmítnou s hláškou a bez tracebacku."""
    trained = _train(tmp_path)
    inference = tmp_path / "inference.parquet"

    code = _infer(trained, tmp_path, predictions=tmp_path / "none.parquet")

    assert code is ExitCode.REFUSED
    assert not inference.exists()
    assert not inference.with_suffix(".costs.json").exists()
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
    inference = tmp_path / "inference.parquet"

    code = _infer(trained, tmp_path)

    assert code is ExitCode.REFUSED
    assert not inference.exists()
    assert not inference.with_suffix(".costs.json").exists()
    assert all(item.note_id not in caplog.text for item in (first, *rest))
    assert "Nesoulad" in caplog.text


def test_stripped_training_and_inference_agree(tmp_path: Path) -> None:
    """Trénink se stripem a inference se stejným configem projdou a zapíšou predikce."""
    trained = _train(tmp_path, strip=True, sections=True)
    inference = tmp_path / "inference.parquet"

    code = _infer(trained, tmp_path)

    assert code is ExitCode.OK
    assert inference.exists()


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
    inference = tmp_path / "inference.parquet"

    code = _infer(trained, tmp_path, config=stripping)

    assert code is ExitCode.REFUSED
    assert not inference.exists()
    assert not inference.with_suffix(".costs.json").exists()
    assert "Nesoulad" in caplog.text


def test_normalized_model_infers_from_raw_notes(tmp_path: Path) -> None:
    """Model s normalizací se načte a klasifikuje syrové zprávy; config inference ji nečte."""
    trained = _train(tmp_path, normalize=True)
    plain = tmp_path / "plain.json"
    plain.write_text(json.dumps(training_config(row_id="row")), encoding="utf-8")
    inference = tmp_path / "inference.parquet"

    code = _infer(trained, tmp_path, config=plain)

    assert code is ExitCode.OK
    assert inference.exists()


def test_model_with_acronyms_infers_without_candidates_file(tmp_path: Path) -> None:
    """Slovník zkratek nese model: po smazání souboru kandidátů inference predikce reprodukuje."""
    trained = _train(tmp_path, normalize=True, vocabulary=ACRONYM_VOCABULARY)
    (tmp_path / "candidates.json").unlink()
    inference = tmp_path / "inference.parquet"

    code = _infer(trained, tmp_path)

    assert code is ExitCode.OK
    assert inference.exists()


def test_writes_aggregate_costs_with_measurement_plan(tmp_path: Path) -> None:
    """Soubor nákladů se načte zpět s plánem 10/100 a skutečným počtem vzorků."""
    trained = _train(tmp_path)
    inference = tmp_path / "inference.parquet"
    costs = tmp_path / "costs" / "inference.json"

    code = _infer(trained, tmp_path, costs=costs)

    written = InferenceCosts.model_validate_json(costs.read_text(encoding="utf-8"))
    count = len(ParquetPredictionTable(inference).read())
    assert code is ExitCode.OK
    assert (
        written.plan
        == model_inference.LATENCY_PLAN
        == LatencyPlan(warmup=10, samples=100)
    )
    assert written.latency.samples == min(100, count)
    assert written.batch.notes == count
    assert written.hardware == machine_hardware()
