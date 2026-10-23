"""Vstupní bod kroku train_smoketest_model: model a predikce, odmítnutí a determinismus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final

import pytest

from mediparse.domain.run_manifest import RunManifest
from mediparse.entrypoints import smoketest_training
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.prediction_table import ParquetPredictionTable
from mediparse.infrastructure.sklearn_classifier import ESTIMATOR, SkopsModelFile
from mediparse.infrastructure.split_table import ParquetSplitTable
from tests.model_support import (
    OVERLAPPING,
    sectioned_notes,
    separable_notes,
    training_config,
    write_raw_split,
    write_split,
)
from tests.support import (
    REPOSITORY_SECTION_HEADERS,
    REVISION,
    fixed_revision,
    missing_revision,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from mediparse.domain.environment import SourceRevision
    from mediparse.domain.input_note import InputNote

PATIENTS: Final = 12


def _inputs(
    root: Path, corpus: list[InputNote] | None = None, **config_fields: object
) -> tuple[Path, Path, Path]:
    corpus = corpus or separable_notes(PATIENTS, OVERLAPPING)
    notes = root / "notes.parquet"
    ParquetNoteTable(notes).write(corpus)
    config_body = training_config(**config_fields)
    if config_body["row_id"] is None:
        del config_body["row_id"]
    name = "-".join(f"{key}{value}" for key, value in sorted(config_fields.items()))
    config = root / f"config-{name}.json"
    config.write_text(json.dumps(config_body), encoding="utf-8")
    split = root / "splits.parquet"
    write_split(split, corpus)
    return notes, config, split


@dataclass(frozen=True)
class _Outputs:
    """Výstupy jednoho běhu kroku v jednom adresáři."""

    model: Path
    predictions: Path
    manifest: Path

    @classmethod
    def under(cls, directory: Path) -> _Outputs:
        return cls(
            directory / "model.skops",
            directory / "predictions.parquet",
            directory / "run_manifest.json",
        )

    def existing(self) -> list[Path]:
        return [
            path
            for path in (self.model, self.predictions, self.manifest)
            if path.exists()
        ]


def _call(
    inputs: tuple[Path, Path, Path],
    outputs: _Outputs,
    lexicon: Path,
    revision: Callable[[], SourceRevision],
) -> ExitCode:
    notes, config, split = inputs
    return smoketest_training.run(
        notes=notes,
        config=config,
        lexicon=lexicon,
        split=split,
        model=outputs.model,
        predictions=outputs.predictions,
        manifest=outputs.manifest,
        revision=revision,
    )


def _run(
    notes: Path,
    config: Path,
    split: Path,
    outputs: _Outputs,
    revision: Callable[[], SourceRevision] = fixed_revision,
) -> ExitCode:
    return _call((notes, config, split), outputs, REPOSITORY_SECTION_HEADERS, revision)


def test_writes_model_and_predictions(tmp_path: Path) -> None:
    """Běh uloží načitatelný model a predikce testovacích zpráv s row_id a foldem 0."""
    notes, config, split = _inputs(tmp_path)
    outputs = _Outputs.under(tmp_path / "build")

    code = _run(notes, config, split, outputs)

    rows = ParquetPredictionTable(outputs.predictions).read()
    assert code is ExitCode.OK
    assert rows
    assert {(row.row_id, row.fold) for row in rows} == {("row", 0)}
    assert (
        SkopsModelFile(outputs.model).load().classify(["diabetes insulin"])[0].predicted
    )
    written = RunManifest.model_validate_json(
        outputs.manifest.read_text(encoding="utf-8")
    )
    assert written.dataset_sha256 == file_sha256(notes)
    assert written.config_sha256 == file_sha256(config)
    assert written.seed == 0
    assert written.split_seed == 0
    assert written.split_folds == 2  # ruff: ignore[magic-value-comparison]
    assert written.split_sha256 == ParquetSplitTable(split).read().split.sha256()
    assert written.row_id == "row"
    assert written.environment.source == REVISION
    assert written.model_revision is None
    assert written.tokenizer_revision is None
    assert written.model == ESTIMATOR
    assert tuple(written.environment.packages) == smoketest_training.PACKAGES


@pytest.mark.parametrize(
    ("index", "missing"),
    [(0, "none.parquet"), (1, "none.json"), (2, "none-splits.parquet")],
    ids=["table", "config", "split"],
)
def test_missing_input_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, index: int, missing: str
) -> None:
    """Chybějící tabulka, config tréninku nebo soubor splitu krok odmítne, nic nevznikne a log jmenuje soubor."""
    paths = list(_inputs(tmp_path))
    paths[index] = tmp_path / missing
    outputs = _Outputs.under(tmp_path / "out")

    code = _run(paths[0], paths[1], paths[2], outputs)

    assert code is ExitCode.REFUSED
    assert outputs.existing() == []
    assert missing in caplog.text


def test_diagnosis_without_positives_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Diagnóza bez pozitivních pacientů krok odmítne a nic nevznikne."""
    notes, config, split = _inputs(tmp_path, diagnosis="pneumonia")
    outputs = _Outputs.under(tmp_path / "out")

    code = _run(notes, config, split, outputs)

    assert code is ExitCode.REFUSED
    assert outputs.existing() == []
    assert "pneumonia" in caplog.text


def test_invalid_config_is_refused(tmp_path: Path) -> None:
    """Config bez row_id krok odmítne a nic nevznikne."""
    notes, config, split = _inputs(tmp_path, row_id=None)
    outputs = _Outputs.under(tmp_path / "out")

    code = _run(notes, config, split, outputs)

    assert code is ExitCode.REFUSED
    assert outputs.existing() == []


def test_split_with_subject_in_two_folds_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Pacient ve dvou foldech krok odmítne hláškou o úniku a bez jeho id v logu.

    Otisk sedí podle kontraktu ``Split.sha256``. Bez invariantu by adaptér soubor
    přijal a krok by skončil REFUSED jinou hláškou; rozhoduje aserce na hlášku úniku.
    """
    notes, config, split = _inputs(tmp_path)
    rows = [(90000001, 0), (90000001, 1), (90000002, 1)]
    lines = "".join(f"{subject}\t{fold}\n" for subject, fold in rows)
    write_raw_split(
        split,
        rows,
        {"folds": 2, "seed": 0, "sha256": hashlib.sha256(lines.encode()).hexdigest()},
    )
    outputs = _Outputs.under(tmp_path / "out")

    code = _run(notes, config, split, outputs)

    assert code is ExitCode.REFUSED
    assert outputs.existing() == []
    assert "ve více foldech: 1" in caplog.text
    assert "90000001" not in caplog.text
    assert "90000002" not in caplog.text


def test_manifest_split_comes_from_split_file(tmp_path: Path) -> None:
    """Seed, počet foldů a otisk manifestu jsou ze souboru splitu, ne z configu."""
    corpus = separable_notes(PATIENTS, OVERLAPPING)
    notes, config, split = _inputs(tmp_path)
    shared = write_split(split, corpus, seed=5, folds=3)
    outputs = _Outputs.under(tmp_path / "out")

    code = _run(notes, config, split, outputs)

    written = RunManifest.model_validate_json(
        outputs.manifest.read_text(encoding="utf-8")
    )
    assert code is ExitCode.OK
    assert (written.split_seed, written.split_folds) == (5, 3)
    assert written.split_sha256 == shared.split.sha256()


def _successful_run(root: Path, seed: int, name: str) -> _Outputs:
    notes, config, split = _inputs(root, seed=seed)
    outputs = _Outputs.under(root / name)
    assert _run(notes, config, split, outputs) is ExitCode.OK
    return outputs


def _manifest_bytes(root: Path, seed: int, name: str) -> bytes:
    return _successful_run(root, seed, name).manifest.read_bytes()


def test_manifest_is_byte_identical_for_same_inputs(tmp_path: Path) -> None:
    """Dva běhy se stejnými vstupy dají bajtově totožný manifest."""
    assert _manifest_bytes(tmp_path, 0, "first") == _manifest_bytes(
        tmp_path, 0, "second"
    )


def test_seed_changes_manifest(tmp_path: Path) -> None:
    """Jiný seed modelu dá jiný seed i hash configu, ale týž split."""
    first = RunManifest.model_validate_json(_manifest_bytes(tmp_path, 0, "a"))
    other = RunManifest.model_validate_json(_manifest_bytes(tmp_path, 1, "b"))

    assert (first.seed, other.seed) == (0, 1)
    assert first.config_sha256 != other.config_sha256
    assert first.split_sha256 == other.split_sha256


def test_unreadable_revision_is_refused_before_training(tmp_path: Path) -> None:
    """Revize, kterou nelze přečíst, krok odmítne a nevznikne nic."""
    notes, config, split = _inputs(tmp_path)
    outputs = _Outputs.under(tmp_path / "out")

    code = _run(notes, config, split, outputs, revision=missing_revision)

    assert code is ExitCode.REFUSED
    assert outputs.existing() == []


def _manifest(outputs: _Outputs) -> RunManifest:
    return RunManifest.model_validate_json(outputs.manifest.read_text(encoding="utf-8"))


def test_manifest_records_strip_sections_and_lexicon_digest(tmp_path: Path) -> None:
    """Se stripem nese manifest klíče diagnostických sekcí a otisk lexikonu."""
    notes, config, split = _inputs(tmp_path, strip_diagnostic_sections=True)
    outputs = _Outputs.under(tmp_path / "out")

    code = _run(notes, config, split, outputs)

    strip = _manifest(outputs).diagnostic_strip
    assert code is ExitCode.OK
    assert strip is not None
    assert strip.sections == ("discharge_diagnosis",)
    assert strip.lexicon_sha256 == file_sha256(REPOSITORY_SECTION_HEADERS)


def test_manifest_without_strip_has_null_record(tmp_path: Path) -> None:
    """Bez stripu je ``diagnostic_strip`` v manifestu ``null``."""
    notes, config, split = _inputs(tmp_path)
    outputs = _Outputs.under(tmp_path / "out")

    code = _run(notes, config, split, outputs)

    assert code is ExitCode.OK
    assert (
        json.loads(outputs.manifest.read_text(encoding="utf-8"))["diagnostic_strip"]
        is None
    )


def _lexicon(path: Path, *, extra_diagnostic: bool, synonym: str | None = None) -> Path:
    body = json.loads(REPOSITORY_SECTION_HEADERS.read_text(encoding="utf-8"))
    if extra_diagnostic:
        body["sections"]["past_medical_history"]["diagnostic"] = True
    if synonym:
        body["sections"]["allergies"]["synonyms"].append(synonym)
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_manifest_lists_diagnostic_sections_sorted(tmp_path: Path) -> None:
    """Sekce jsou seřazené, i když je lexikon deklaruje v opačném pořadí."""
    lexicon = _lexicon(tmp_path / "lexicon.json", extra_diagnostic=True)
    notes, config, split = _inputs(tmp_path, strip_diagnostic_sections=True)
    outputs = _Outputs.under(tmp_path / "out")

    code = _call((notes, config, split), outputs, lexicon, fixed_revision)

    strip = _manifest(outputs).diagnostic_strip
    assert code is ExitCode.OK
    assert strip is not None
    assert strip.sections == ("discharge_diagnosis", "past_medical_history")


def test_synonym_change_changes_lexicon_digest(tmp_path: Path) -> None:
    """Změna synonyma při stejném datasetu a configu změní otisk lexikonu."""
    notes, config, split = _inputs(tmp_path, strip_diagnostic_sections=True)
    digests = []
    for name, synonym in (("a", None), ("b", "Allergy list")):
        lexicon = _lexicon(
            tmp_path / f"{name}.json", extra_diagnostic=False, synonym=synonym
        )
        outputs = _Outputs.under(tmp_path / name)
        assert (
            _call((notes, config, split), outputs, lexicon, fixed_revision)
            is ExitCode.OK
        )
        strip = _manifest(outputs).diagnostic_strip
        assert strip is not None
        digests.append(strip.lexicon_sha256)

    assert digests[0] != digests[1]


def test_missing_lexicon_is_refused(tmp_path: Path) -> None:
    """Chybějící lexikon při zapnutém stripu krok odmítne a nic nevznikne."""
    notes, config, split = _inputs(tmp_path, strip_diagnostic_sections=True)
    outputs = _Outputs.under(tmp_path / "out")

    code = _call(
        (notes, config, split), outputs, tmp_path / "none.json", fixed_revision
    )

    assert code is ExitCode.REFUSED
    assert outputs.existing() == []


def test_strip_changes_scores_on_sectioned_notes(tmp_path: Path) -> None:
    """Trénink se stripem čte stripnutý text, takže skóre predikcí se liší od běhu bez stripu."""
    corpus = sectioned_notes(PATIENTS)
    scores = []
    for strip in (False, True):
        root = tmp_path / str(strip)
        root.mkdir()
        notes, config, split = _inputs(root, corpus, strip_diagnostic_sections=strip)
        outputs = _Outputs.under(root / "out")
        assert _run(notes, config, split, outputs) is ExitCode.OK
        scores.append([
            row.score for row in ParquetPredictionTable(outputs.predictions).read()
        ])

    assert scores[0] != scores[1]


def test_strip_removing_all_text_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Strip, po kterém nezbyde žádný token, krok odmítne místo pádu vectorizeru."""
    corpus = [
        replace(note, text=f"Discharge Diagnosis:\n{note.text}\n")
        for note in separable_notes(PATIENTS, OVERLAPPING)
    ]
    notes, config, split = _inputs(tmp_path, corpus, strip_diagnostic_sections=True)
    outputs = _Outputs.under(tmp_path / "out")

    code = _run(notes, config, split, outputs)

    assert code is ExitCode.REFUSED
    assert outputs.existing() == []
    assert "Traceback" not in caplog.text
    assert "žádný token" in caplog.text
    assert "diabetes insulin" not in caplog.text
