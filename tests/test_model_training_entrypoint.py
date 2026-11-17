"""Vstupní bod kroku train_model: model a predikce, odmítnutí a determinismus."""

from __future__ import annotations

import hashlib
import json
import shutil
from contextlib import contextmanager
from dataclasses import replace
from threading import Event
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final
from unittest.mock import Mock

import pynvml
import pytest

from mediparse.domain.costs import GpuMeasurement, GpuUsage, TrainingCosts
from mediparse.domain.environment import GpuDevice
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.model_input import TextPreprocessing
from mediparse.domain.run_manifest import RunManifest
from mediparse.entrypoints import model_training
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.entrypoints.model_training import TrainingPaths
from mediparse.infrastructure import sklearn_classifier
from mediparse.infrastructure.diagnosis_candidates_file import load_text_normalization
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.json_file import write_model
from mediparse.infrastructure.machine_hardware import machine_hardware
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.prediction_table import ParquetPredictionTable
from mediparse.infrastructure.sklearn_classifier import ESTIMATOR, SkopsModelFile
from mediparse.infrastructure.split_table import ParquetSplitTable
from tests.model_support import (
    ACRONYM_VOCABULARY,
    OVERLAPPING,
    SPELLING_VOCABULARY,
    VARIANT_VOCABULARY,
    model_input,
    sectioned_notes,
    separable_notes,
    training_config,
    with_negative_controls,
    write_raw_split,
    write_split,
)
from tests.support import (
    REPOSITORY_DIAGNOSIS_CANDIDATES,
    REPOSITORY_SECTION_HEADERS,
    REVISION,
    fixed_revision,
    missing_revision,
)
from tests.tuning_support import tuning_manifest

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path

    from mediparse.domain.environment import Hardware, SourceRevision
    from mediparse.domain.input_note import InputNote
    from mediparse.domain.tuning import ParamValue

PATIENTS: Final = 48


def _inputs(
    root: Path, corpus: list[InputNote] | None = None, **config_fields: object
) -> TrainingPaths:
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
    return TrainingPaths(
        cohort=notes,
        config=config,
        candidates=REPOSITORY_DIAGNOSIS_CANDIDATES,
        lexicon=REPOSITORY_SECTION_HEADERS,
        split=split,
        tuning=None,
        model=root / "out/model.skops",
        predictions=root / "out/predictions.parquet",
        manifest=root / "out/run_manifest.json",
        costs=root / "out/training_costs.json",
    )


def _under(paths: TrainingPaths, directory: Path) -> TrainingPaths:
    return replace(
        paths,
        model=directory / "model.skops",
        predictions=directory / "predictions.parquet",
        manifest=directory / "run_manifest.json",
        costs=directory / "training_costs.json",
    )


def _existing(paths: TrainingPaths) -> list[Path]:
    return [
        path
        for path in (paths.model, paths.predictions, paths.manifest, paths.costs)
        if path.exists()
    ]


def _run(
    paths: TrainingPaths, revision: Callable[[], SourceRevision] = fixed_revision
) -> ExitCode:
    return model_training.run(
        paths=paths,
        environ={},
        revision=revision,
        cuda_visible_devices="",
        cuda_device_order=None,
    )


def test_writes_model_and_predictions(tmp_path: Path) -> None:
    """Běh uloží načitatelný model a predikce testovacích zpráv s row_id a foldem 0."""
    paths = _inputs(tmp_path)
    outputs = _under(paths, tmp_path / "build")

    code = _run(outputs)

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
    assert written.dataset_sha256 == file_sha256(paths.cohort)
    assert (
        written.predictions_sha256
        == hashlib.sha256(outputs.predictions.read_bytes()).hexdigest()
    )
    assert written.config_sha256 == file_sha256(paths.config)
    assert written.tuning_manifest_sha256 is None
    assert (
        written.parameters
        == json.loads(paths.config.read_text(encoding="utf-8"))["parameters"]
    )
    assert written.threshold == SkopsModelFile(outputs.model).load().threshold
    assert written.threshold_folds == 3
    assert written.converged is True
    assert written.seed == 0
    assert written.split_seed == 0
    assert written.split_folds == 2
    assert written.split_sha256 == ParquetSplitTable(paths.split).read().split.sha256()
    assert written.row_id == "row"
    assert written.environment.source == REVISION
    assert written.hardware == machine_hardware(gpus=())
    costs = TrainingCosts.model_validate_json(outputs.costs.read_text(encoding="utf-8"))
    assert costs.gpu.devices == ()
    assert written.hardware.gpus == ()
    assert written.model_revision is None
    assert written.tokenizer_revision is None
    assert written.model == ESTIMATOR
    assert tuple(written.environment.packages) == model_training.PACKAGES


@pytest.mark.parametrize(
    ("field", "missing"),
    [
        ("cohort", "none.parquet"),
        ("config", "none.json"),
        ("split", "none-splits.parquet"),
    ],
    ids=["table", "config", "split"],
)
def test_missing_input_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, field: str, missing: str
) -> None:
    """Chybějící tabulka, config tréninku nebo soubor splitu krok odmítne, nic nevznikne a log jmenuje soubor."""
    paths = replace(_inputs(tmp_path), **{field: tmp_path / missing})

    code = _run(paths)

    assert code is ExitCode.REFUSED
    assert _existing(paths) == []
    assert missing in caplog.text


def test_diagnosis_without_positives_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Diagnóza bez pozitivních pacientů krok odmítne a nic nevznikne."""
    paths = _inputs(tmp_path, diagnosis="pneumonia")

    code = _run(paths)

    assert code is ExitCode.REFUSED
    assert _existing(paths) == []
    assert "pneumonia" in caplog.text


def test_invalid_config_is_refused(tmp_path: Path) -> None:
    """Config bez row_id krok odmítne a nic nevznikne."""
    paths = _inputs(tmp_path, row_id=None)

    code = _run(paths)

    assert code is ExitCode.REFUSED
    assert _existing(paths) == []


def test_split_with_subject_in_two_folds_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Pacient ve dvou foldech krok odmítne hláškou o úniku a bez jeho id v logu.

    Otisk sedí podle kontraktu ``Split.sha256``. Bez invariantu by adaptér soubor
    přijal a krok by skončil REFUSED jinou hláškou; rozhoduje aserce na hlášku úniku.
    """
    paths = _inputs(tmp_path)
    rows = [(90000001, 0), (90000001, 1), (90000002, 1)]
    lines = "".join(f"{subject}\t{fold}\n" for subject, fold in rows)
    write_raw_split(
        paths.split,
        rows,
        {"folds": 2, "seed": 0, "sha256": hashlib.sha256(lines.encode()).hexdigest()},
    )

    code = _run(paths)

    assert code is ExitCode.REFUSED
    assert _existing(paths) == []
    assert "ve více částech splitu: 1" in caplog.text
    assert "90000001" not in caplog.text
    assert "90000002" not in caplog.text


def test_manifest_split_comes_from_split_file(tmp_path: Path) -> None:
    """Seed, počet foldů a otisk manifestu jsou ze souboru splitu, ne z configu."""
    corpus = separable_notes(PATIENTS, OVERLAPPING)
    paths = _inputs(tmp_path)
    shared = write_split(paths.split, corpus, seed=5, folds=3)

    code = _run(paths)

    written = RunManifest.model_validate_json(
        paths.manifest.read_text(encoding="utf-8")
    )
    assert code is ExitCode.OK
    assert (written.split_seed, written.split_folds) == (5, 3)
    assert written.split_sha256 == shared.split.sha256()


def _successful_run(root: Path, seed: int, name: str) -> TrainingPaths:
    paths = _inputs(root, seed=seed)
    outputs = _under(paths, root / name)
    assert _run(outputs) is ExitCode.OK
    return outputs


def _manifest_bytes(root: Path, seed: int, name: str) -> bytes:
    return _successful_run(root, seed, name).manifest.read_bytes()


def test_repeated_runs_preserve_manifest_and_prediction_scores(tmp_path: Path) -> None:
    """Dva běhy se stejnými vstupy zachovají manifest i všechny labely a skóre."""
    first = _successful_run(tmp_path, 0, "first")
    second = _successful_run(tmp_path, 0, "second")
    assert first.manifest.read_bytes() == second.manifest.read_bytes()
    assert (
        ParquetPredictionTable(first.predictions).read()
        == ParquetPredictionTable(second.predictions).read()
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
    paths = _inputs(tmp_path)

    code = _run(paths, revision=missing_revision)

    assert code is ExitCode.REFUSED
    assert _existing(paths) == []


def test_unknown_hardware_is_refused_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Nezjistitelný HW odmítne běh před nákladným fit i před zápisem modelu."""
    paths = _inputs(tmp_path)
    message = "Hardware stroje nelze zjistit ze systémových údajů."

    def unavailable_hardware(gpus: tuple[GpuDevice, ...]) -> Hardware:
        assert gpus == ()
        raise InvalidInputError(message)

    def unexpected_fit(*_args: object) -> None:
        pytest.fail("Trénink nesmí začít před úspěšným zjištěním HW.")

    monkeypatch.setattr(model_training, "machine_hardware", unavailable_hardware)
    monkeypatch.setattr(model_training.LinearSvmTrainer, "fit", unexpected_fit)

    code = _run(paths)

    assert code is ExitCode.REFUSED
    assert _existing(paths) == []
    assert message in caplog.text


def _manifest(outputs: TrainingPaths) -> RunManifest:
    return RunManifest.model_validate_json(outputs.manifest.read_text(encoding="utf-8"))


def test_manifest_records_strip_sections_and_lexicon_digest(tmp_path: Path) -> None:
    """Se stripem nese manifest klíče diagnostických sekcí a otisk lexikonu."""
    paths = _inputs(
        tmp_path,
        input=model_input(strip=True),
    )

    code = _run(paths)

    strip = _manifest(paths).diagnostic_strip
    assert code is ExitCode.OK
    assert strip is not None
    assert strip.sections == ("discharge_diagnosis",)
    assert strip.lexicon_sha256 == file_sha256(REPOSITORY_SECTION_HEADERS)


def test_manifest_without_strip_has_null_record(tmp_path: Path) -> None:
    """Bez stripu je ``diagnostic_strip`` v manifestu ``null``."""
    paths = _inputs(tmp_path)

    code = _run(paths)

    assert code is ExitCode.OK
    assert (
        json.loads(paths.manifest.read_text(encoding="utf-8"))["diagnostic_strip"]
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
    paths = _inputs(
        tmp_path,
        input=model_input(strip=True),
    )

    code = _run(replace(paths, lexicon=lexicon), revision=fixed_revision)

    strip = _manifest(paths).diagnostic_strip
    assert code is ExitCode.OK
    assert strip is not None
    assert strip.sections == ("discharge_diagnosis", "past_medical_history")


def test_synonym_change_changes_lexicon_digest(tmp_path: Path) -> None:
    """Změna synonyma při stejném datasetu a configu změní otisk lexikonu."""
    paths = _inputs(
        tmp_path,
        input=model_input(strip=True),
    )
    digests = []
    for name, synonym in (("a", None), ("b", "Allergy list")):
        lexicon = _lexicon(
            tmp_path / f"{name}.json", extra_diagnostic=False, synonym=synonym
        )
        outputs = _under(paths, tmp_path / name)
        assert (
            _run(replace(outputs, lexicon=lexicon), revision=fixed_revision)
            is ExitCode.OK
        )
        strip = _manifest(outputs).diagnostic_strip
        assert strip is not None
        digests.append(strip.lexicon_sha256)

    assert digests[0] != digests[1]


def test_missing_lexicon_is_refused(tmp_path: Path) -> None:
    """Chybějící lexikon při zapnutém stripu krok odmítne a nic nevznikne."""
    paths = _inputs(
        tmp_path,
        input=model_input(strip=True),
    )

    code = _run(replace(paths, lexicon=tmp_path / "none.json"), revision=fixed_revision)

    assert code is ExitCode.REFUSED
    assert _existing(paths) == []


def test_strip_changes_scores_on_sectioned_notes(tmp_path: Path) -> None:
    """Trénink se stripem čte stripnutý text, takže skóre predikcí se liší od běhu bez stripu."""
    corpus = with_negative_controls(sectioned_notes(PATIENTS))
    scores = []
    for strip in (False, True):
        root = tmp_path / str(strip)
        root.mkdir()
        paths = _inputs(
            root,
            corpus,
            input=model_input(strip=strip),
        )
        assert _run(paths) is ExitCode.OK
        scores.append([
            row.score for row in ParquetPredictionTable(paths.predictions).read()
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
    paths = _inputs(
        tmp_path,
        corpus,
        input=model_input(strip=True),
    )

    code = _run(paths)

    assert code is ExitCode.REFUSED
    assert _existing(paths) == []
    assert "Traceback" not in caplog.text
    assert "žádný token" in caplog.text
    assert "diabetes insulin" not in caplog.text


def test_manifest_records_normalization_with_acronyms(tmp_path: Path) -> None:
    """Manifest zapíše ``None`` bez normalizace a slovník zkratek z configu s ní."""
    recorded = []
    for normalize in (False, True):
        root = tmp_path / str(normalize)
        root.mkdir()
        paths = _inputs(
            root,
            input=model_input(preprocessing="normalized" if normalize else "none"),
        )
        assert _run(paths) is ExitCode.OK
        manifest = _manifest(paths)
        assert manifest.text_preprocessing is (
            TextPreprocessing.NORMALIZED if normalize else TextPreprocessing.NONE
        )
        recorded.append(manifest.text_normalization)

    assert recorded[0] is None
    assert recorded[1] == load_text_normalization(
        REPOSITORY_DIAGNOSIS_CANDIDATES, variants=False
    )


def test_normalization_without_candidates_file_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Zapnutá normalizace bez souboru kandidátů krok odmítne a nic nezapíše."""
    paths = _inputs(
        tmp_path,
        input=model_input(preprocessing="normalized"),
    )

    code = _run(replace(paths, candidates=tmp_path / "none.json"))

    assert code is ExitCode.REFUSED
    assert _existing(paths) == []
    assert "none.json" in caplog.text


def test_candidates_file_is_not_read_without_normalization(tmp_path: Path) -> None:
    """Při vypnuté normalizaci se soubor kandidátů nečte, neexistující cesta projde."""
    paths = _inputs(
        tmp_path,
        input=model_input(),
    )

    code = _run(replace(paths, candidates=tmp_path / "none.json"))

    assert code is ExitCode.OK


def test_normalization_merges_abbreviation_and_full_name(tmp_path: Path) -> None:
    """Zapojení v composition rootu: zkratka a vypsaný název dají se slovníkem stejné skóre.

    Bez něj by test prošel i tehdy, kdyby entrypoint slovník trenérovi nepředal.
    """
    corpus = with_negative_controls(separable_notes(PATIENTS, ACRONYM_VOCABULARY))
    scores = {}
    for normalize in (False, True):
        root = tmp_path / str(normalize)
        root.mkdir()
        paths = _inputs(
            root,
            corpus,
            input=model_input(preprocessing="normalized" if normalize else "none"),
        )
        candidates = root / "candidates.json"
        shutil.copy(REPOSITORY_DIAGNOSIS_CANDIDATES, candidates)
        assert _run(replace(paths, candidates=candidates)) is ExitCode.OK
        classifier = SkopsModelFile(paths.model).load()
        scores[normalize] = {
            item.score for item in classifier.classify(list(ACRONYM_VOCABULARY))
        }

    assert len(scores[False]) > 1
    assert len(scores[True]) == 1


def test_normalization_changes_scores_on_spelling_variants(tmp_path: Path) -> None:
    """Zprávy se liší jen zápisem dávky a věku: bez normalizace je model rozliší, s ní ne.

    Výchozí slovník testů žádné rozdíly zápisu nemá a test by prošel naprázdno.
    """
    corpus = with_negative_controls(separable_notes(PATIENTS, SPELLING_VOCABULARY))
    scores = {}
    for normalize in (False, True):
        root = tmp_path / str(normalize)
        root.mkdir()
        paths = _inputs(
            root,
            corpus,
            input=model_input(preprocessing="normalized" if normalize else "none"),
        )
        assert _run(paths) is ExitCode.OK
        scores[normalize] = {
            item.score
            for item in SkopsModelFile(paths.model).load().classify(SPELLING_VOCABULARY)
        }

    assert len(scores[False]) > 1
    assert len(scores[True]) == 1


def test_variant_switch_controls_model_and_manifest(tmp_path: Path) -> None:
    """Přepínač sjednotí skóre varianty a názvu a manifest nese právě použitý slovník."""
    corpus = with_negative_controls(separable_notes(PATIENTS, VARIANT_VOCABULARY))
    scores = {}
    for enabled in (False, True):
        root = tmp_path / str(enabled)
        root.mkdir()
        paths = _inputs(
            root,
            corpus,
            input=model_input(preprocessing="dictionary" if enabled else "normalized"),
        )
        assert _run(paths) is ExitCode.OK
        manifest = _manifest(paths)
        assert manifest.text_preprocessing is (
            TextPreprocessing.DICTIONARY if enabled else TextPreprocessing.NORMALIZED
        )
        normalization = manifest.text_normalization
        assert normalization is not None
        assert normalization.acronyms
        if enabled:
            assert normalization.variants["kindey"] == "kidney"
        else:
            assert normalization.variants == {}
        classifier = SkopsModelFile(paths.model).load()
        scores[enabled] = {
            item.score for item in classifier.classify(VARIANT_VOCABULARY)
        }
    assert len(scores[False]) > 1
    assert len(scores[True]) == 1


def test_training_records_visible_gpu_and_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manifest a samostatné náklady nesou stejnou podvrženou kartu i měření."""
    paths = _inputs(tmp_path)
    device = GpuDevice(name="Fake GPU", memory_total_bytes=1000)
    measurement = GpuMeasurement(
        seconds=2.0,
        devices=(
            GpuUsage(
                device=device,
                energy_joules=3.5,
                memory_used_start_bytes=100,
                memory_used_peak_bytes=300,
            ),
        ),
    )
    closed = Event()

    def result() -> GpuMeasurement:
        assert closed.is_set(), "Náklady se smí číst až po uzavření okna."
        return measurement

    @contextmanager
    def monitor(visible: str | None, order: str | None) -> Generator[SimpleNamespace]:
        assert visible is not None
        assert not visible
        assert order is None
        try:
            yield SimpleNamespace(devices=(device,), measurement=result)
        finally:
            closed.set()
        assert paths.model.exists(), "Trénink musí běžet uvnitř okna."

    monkeypatch.setattr(model_training, "gpu_monitor", monitor)
    code = _run(paths)

    assert code is ExitCode.OK
    assert _manifest(paths).hardware.gpus == (device,)
    written = TrainingCosts.model_validate_json(paths.costs.read_text(encoding="utf-8"))
    assert written.gpu == measurement


def test_nvml_error_refuses_training_before_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chyba NVML nesmí začít trénink ani zapsat model, manifest nebo náklady."""
    paths = _inputs(tmp_path)
    monkeypatch.setattr(
        pynvml,
        "nvmlInit",
        Mock(side_effect=pynvml.NVMLError(pynvml.NVML_ERROR_NO_PERMISSION)),
    )
    fit = Mock(side_effect=AssertionError("Trénink po chybě NVML nesmí začít."))
    monkeypatch.setattr(model_training.LinearSvmTrainer, "fit", fit)

    code = model_training.run(
        paths=model_training.TrainingPaths(
            tuning=None,
            cohort=paths.cohort,
            config=paths.config,
            split=paths.split,
            candidates=REPOSITORY_DIAGNOSIS_CANDIDATES,
            lexicon=REPOSITORY_SECTION_HEADERS,
            model=paths.model,
            predictions=paths.predictions,
            manifest=paths.manifest,
            costs=paths.costs,
        ),
        environ={},
        revision=fixed_revision,
        cuda_visible_devices=None,
        cuda_device_order=None,
    )

    assert code is ExitCode.REFUSED
    assert _existing(paths) == []
    fit.assert_not_called()


def test_training_main_forwards_cuda_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Composition root předá obě proměnné CUDA i cestu nákladů do run."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-synthetic")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    execute = Mock(return_value=ExitCode.OK)
    monkeypatch.setattr(model_training, "run", execute)
    monkeypatch.setattr(model_training, "configure_logging", Mock())
    code = model_training.main(
        paths=model_training.TrainingPaths(
            tuning=tmp_path / "tuning.json",
            cohort=tmp_path / "notes.parquet",
            config=tmp_path / "config.json",
            candidates=tmp_path / "candidates.json",
            lexicon=tmp_path / "lexicon.json",
            split=tmp_path / "splits.parquet",
            model=tmp_path / "model.skops",
            predictions=tmp_path / "predictions.parquet",
            manifest=tmp_path / "manifest.json",
            costs=tmp_path / "costs.json",
        ),
        log=tmp_path / "train.log",
    )
    assert code is ExitCode.OK
    assert execute.call_args.kwargs["cuda_visible_devices"] == "GPU-synthetic"
    assert execute.call_args.kwargs["cuda_device_order"] == "PCI_BUS_ID"
    assert execute.call_args.kwargs["paths"].costs == tmp_path / "costs.json"
    assert execute.call_args.kwargs["paths"].tuning == tmp_path / "tuning.json"


@pytest.mark.parametrize("phase", ["input", "normalization"])
def test_invalid_preparation_is_refused_before_gpu_monitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    """Chybný lexikon i slovník normalizace se odmítnou před otevřením GPU okna."""
    inputs = _inputs(
        tmp_path,
        input=model_input(
            strip=phase == "input",
            preprocessing="normalized" if phase == "normalization" else "none",
        ),
    )
    monitor = Mock(
        side_effect=AssertionError("Neplatná příprava nesmí otevřít GPU okno.")
    )
    monkeypatch.setattr(model_training, "gpu_monitor", monitor)

    code = (
        _run(
            replace(inputs, lexicon=tmp_path / "missing-lexicon.json"),
        )
        if phase == "input"
        else _run(replace(inputs, candidates=tmp_path / "missing-candidates.json"))
    )

    assert code is ExitCode.REFUSED
    assert _existing(inputs) == []
    monitor.assert_not_called()


_TUNED_PARAMETERS: dict[str, ParamValue] = {
    "analyzer": "word",
    "ngram_max": 2,
    "c": 0.05,
    "class_weight": "balanced",
    "min_df": 2,
    "max_df": 0.9,
    "sublinear_tf": False,
}


def _write_tuning(inputs: TrainingPaths, path: Path, **changes: object) -> Path:
    write_model(
        path,
        tuning_manifest(
            best=_TUNED_PARAMETERS,
            **(
                {
                    "dataset_sha256": file_sha256(inputs.cohort),
                    "split_sha256": ParquetSplitTable(inputs.split)
                    .read()
                    .split.sha256(),
                }
                | changes
            ),
        ),
    )
    return path


def test_training_uses_tuned_parameters_threshold_and_provenance(
    tmp_path: Path,
) -> None:
    """Načtený model nese přesné parametry manifestu a trénink jeho otisk i práh."""
    inputs = _inputs(tmp_path, parameters=None)
    tuning = _write_tuning(inputs, tmp_path / "tuning.json")

    assert (
        _run(
            replace(inputs, tuning=tuning),
        )
        is ExitCode.OK
    )

    model = SkopsModelFile(inputs.model).load()
    vectorizer = model.model.estimator_["tfidf"]
    svm = model.model.estimator_["svm"]
    assert vectorizer.analyzer == "word"
    assert vectorizer.ngram_range == (1, 2)
    assert vectorizer.min_df == 2
    assert vectorizer.max_df == 0.9  # ruff: ignore[float-equality-comparison]
    assert vectorizer.sublinear_tf is False
    assert svm.C == 0.05  # ruff: ignore[float-equality-comparison]
    assert svm.class_weight == "balanced"
    manifest = _manifest(inputs)
    assert manifest.tuning_manifest_sha256 == file_sha256(tuning)
    assert manifest.parameters == _TUNED_PARAMETERS
    assert manifest.threshold == model.threshold
    assert manifest.converged is True


@pytest.mark.parametrize("both", [True, False])
def test_training_requires_exactly_one_parameter_source(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    *,
    both: bool,
) -> None:
    """Chybějící i dvojitý zdroj parametrů odmítne běh před tréninkem."""
    inputs = _inputs(tmp_path, **({} if both else {"parameters": None}))
    tuning = _write_tuning(inputs, tmp_path / "tuning.json") if both else None
    fit = Mock(side_effect=AssertionError("Nejednoznačný zdroj nesmí trénovat."))
    monkeypatch.setattr(model_training.LinearSvmTrainer, "fit", fit)

    assert (
        _run(
            replace(inputs, tuning=tuning),
        )
        is ExitCode.REFUSED
    )
    assert "právě jeden" in caplog.text
    assert _existing(inputs) == []
    fit.assert_not_called()


@pytest.mark.parametrize(
    ("field", "other"),
    [
        ("fold", 1),
        ("dataset_sha256", "f" * 64),
        ("split_sha256", "f" * 64),
        ("diagnostic_strip", {"sections": ["diagnosis"], "lexicon_sha256": "f" * 64}),
        ("text_preprocessing", "normalized"),
    ],
)
def test_training_refuses_tuning_with_different_lineage(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    other: object,
) -> None:
    """Každý počítaný vstup se porovná s tréninkem a neshoda zabrání fitu."""
    inputs = _inputs(tmp_path, parameters=None)
    changes = {field: other}
    mismatched = field
    if field == "text_preprocessing":
        changes["text_normalization"] = {"acronyms": {}, "variants": {}}
        mismatched += ", text_normalization"
    tuning = _write_tuning(inputs, tmp_path / "tuning.json", **changes)
    fit = Mock(side_effect=AssertionError("Cizí původ nesmí trénovat."))
    monkeypatch.setattr(model_training.LinearSvmTrainer, "fit", fit)

    assert (
        _run(
            replace(inputs, tuning=tuning),
        )
        is ExitCode.REFUSED
    )
    assert (
        f"Manifest {tuning}: Manifest ladění neodpovídá tréninku: {mismatched}."
        in caplog.messages
    )
    assert _existing(inputs) == []
    fit.assert_not_called()


def test_training_manifest_records_nonconvergence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nekonvergence skutečného fitu se propíše do jediného trvalého záznamu v manifestu."""
    original = sklearn_classifier.svm_estimator
    monkeypatch.setattr(
        sklearn_classifier,
        "svm_estimator",
        lambda params, seed: original(params, seed).set_params(max_iter=1),
    )
    paths = _inputs(tmp_path)

    assert _run(paths) is ExitCode.OK
    assert _manifest(paths).converged is False
