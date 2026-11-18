"""Protokol screeningu nad syntetikou s fakes, skutečným dělením pacientů a metrikami."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from itertools import batched, product
from typing import TYPE_CHECKING, override

import pytest

from mediparse.application.ports.experiment import PatientPartitioner
from mediparse.application.screening import Screening, ThresholdTuner
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.evaluation import Classification
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.screening import (
    ClassifierFamily,
    ScreeningConfig,
    ScreeningReport,
    ScreeningResult,
    ScreeningUnit,
    VectorizerSpec,
)
from mediparse.infrastructure.screening_table import ParquetScreeningCheckpoint
from mediparse.infrastructure.sklearn_metrics import SklearnScorer
from mediparse.infrastructure.sklearn_split import StratifiedGroupFolds
from tests.screening_support import DEVELOPMENT, SHARED
from tests.screening_support import note as _note
from tests.screening_support import notes as _notes

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from mediparse.domain.identity import SubjectId
    from mediparse.domain.input_note import InputNote
    from mediparse.domain.seed import Seed
    from mediparse.domain.split import SharedSplit

type Features = tuple[str, ...]

ERRORS_PER_CLASS = 2
PARTITIONER = StratifiedGroupFolds()
SINGLE_POSITIVE_SUBJECT = 90000001

CONFIG = ScreeningConfig.model_validate_json(
    json.dumps({
        "seed": 7,
        "folds": 2,
        "threshold_folds": 3,
        "lsa_components": 2,
        "input": {"strip_diagnostic_sections": False, "text_preprocessing": "none"},
        "vectorizers": [
            {"name": "z-char", "analyzer": "char_wb", "ngram_range": [3, 5]},
            {"name": "a-word", "analyzer": "word", "ngram_range": [1, 2]},
        ],
        "families": ["linear_svm", "complement_nb", "logreg"],
    })
)


@dataclass(frozen=True)
class _Fit:
    family: ClassifierFamily
    features: Features
    labels: tuple[bool, ...]
    subjects: tuple[SubjectId, ...]
    folds: int
    seed: Seed


@dataclass
class _Recording:
    featurizations: list[tuple[VectorizerSpec, Features, Features]] = field(
        default_factory=list
    )
    fits: list[_Fit] = field(default_factory=list)
    classifications: list[Features] = field(default_factory=list)
    classified: list[tuple[Classification, ...]] = field(default_factory=list)
    writes: list[tuple[ScreeningResult, ...]] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Table:
    notes: tuple[InputNote, ...]

    requested: list[frozenset[SubjectId]] = field(default_factory=list)

    @staticmethod
    def read() -> tuple[InputNote, ...]:
        pytest.fail("Screening nesmí číst celou kohortu.")

    def subjects(self) -> frozenset[SubjectId]:
        return frozenset(note.subject_id for note in self.notes)

    def read_patients(self, subjects: frozenset[SubjectId]) -> tuple[InputNote, ...]:
        self.requested.append(subjects)
        return tuple(note for note in self.notes if note.subject_id in subjects)


@dataclass(frozen=True)
class _SplitSource:
    @staticmethod
    def read() -> SharedSplit:
        return SHARED


@dataclass(frozen=True)
class _Featurizer:
    recording: _Recording

    def fit_transform(
        self, spec: VectorizerSpec, train: Sequence[str], test: Sequence[str]
    ) -> tuple[Features, Features]:
        self.recording.featurizations.append((spec, tuple(train), tuple(test)))
        return tuple(train), tuple(test)


@dataclass(frozen=True)
class _Classifier:
    recording: _Recording
    threshold: float
    converged: bool

    def classify(self, features: Features) -> tuple[Classification, ...]:
        self.recording.classifications.append(features)
        seen = {True: 0, False: 0}
        predictions = []
        for index, text in enumerate(features):
            positive = "positive" in text
            predicted = not positive if seen[positive] < ERRORS_PER_CLASS else positive
            seen[positive] += 1
            score = (0.9 if predicted else 0.1) + index / (100 * len(features))
            predictions.append(Classification(predicted, score))
        classified = tuple(predictions)
        self.recording.classified.append(classified)
        return classified


@dataclass(frozen=True)
class _Tuner(ThresholdTuner[Features]):
    recording: _Recording
    fail_at: int | None = None

    @override
    def fit(
        self,
        family: ClassifierFamily,
        features: Features,
        labels: Sequence[bool],
        subjects: Sequence[SubjectId],
        folds: int,
        seed: Seed,
    ) -> _Classifier:
        if len(self.recording.fits) + 1 == self.fail_at:
            raise RuntimeError
        self.recording.fits.append(
            _Fit(family, features, tuple(labels), tuple(subjects), folds, seed)
        )
        threshold = {
            ClassifierFamily.COMPLEMENT_NB: 0.25,
            ClassifierFamily.LOGREG: 0.5,
            ClassifierFamily.LINEAR_SVM: 0.75,
        }[family]
        return _Classifier(
            self.recording, threshold, converged=family is ClassifierFamily.LOGREG
        )


@dataclass(frozen=True)
class _Progress:
    done: frozenset[ScreeningUnit]
    events: list[str]
    writes: list[tuple[ScreeningResult, ...]]

    def completed(self) -> frozenset[ScreeningUnit]:
        self.events.append("completed")
        return self.done

    def write(self, results: Sequence[ScreeningResult]) -> None:
        self.events.append("write")
        self.writes.append(tuple(results))

    def finish(self) -> None:
        self.events.append("finish")


def _step(
    recording: _Recording,
    notes: tuple[InputNote, ...],
    partitioner: PatientPartitioner = PARTITIONER,
) -> Screening[Features]:
    return Screening(
        table=_Table(notes),
        split=_SplitSource(),
        partitioner=partitioner,
        featurizer=_Featurizer(recording),
        tuner=_Tuner(recording),
        scorer=SklearnScorer(),
        progress=_Progress(frozenset(), recording.events, recording.writes),
    )


def test_only_development_texts_reach_featurizer_without_patient_leak() -> None:
    """Veškeré texty jsou přesně vývojové; pacient s více zprávami nepřekročí hranici."""
    notes, recording = _notes(), _Recording()

    _step(recording, notes).run(CONFIG)

    by_text = {note.text: note for note in notes}
    seen = set()
    for _, train, test in recording.featurizations:
        seen.update((*train, *test))
        assert {by_text[text].subject_id for text in train}.isdisjoint(
            by_text[text].subject_id for text in test
        )
    assert seen == {note.text for note in notes if note.subject_id in DEVELOPMENT}
    assert len(recording.featurizations) == CONFIG.folds * len(CONFIG.vectorizers)
    assert [spec for spec, _, _ in recording.featurizations] == list(
        CONFIG.vectorizers
    ) * CONFIG.folds


def test_tuner_gets_only_training_labels_subjects_and_features() -> None:
    """Práh se volí jen z tréninku diagnózy; klasifikátor dostane jen příznaky testu."""
    notes, recording = _notes(), _Recording()

    _step(recording, notes).run(CONFIG)

    by_text = {note.text: note for note in notes}
    cells = tuple(product(CONFIG.families, Diagnosis))
    fit_groups = batched(recording.fits, len(cells), strict=True)
    classified_groups = batched(recording.classifications, len(cells), strict=True)
    for (_, train, test), fits, classified in zip(
        recording.featurizations, fit_groups, classified_groups, strict=True
    ):
        for (family, diagnosis), fit in zip(cells, fits, strict=True):
            assert fit == _Fit(
                family,
                train,
                tuple(by_text[text].has_label(diagnosis) for text in train),
                tuple(by_text[text].subject_id for text in train),
                CONFIG.threshold_folds,
                CONFIG.seed,
            )
        assert classified == (test,) * len(cells)


def test_metrics_match_manual_confusion_matrix() -> None:
    """TP 8, FP 2, FN 2, TN 8 dává čtyři metriky 0,8 přes reálný SklearnScorer."""
    recording = _Recording()

    _step(recording, _notes()).run(CONFIG)

    rows = [
        row
        for write in recording.writes
        for row in write
        if row.diagnosis is Diagnosis.DIABETES
    ]
    assert rows
    for row in rows:
        assert (row.test_notes, row.test_positives) == (20, 10)
        assert row.metrics.precision == pytest.approx(0.8)
        assert row.metrics.recall == pytest.approx(0.8)
        assert row.metrics.f1 == pytest.approx(0.8)
        assert row.metrics.accuracy == pytest.approx(0.8)


def test_each_unit_is_written_once_in_config_order() -> None:
    """Dvanáct zápisů uchová pořadí mřížky a pět diagnóz každé jednotky."""
    recording = _Recording()
    _step(recording, _notes()).run(CONFIG)
    assert recording.events == ["completed", *["write"] * 12, "finish"]
    assert [
        (rows[0].fold, rows[0].vectorizer, rows[0].family) for rows in recording.writes
    ] == list(product(range(2), ("z-char", "a-word"), CONFIG.families))
    for rows in recording.writes:
        assert [row.diagnosis for row in rows] == list(Diagnosis)
        assert len({(row.fold, row.vectorizer, row.family) for row in rows}) == 1


def test_completed_units_are_skipped_without_refit() -> None:
    """Čtyři hotové jednotky ušetří dvacet fitů a první featurizaci."""
    recording = _Recording()
    units = list(product(range(2), ("z-char", "a-word"), CONFIG.families))
    done = frozenset(ScreeningUnit(*unit) for unit in units[:4])
    step = replace(
        _step(recording, _notes()),
        progress=_Progress(done, recording.events, recording.writes),
    )
    step.run(CONFIG)
    assert len(recording.fits) == 40
    assert [spec for spec, _, _ in recording.featurizations] == [
        CONFIG.vectorizers[1],
        CONFIG.vectorizers[0],
        CONFIG.vectorizers[1],
    ]
    assert [
        (rows[0].fold, rows[0].vectorizer, rows[0].family) for rows in recording.writes
    ] == units[4:]


def test_cells_preserve_test_identity_labels_and_classifications() -> None:
    """Buňka zachová pořadí testu, jeho labely a přesné výstupy klasifikátoru."""
    notes, recording = _notes(), _Recording()

    _step(recording, notes).run(CONFIG)

    by_text = {note.text: note for note in notes}
    for row, texts, classified in zip(
        [row for write in recording.writes for row in write],
        recording.classifications,
        recording.classified,
        strict=True,
    ):
        assert row.note_ids == tuple(by_text[text].note_id for text in texts)
        assert row.labels == tuple(
            by_text[text].has_label(row.diagnosis) for text in texts
        )
        assert row.classifications is classified


@pytest.mark.parametrize(
    ("family", "threshold", "converged"),
    [
        (ClassifierFamily.COMPLEMENT_NB, 0.25, False),
        (ClassifierFamily.LOGREG, 0.5, True),
        (ClassifierFamily.LINEAR_SVM, 0.75, False),
    ],
    ids=["complement-nb", "logreg", "linear-svm"],
)
def test_rows_carry_tuner_metadata(
    family: ClassifierFamily, threshold: float, *, converged: bool
) -> None:
    """Každý řádek přenese práh a konvergenci tuneru své rodiny."""
    recording = _Recording()

    _step(recording, _notes()).run(CONFIG)

    rows = [row for write in recording.writes for row in write if row.family is family]
    assert rows
    for row in rows:
        assert row.threshold == pytest.approx(threshold)
        assert row.converged is converged


def test_report_describes_development_and_original_split() -> None:
    """Počítají se přítomné vývojové zprávy a pacienti, provenance nese původní split."""
    report = _step(_Recording(), _notes()).run(CONFIG)

    assert report == ScreeningReport(40, 20, SHARED.config, SHARED.split.sha256())


@dataclass
class _RecordingPartitioner:
    calls: list[tuple[int, Seed]] = field(default_factory=list)

    def partition(
        self,
        subjects: Sequence[SubjectId],
        strata: Sequence[int],
        parts: int,
        seed: Seed,
    ) -> tuple[frozenset[SubjectId], ...]:
        self.calls.append((parts, seed))
        return PARTITIONER.partition(subjects, strata, parts, seed)


def test_partitioner_receives_screening_folds_and_seed() -> None:
    """Skutečné dělení dostane počet vnějších foldů a seed screeningu."""
    partitioner = _RecordingPartitioner()

    _step(_Recording(), _notes(), partitioner).run(CONFIG)

    assert partitioner.calls == [(CONFIG.folds, CONFIG.seed)]


@dataclass(frozen=True)
class _FixedPartitioner(PatientPartitioner):
    @staticmethod
    @override
    def partition(
        subjects: Sequence[SubjectId],
        strata: Sequence[int],
        parts: int,
        seed: Seed,
    ) -> tuple[frozenset[SubjectId], ...]:
        return (
            frozenset(range(90000001, 90000007)),
            frozenset(range(90000007, 90000013)),
        )


def test_invalid_last_fold_is_refused_before_any_fit_or_write() -> None:
    """První test má 10+10 zpráv, druhý 9+10; chyba neobsahuje syntetické id."""
    notes = tuple(
        _note(subject, visit, positive=subject % 2 == 1)
        for subject, visits in [
            (90000001, 4),
            (90000002, 4),
            (90000003, 3),
            (90000004, 3),
            (90000005, 3),
            (90000006, 3),
            (90000007, 3),
            (90000008, 4),
            (90000009, 3),
            (90000010, 3),
            (90000011, 3),
            (90000012, 3),
        ]
        for visit in range(visits)
    )
    recording = _Recording()

    with pytest.raises(
        InvalidInputError,
        match=(
            r"^Diagnóza diabetes: 9 pozitivních a 10 negativních zpráv, "
            r"metriky potřebují v každé třídě aspoň 10\.$"
        ),
    ) as raised:
        _step(recording, notes, _FixedPartitioner()).run(CONFIG)

    assert "9000000" not in str(raised.value)
    assert "note_id" not in str(raised.value)
    assert "subject_id" not in str(raised.value)
    assert recording == _Recording()


def test_class_check_counts_development_patients_not_notes() -> None:
    """Jediný pozitivní vývojový pacient nestačí ani s více zprávami a pozitivní evaluací."""
    notes = tuple(
        replace(note, labels=note.labels - {Diagnosis.DIABETES})
        if note.subject_id in DEVELOPMENT and note.subject_id != SINGLE_POSITIVE_SUBJECT
        else note
        for note in _notes()
    )
    recording = _Recording()

    with pytest.raises(
        InvalidInputError,
        match=(
            r"^Diagnóza diabetes: 1 pozitivních a 19 negativních pacientů, "
            r"odložení potřebuje v každé třídě aspoň 2\.$"
        ),
    ):
        _step(recording, notes).run(CONFIG)

    assert recording == _Recording()


def test_threshold_cv_requires_enough_training_patients_before_any_fit() -> None:
    """Vývoj má dvě třídy po dvou pacientech, trénink foldu jen po jednom místo tří."""
    notes = tuple(
        _note(subject, visit, positive=subject % 2 == 1)
        for subject in range(90000001, 90000005)
        for visit in range(10)
    )
    recording = _Recording()

    with pytest.raises(
        InvalidInputError,
        match=(
            r"^Diagnóza diabetes: 1 pozitivních a 1 negativních pacientů, "
            r"odložení potřebuje v každé třídě aspoň 3\.$"
        ),
    ) as raised:
        _step(recording, notes).run(CONFIG)

    assert "9000000" not in str(raised.value)
    assert recording == _Recording()


def test_empty_development_is_refused() -> None:
    """Prázdná vývojová část nevytvoří prázdný výsledek ani fit."""
    recording = _Recording()

    with pytest.raises(
        InvalidInputError,
        match=(
            r"^Diagnóza diabetes: 0 pozitivních a 0 negativních pacientů, "
            r"odložení potřebuje v každé třídě aspoň 2\.$"
        ),
    ):
        _step(recording, ()).run(CONFIG)

    assert recording == _Recording()


def test_table_gets_exact_development_patients() -> None:
    """Screening čte přesně pacienty vývoje; fake odmítá čtení celé tabulky."""
    step = _step(_Recording(), _notes())
    table = _Table(_notes())

    replace(step, table=table).run(CONFIG)

    assert table.requested == [SHARED.split.development]


def test_outside_patient_is_rejected_before_reading_notes() -> None:
    """Pacient mimo split se odmítne před prvním čtením textů."""
    table = _Table((*_notes(), _note(999, 1, positive=False)))
    step = replace(_step(_Recording(), ()), table=table)

    with pytest.raises(InvalidInputError, match=r"^Pacienti mimo split: 1\.$"):
        step.run(CONFIG)

    assert table.requested == []


def test_interrupted_run_resumes_to_identical_outputs(tmp_path: Path) -> None:
    """Pád na 23. fitu ponechá čtyři jednotky a obnova dá shodné bajty."""
    checkpoint = ParquetScreeningCheckpoint(
        tmp_path / "checkpoint", "key", tmp_path / "results", tmp_path / "predictions"
    )
    recording = _Recording()
    with pytest.raises(RuntimeError):
        replace(
            _step(recording, _notes()), progress=checkpoint, tuner=_Tuner(recording, 23)
        ).run(CONFIG)
    assert checkpoint.completed() == frozenset(
        ScreeningUnit(*unit)
        for unit in list(product(range(2), ("z-char", "a-word"), CONFIG.families))[:4]
    )
    resumed = _Recording()
    replace(_step(resumed, _notes()), progress=checkpoint).run(CONFIG)
    assert len(resumed.fits) == 40
    whole = ParquetScreeningCheckpoint(
        tmp_path / "whole",
        "key",
        tmp_path / "whole-results",
        tmp_path / "whole-predictions",
    )
    replace(_step(_Recording(), _notes()), progress=whole).run(CONFIG)
    assert checkpoint.results.read_bytes() == whole.results.read_bytes()
    assert checkpoint.predictions.read_bytes() == whole.predictions.read_bytes()
    assert not checkpoint.directory.exists()
