"""Protokol screeningu nad syntetikou s fakes, skutečným dělením pacientů a metrikami."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from itertools import batched, product
from typing import TYPE_CHECKING, override

import pytest

from mediparse.application.classical_screening import ClassicalScreening, ThresholdTuner
from mediparse.application.ports.experiment import PatientPartitioner
from mediparse.domain.classical_screening import (
    ClassicalScreeningConfig,
    ClassifierFamily,
    ScreeningReport,
    ScreeningResult,
    VectorizerSpec,
)
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.evaluation import Classification
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.sklearn_metrics import SklearnScorer
from mediparse.infrastructure.sklearn_split import StratifiedGroupFolds
from tests.screening_support import DEVELOPMENT, SHARED
from tests.screening_support import note as _note
from tests.screening_support import notes as _notes

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.domain.identity import SubjectId
    from mediparse.domain.input_note import InputNote
    from mediparse.domain.seed import Seed
    from mediparse.domain.split import SharedSplit

type Features = tuple[str, ...]

ERRORS_PER_CLASS = 2
PARTITIONER = StratifiedGroupFolds()
SINGLE_POSITIVE_SUBJECT = 90000001

CONFIG = ClassicalScreeningConfig.model_validate_json(
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
    prediction_writes: list[tuple[ScreeningResult, ...]] = field(default_factory=list)


@dataclass(frozen=True)
class _Table:
    notes: tuple[InputNote, ...]

    def read(self) -> tuple[InputNote, ...]:
        return self.notes


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
class _Sink:
    writes: list[tuple[ScreeningResult, ...]]

    def write(self, results: Sequence[ScreeningResult]) -> None:
        self.writes.append(tuple(results))


def _step(
    recording: _Recording,
    notes: tuple[InputNote, ...],
    partitioner: PatientPartitioner = PARTITIONER,
) -> ClassicalScreening[Features]:
    return ClassicalScreening(
        table=_Table(notes),
        split=_SplitSource(),
        partitioner=partitioner,
        featurizer=_Featurizer(recording),
        tuner=_Tuner(recording),
        scorer=SklearnScorer(),
        results=_Sink(recording.writes),
        predictions=_Sink(recording.prediction_writes),
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

    rows = [row for row in recording.writes[0] if row.diagnosis is Diagnosis.DIABETES]
    assert rows
    for row in rows:
        assert (row.test_notes, row.test_positives) == (20, 10)
        assert row.metrics.precision == pytest.approx(0.8)
        assert row.metrics.recall == pytest.approx(0.8)
        assert row.metrics.f1 == pytest.approx(0.8)
        assert row.metrics.accuracy == pytest.approx(0.8)


def test_rows_follow_config_order_in_single_write() -> None:
    """Jediný zápis řadí fold, vektorizér, rodinu a diagnózu podle protokolu."""
    recording = _Recording()

    _step(recording, _notes()).run(CONFIG)

    assert len(recording.writes) == 1
    assert len(recording.prediction_writes) == 1
    rows = recording.writes[0]
    assert all(
        result is prediction
        for result, prediction in zip(rows, recording.prediction_writes[0], strict=True)
    )
    assert [(r.fold, r.vectorizer, r.family, r.diagnosis) for r in rows] == list(
        product(range(2), ("z-char", "a-word"), CONFIG.families, Diagnosis)
    )


def test_cells_preserve_test_identity_labels_and_classifications() -> None:
    """Buňka zachová pořadí testu, jeho labely a přesné výstupy klasifikátoru."""
    notes, recording = _notes(), _Recording()

    _step(recording, notes).run(CONFIG)

    by_text = {note.text: note for note in notes}
    for row, texts, classified in zip(
        recording.writes[0],
        recording.classifications,
        recording.classified,
        strict=True,
    ):
        assert row.note_ids == tuple(by_text[text].note_id for text in texts)
        assert row.labels == tuple(
            by_text[text].has_label(row.diagnosis) for text in texts
        )
        assert row.classifications is classified


@pytest.mark.parametrize("field", ["note_ids", "labels", "classifications"])
def test_cell_rejects_mismatched_lengths(field: str) -> None:
    """Každá z n-tic musí mít shodnou délku se zbývajícími dvěma."""
    recording = _Recording()
    _step(recording, _notes()).run(CONFIG)
    with pytest.raises(ValueError, match="délk"):
        replace(recording.writes[0][0], **{field: ()})


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

    rows = [row for row in recording.writes[0] if row.family is family]
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
