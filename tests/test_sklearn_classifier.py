"""Adaptéry scikit-learn: odložení po pacientech, trénink, uložení modelu a metriky."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest

from mediparse.domain.labels import Diagnosis
from mediparse.infrastructure.sklearn_classifier import (
    GroupedHoldout,
    LinearSvmTrainer,
    SklearnScorer,
    SkopsModelFile,
)
from tests.support import separable_notes

if TYPE_CHECKING:
    from pathlib import Path

FOLDS: Final = 2
PATIENTS: Final = 12


def _corpus() -> tuple[list[str], list[bool], list[int]]:
    notes = separable_notes(PATIENTS)
    return (
        [note.text for note in notes],
        [Diagnosis.DIABETES in note.labels for note in notes],
        [note.subject_id for note in notes],
    )


def test_holdout_keeps_patient_in_one_part() -> None:
    """Žádný pacient není v tréninku i testu a části pokrývají všechny zprávy."""
    _, labels, groups = _corpus()

    holdout = GroupedHoldout(folds=FOLDS, seed=0).split(labels, groups)

    train = {groups[index] for index in holdout.train}
    test = {groups[index] for index in holdout.test}
    assert train.isdisjoint(test)
    assert sorted(holdout.train + holdout.test) == list(range(len(groups)))


def test_holdout_is_deterministic_for_seed() -> None:
    """Stejný seed dá stejný split."""
    _, labels, groups = _corpus()

    first = GroupedHoldout(folds=FOLDS, seed=3).split(labels, groups)
    second = GroupedHoldout(folds=FOLDS, seed=3).split(labels, groups)

    assert first == second
    assert any(
        GroupedHoldout(folds=FOLDS, seed=seed).split(labels, groups) != first
        for seed in range(4, 12)
    )


def test_trainer_predicts_training_examples() -> None:
    """Trénink na separovatelném korpusu predikuje trénovací příklady."""
    texts, labels, _ = _corpus()

    classifier = LinearSvmTrainer(regularization=1.0, seed=0).fit(texts, labels)

    assert classifier.predict(texts) == tuple(labels)


def test_model_file_round_trip(tmp_path: Path) -> None:
    """Uložený model se načte s trusted=[] a predikuje shodně."""
    texts, labels, _ = _corpus()
    classifier = LinearSvmTrainer(regularization=1.0, seed=0).fit(texts, labels)
    model_file = SkopsModelFile(tmp_path / "out" / "model.skops")

    model_file.save(classifier)
    loaded = model_file.load()

    assert loaded.predict(texts) == classifier.predict(texts)


def test_model_file_rejects_foreign_classifier(tmp_path: Path) -> None:
    """Uložit lze jen klasifikátor z tohoto adaptéru."""

    class Foreign:
        @staticmethod
        def predict(texts: list[str]) -> tuple[bool, ...]:
            return tuple(bool(text) for text in texts)

    with pytest.raises(TypeError):
        SkopsModelFile(tmp_path / "model.skops").save(Foreign())


def test_scorer_on_known_example() -> None:
    """Dvě ze tří pozitivních, jeden falešný poplach."""
    truth = [True, True, True, False, False]
    predicted = [True, True, False, True, False]

    metrics = SklearnScorer().score(truth, predicted)

    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(2 / 3)
    assert metrics.accuracy == pytest.approx(3 / 5)


def test_scorer_without_predicted_positives_is_zero() -> None:
    """Bez predikovaných pozitivních je přesnost 0.0 a bez varování."""
    metrics = SklearnScorer().score([True, False], [False, False])

    assert (metrics.precision, metrics.recall, metrics.f1) == (0.0, 0.0, 0.0)
    assert metrics.accuracy == pytest.approx(0.5)
