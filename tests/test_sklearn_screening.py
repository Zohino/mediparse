"""Adaptéry screeningu: dělení pacientů, odezva a konvergence skutečných estimatorů."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, Self, override

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from mediparse.domain.classical_screening import ClassifierFamily, VectorizerSpec
from mediparse.domain.clinical_text import TextNormalization
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.sklearn_classifier import tfidf_vectorizer
from mediparse.infrastructure.sklearn_screening import (
    SklearnThresholdTuner,
    TfidfFeaturizer,
    classifier_estimator,
    threshold_splits,
)
from tests.model_support import OVERLAPPING, separable_notes

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from mediparse.domain.seed import Seed


@st.composite
def _patient_labels(draw: st.DrawFn) -> tuple[list[int], list[bool], int]:
    folds = draw(st.integers(2, 4))
    labels = draw(st.lists(st.booleans(), min_size=2 * folds, max_size=30))
    labels[: 2 * folds] = [False, True] * folds
    visits = draw(
        st.lists(st.integers(1, 4), min_size=len(labels), max_size=len(labels))
    )
    rows = [
        (subject, label)
        for subject, (label, count) in enumerate(zip(labels, visits, strict=True))
        for _ in range(count)
    ]
    rows = list(draw(st.permutations(rows)))
    return [subject for subject, _ in rows], [label for _, label in rows], folds


@given(_patient_labels())
def test_threshold_splits_separate_patients_and_cover_every_index_once(
    corpus: tuple[list[int], list[bool], int],
) -> None:
    """Žádný pacient nepřekročí hranici a každý index je právě jednou v testu."""
    subjects, labels, folds = corpus
    splits = threshold_splits(labels, subjects, folds, 7)
    tested = []
    for train, test in splits:
        assert {subjects[i] for i in train}.isdisjoint(subjects[i] for i in test)
        assert sorted([*train, *test]) == list(range(len(subjects)))
        tested.extend(test)
    assert len(splits) == folds
    assert sorted(tested) == list(range(len(subjects)))


@pytest.mark.parametrize("family", ClassifierFamily)
@pytest.mark.parametrize(
    ("analyzer", "ngram_range"), [("word", (1, 2)), ("char_wb", (3, 5))]
)
def test_prediction_matches_threshold_on_score_scale(
    family: ClassifierFamily,
    analyzer: Literal["word", "char_wb"],
    ngram_range: tuple[int, int],
) -> None:
    """Všechny rodiny a analyzátory vracejí predikci při prahu na téže škále skóre."""
    notes = separable_notes(24, OVERLAPPING)
    train, test = TfidfFeaturizer(None).fit_transform(
        VectorizerSpec(name="test", analyzer=analyzer, ngram_range=ngram_range),
        [note.text for note in notes],
        ["diabetes clinic", "fracture review", "insulin splint visit"],
    )
    model = SklearnThresholdTuner().fit(
        family,
        train,
        [note.has_label(Diagnosis.DIABETES) for note in notes],
        [note.subject_id for note in notes],
        3,
        7,
    )
    predictions = model.classify(test)
    assert len(predictions) == test.shape[0]
    assert all(
        item.predicted == (item.score >= model.threshold) for item in predictions
    )


def _nonconverging(family: ClassifierFamily, seed: Seed) -> ClassifierMixin:
    del family, seed
    return LogisticRegression(max_iter=1)


@pytest.mark.parametrize(
    ("factory", "converged"), [(classifier_estimator, True), (_nonconverging, False)]
)
def test_convergence_includes_inner_fits_and_refit(
    factory: Callable[[ClassifierFamily, Seed], ClassifierMixin], *, converged: bool
) -> None:
    """Nekonvergence je výsledek i pod filterwarnings=error, výchozí LR konverguje."""
    notes = separable_notes(24, OVERLAPPING)
    features, _ = TfidfFeaturizer(None).fit_transform(
        VectorizerSpec(name="word", analyzer="word", ngram_range=(1, 1)),
        [note.text for note in notes],
        ["clinic review"],
    )
    model = SklearnThresholdTuner(estimator=factory).fit(
        ClassifierFamily.LOGREG,
        features,
        [note.has_label(Diagnosis.DIABETES) for note in notes],
        [note.subject_id for note in notes],
        3,
        7,
    )
    assert model.converged is converged


def test_empty_vocabulary_is_invalid_input() -> None:
    """Texty bez tokenu odmítne doménová výjimka."""
    with pytest.raises(InvalidInputError, match="prázdný"):
        TfidfFeaturizer(None).fit_transform(
            VectorizerSpec(name="word", analyzer="word", ngram_range=(1, 1)),
            ["!", "?"],
            ["clinic"],
        )


def test_char_ngrams_apply_shared_preprocessor() -> None:
    """Znakové příznaky vznikají z rozvinutého COPD, ručně určené n-gramy mají 3 až 5 znaků."""
    vectorizer = tfidf_vectorizer(
        TextNormalization(
            acronyms={"COPD": "chronic obstructive pulmonary disease"}, variants={}
        ),
        analyzer="char_wb",
        ngram_range=(3, 5),
    )
    vectorizer.fit_transform(["COPD"])
    vocabulary = set(vectorizer.get_feature_names_out())
    assert {"chr", "chron", "pulmo", " dise"} <= vocabulary
    assert "cop" not in vocabulary
    minimum, maximum = 3, 5
    assert all(minimum <= len(feature) <= maximum for feature in vocabulary)


def test_featurizer_fits_only_training_vocabulary() -> None:
    """Slovo přítomné pouze v testu nepřidá příznak do tréninkového slovníku."""
    train, test = TfidfFeaturizer(None).fit_transform(
        VectorizerSpec(name="word", analyzer="word", ngram_range=(1, 2)),
        ["alpha beta"],
        ["gamma"],
    )
    expected_features = 3
    assert train.shape == (1, expected_features)
    assert test.shape == train.shape
    assert test.nnz == 0


def _observing_estimator(
    on_fit: Callable[[int], None],
) -> Callable[[ClassifierFamily, Seed], ClassifierMixin]:
    class _ObservedLogreg(LogisticRegression):
        @override
        def fit(
            self,
            X: csr_matrix,
            y: Sequence[bool],
            sample_weight: Sequence[float] | None = None,
        ) -> Self:
            on_fit(len(y))
            return super().fit(X, y, sample_weight=sample_weight)

    def factory(family: ClassifierFamily, seed: Seed) -> ClassifierMixin:
        del family
        return _ObservedLogreg(random_state=seed)

    return factory


def test_tuner_preserves_nonconvergence_unrelated_warning() -> None:
    """Tuner předá volajícímu UserWarning a nepovažuje ho za nekonvergenci."""
    notes = separable_notes(24, OVERLAPPING)
    features, _ = TfidfFeaturizer(None).fit_transform(
        VectorizerSpec(name="word", analyzer="word", ngram_range=(1, 1)),
        [note.text for note in notes],
        ["clinic review"],
    )

    def warn_on_fit(size: int) -> None:
        del size
        warnings.warn("Syntetické varování estimatoru.", UserWarning, stacklevel=2)

    with pytest.warns(UserWarning, match="Syntetické varování estimatoru"):
        model = SklearnThresholdTuner(estimator=_observing_estimator(warn_on_fit)).fit(
            ClassifierFamily.LOGREG,
            features,
            [note.has_label(Diagnosis.DIABETES) for note in notes],
            [note.subject_id for note in notes],
            3,
            7,
        )

    assert model.converged is True


class _FixedScoresClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, features: csr_matrix, labels: Sequence[bool]) -> Self:
        del features
        self.classes_ = np.unique(labels)
        return self

    @staticmethod
    def decision_function(features: csr_matrix) -> np.ndarray:
        return features[:, 0].toarray().ravel()


def test_tuner_uses_patient_cv_and_refits_all_training_rows() -> None:
    """Velikosti všech vnitřních fitů odpovídají zadaným částem pacientů, navíc je jeden refit."""
    notes = separable_notes(24, OVERLAPPING)
    labels = [note.has_label(Diagnosis.DIABETES) for note in notes]
    subjects = [note.subject_id for note in notes]
    features, _ = TfidfFeaturizer(None).fit_transform(
        VectorizerSpec(name="word", analyzer="word", ngram_range=(1, 1)),
        [note.text for note in notes],
        ["clinic review"],
    )
    sizes: list[int] = []
    expected = [
        len(labels) - len(test) for _, test in threshold_splits(labels, subjects, 3, 7)
    ]
    expected.append(len(labels))

    SklearnThresholdTuner(estimator=_observing_estimator(sizes.append)).fit(
        ClassifierFamily.LOGREG,
        features,
        labels,
        subjects,
        3,
        7,
    )

    assert sorted(sizes) == sorted(expected)


def test_tuner_maximizes_f1_instead_of_balanced_accuracy() -> None:
    """Každá část má 2 pozitivní a 98 negativních se skóre z nezávislého orákula.

    Nad 0,6 jsou TP=1, FP=0, FN=1, TN=98: F1=2/3, BA=3/4.
    Mezi 0,1 a 0,3 jsou TP=2, FP=8, FN=0, TN=90: F1=1/3, BA=47/49.
    Mezi 0,3 a 0,6 klesnou F1 na 2/11 a BA na 139/196, při prahu 0,1
    jsou F1=2/51 a BA=1/2. F1 tedy volí (0,6; 0,9], BA (0,1; 0,3].
    """
    scores = [0.9, *[0.6] * 8, 0.3, *[0.1] * 90]
    part_labels = [True, *[False] * 8, True, *[False] * 90]
    features = csr_matrix([[score] for _ in range(3) for score in scores])
    labels = part_labels * 3
    subjects = [subject for subject in range(3) for _ in scores]
    tuner = SklearnThresholdTuner(
        estimator=lambda _family, _seed: _FixedScoresClassifier()
    )

    classifier = tuner.fit(ClassifierFamily.LOGREG, features, labels, subjects, 3, 7)

    lower, upper = 0.6, 0.9
    assert lower < classifier.threshold <= upper


@pytest.mark.parametrize("phase", ["inner", "refit"])
def test_tuner_detects_convergence_warning_in_each_fit_phase(phase: str) -> None:
    """Samostatné varování pouze vnitřních fitů i pouze refitu znamená nekonvergenci."""
    notes = separable_notes(24, OVERLAPPING)
    features, _ = TfidfFeaturizer(None).fit_transform(
        VectorizerSpec(name="word", analyzer="word", ngram_range=(1, 1)),
        [note.text for note in notes],
        ["clinic review"],
    )

    def warn_in_phase(size: int) -> None:
        if (size < len(notes)) == (phase == "inner"):
            warnings.warn(
                "Syntetická nekonvergence fáze fitu.", ConvergenceWarning, stacklevel=2
            )

    model = SklearnThresholdTuner(estimator=_observing_estimator(warn_in_phase)).fit(
        ClassifierFamily.LOGREG,
        features,
        [note.has_label(Diagnosis.DIABETES) for note in notes],
        [note.subject_id for note in notes],
        3,
        7,
    )

    assert model.converged is False


def test_featurizer_logs_only_name_and_manual_counts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Log nese ručně spočtené počty 2/1 zpráv a 3 příznaků, nikoli vstupní tokeny."""
    with caplog.at_level("INFO", logger="mediparse.infrastructure.sklearn_screening"):
        TfidfFeaturizer(None).fit_transform(
            VectorizerSpec(name="word", analyzer="word", ngram_range=(1, 1)),
            ["quartz amber", "opal"],
            ["jasper"],
        )

    assert caplog.messages == [
        "Vektorizér word: 2 trénovacích a 1 testovacích zpráv, 3 příznaků."
    ]
    assert all(
        token not in caplog.text for token in ("quartz", "amber", "opal", "jasper")
    )
