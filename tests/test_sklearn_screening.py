"""Adaptéry screeningu: dělení pacientů, odezva a konvergence skutečných estimatorů."""

from __future__ import annotations

import re
import warnings
from random import Random
from typing import TYPE_CHECKING, Literal, Self, override

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from mediparse.domain.classical_screening import ClassifierFamily, VectorizerSpec
from mediparse.domain.clinical_text import TextNormalization
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.sklearn_classifier import tfidf_vectorizer
from mediparse.infrastructure.sklearn_screening import (
    Lsa,
    ScreeningFeatures,
    SklearnThresholdTuner,
    TfidfFeaturizer,
    classifier_estimator,
    screening_lsa,
    threshold_splits,
)
from tests.model_support import OVERLAPPING, separable_notes

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from mediparse.domain.seed import Seed


def _word_training() -> tuple[ScreeningFeatures, list[bool], list[int]]:
    """Připraví překrývající se korpus pro testy tuneru.

    Returns:
        Slovní unigramy, labely diabetu a pacienti v pořadí řádků.
    """
    notes = separable_notes(24, OVERLAPPING)
    features, _ = TfidfFeaturizer(None).fit_transform(
        VectorizerSpec(name="word", analyzer="word", ngram_range=(1, 1)),
        [note.text for note in notes],
        ["clinic review"],
    )
    return (
        features,
        [note.has_label(Diagnosis.DIABETES) for note in notes],
        [note.subject_id for note in notes],
    )


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
    train, test = TfidfFeaturizer(None, Lsa(2, 7)).fit_transform(
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
    assert len(predictions) == test.sparse.shape[0]
    assert all(
        item.predicted == (item.score >= model.threshold) for item in predictions
    )


@pytest.mark.parametrize(
    "family", [ClassifierFamily.LINEAR_SVM, ClassifierFamily.NB_SVM]
)
def test_svm_factory_preserves_seed(family: ClassifierFamily) -> None:
    """Obě rodiny s duálním řešičem dostanou seed screeningu."""
    assert classifier_estimator(family, 13).get_params()["random_state"] == 13  # ruff: ignore[magic-value-comparison]


def test_lightgbm_factory_preserves_determinism_parameters() -> None:
    """Veřejné parametry LightGBM drží seed a determinismus; SVD patří featurizéru."""
    estimator = classifier_estimator(ClassifierFamily.LIGHTGBM_LSA, 13)
    params = estimator.get_params()

    expected = {
        "random_state": 13,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }
    assert {key: params[key] for key in expected} == expected


def test_lightgbm_repeats_nonconstant_scores_and_threshold() -> None:
    """Na 240 syntetických zprávách se stromy dělí a opakovaný seed opakuje skóre i práh."""
    random = Random(31)
    vocabulary = [f"word{i}" for i in range(300)]
    labels = [i % 2 == 0 for i in range(240)]
    texts = [
        " ".join([
            *random.sample(vocabulary, 20),
            "positive" if label else "negative",
        ])
        for label in labels
    ]
    spec = VectorizerSpec(name="word", analyzer="word", ngram_range=(1, 1))
    features, _ = TfidfFeaturizer(None, Lsa(8, 7)).fit_transform(spec, texts, texts)
    repeated, _ = TfidfFeaturizer(None, Lsa(8, 7)).fit_transform(spec, texts, texts)
    subjects = list(range(len(labels)))
    tuner = SklearnThresholdTuner()
    first = tuner.fit(ClassifierFamily.LIGHTGBM_LSA, features, labels, subjects, 3, 7)
    second = tuner.fit(ClassifierFamily.LIGHTGBM_LSA, repeated, labels, subjects, 3, 7)
    different = tuner.fit(
        ClassifierFamily.LIGHTGBM_LSA, features, labels, subjects, 3, 11
    )

    predictions = first.classify(features)
    assert len({prediction.score for prediction in predictions}) > 1
    assert second.classify(repeated) == predictions
    assert second.threshold == first.threshold
    assert different.classify(features) != predictions
    assert first.converged
    assert second.converged
    assert different.converged


def test_lsa_fits_once_on_outer_training_and_only_transforms_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Jediný fit SVD vidí přesně vnější trénink, test se jen transformuje."""
    notes = separable_notes(24, OVERLAPPING)
    labels = [note.has_label(Diagnosis.DIABETES) for note in notes]
    subjects = [note.subject_id for note in notes]
    observed: list[csr_matrix] = []
    transformed: list[csr_matrix] = []
    seeds: list[tuple[int, object]] = []
    fit_transform = TruncatedSVD.fit_transform
    transform = TruncatedSVD.transform

    def observe_fit(
        svd: TruncatedSVD, features: csr_matrix, labels: object = None
    ) -> np.ndarray:
        observed.append(features.copy())
        seeds.append((svd.n_components, svd.random_state))
        return fit_transform(svd, features, labels)

    def observe_transform(svd: TruncatedSVD, features: csr_matrix) -> np.ndarray:
        transformed.append(features.copy())
        return transform(svd, features)

    def refuse_fit(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("Tuner nesmí znovu učit SVD.")

    monkeypatch.setattr(TruncatedSVD, "fit_transform", observe_fit)
    monkeypatch.setattr(TruncatedSVD, "transform", observe_transform)
    train, test = TfidfFeaturizer(None, Lsa(2, 7)).fit_transform(
        VectorizerSpec(name="word", analyzer="word", ngram_range=(1, 1)),
        [note.text for note in notes],
        ["clinic review", "testonly"],
    )
    assert len(observed) == 1
    np.testing.assert_array_equal(observed[0].toarray(), train.sparse.toarray())
    assert len(transformed) == 1
    np.testing.assert_array_equal(transformed[0].toarray(), test.sparse.toarray())
    assert seeds == [(2, 7)]
    assert train.lsa is not None
    assert train.lsa.shape == (len(notes), 2)
    assert test.lsa is not None
    assert test.lsa.shape == (2, 2)
    assert test.sparse[1].nnz == 0
    monkeypatch.setattr(TruncatedSVD, "fit_transform", refuse_fit)
    monkeypatch.setattr(TruncatedSVD, "fit", refuse_fit)
    tuner = SklearnThresholdTuner()
    for _ in Diagnosis:
        tuner.fit(
            ClassifierFamily.LIGHTGBM_LSA, train, labels, subjects, 3, 7
        ).classify(test)


def test_features_without_lsa_reject_lightgbm_and_share_sparse() -> None:
    """Bez LSA se LightGBM odmítne, ostatní rodiny dostanou tutéž řídkou matici."""
    train, _ = TfidfFeaturizer(None).fit_transform(
        VectorizerSpec(name="word", analyzer="word", ngram_range=(1, 1)),
        ["alpha beta"],
        ["gamma"],
    )
    assert train.lsa is None
    for family in ClassifierFamily:
        if family is ClassifierFamily.LIGHTGBM_LSA:
            message = "LightGBM vyžaduje příznaky LSA."
            with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
                train.for_family(family)
        else:
            assert train.for_family(family) is train.sparse


def _nonconverging(family: ClassifierFamily, seed: Seed) -> BaseEstimator:
    del family, seed
    return LogisticRegression(max_iter=1)


@pytest.mark.parametrize(
    ("factory", "converged"), [(classifier_estimator, True), (_nonconverging, False)]
)
def test_convergence_includes_inner_fits_and_refit(
    factory: Callable[[ClassifierFamily, Seed], BaseEstimator], *, converged: bool
) -> None:
    """Nekonvergence je výsledek i pod filterwarnings=error, výchozí LR konverguje."""
    features, labels, subjects = _word_training()
    model = SklearnThresholdTuner(estimator=factory).fit(
        ClassifierFamily.LOGREG,
        features,
        labels,
        subjects,
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
    assert train.sparse.shape == (1, expected_features)
    assert test.sparse.shape == train.sparse.shape
    assert test.sparse.nnz == 0


def _observing_estimator(
    on_fit: Callable[[int], None],
) -> Callable[[ClassifierFamily, Seed], BaseEstimator]:
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

    def factory(family: ClassifierFamily, seed: Seed) -> BaseEstimator:
        del family
        return _ObservedLogreg(random_state=seed)

    return factory


def test_tuner_preserves_nonconvergence_unrelated_warning() -> None:
    """Tuner předá volajícímu UserWarning a nepovažuje ho za nekonvergenci."""
    features, labels, subjects = _word_training()

    def warn_on_fit(size: int) -> None:
        del size
        warnings.warn("Syntetické varování estimatoru.", UserWarning, stacklevel=2)

    with pytest.warns(UserWarning, match="Syntetické varování estimatoru"):
        model = SklearnThresholdTuner(estimator=_observing_estimator(warn_on_fit)).fit(
            ClassifierFamily.LOGREG,
            features,
            labels,
            subjects,
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
    features, labels, subjects = _word_training()
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

    classifier = tuner.fit(
        ClassifierFamily.LOGREG,
        ScreeningFeatures(features, None),
        labels,
        subjects,
        3,
        7,
    )

    lower, upper = 0.6, 0.9
    assert lower < classifier.threshold <= upper


@pytest.mark.parametrize("phase", ["inner", "refit"])
def test_tuner_detects_convergence_warning_in_each_fit_phase(phase: str) -> None:
    """Samostatné varování pouze vnitřních fitů i pouze refitu znamená nekonvergenci."""
    features, labels, subjects = _word_training()

    def warn_in_phase(size: int) -> None:
        if (size < len(labels)) == (phase == "inner"):
            warnings.warn(
                "Syntetická nekonvergence fáze fitu.", ConvergenceWarning, stacklevel=2
            )

    model = SklearnThresholdTuner(estimator=_observing_estimator(warn_in_phase)).fit(
        ClassifierFamily.LOGREG,
        features,
        labels,
        subjects,
        3,
        7,
    )

    assert model.converged is False


def test_featurizer_logs_only_name_and_manual_counts(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Log nese ruční počty zpráv, příznaků, nnz a pevný peak bez vstupních tokenů."""
    monkeypatch.setattr(
        "mediparse.infrastructure.sklearn_screening.peak_rss_gb",
        lambda: "peak RSS 1.50 GB",
    )
    with caplog.at_level("INFO", logger="mediparse.infrastructure.sklearn_screening"):
        TfidfFeaturizer(None).fit_transform(
            VectorizerSpec(name="word", analyzer="word", ngram_range=(1, 1)),
            ["quartz amber", "opal"],
            ["jasper"],
        )

    assert caplog.messages == [
        "Vektorizér word: 2 trénovacích a 1 testovacích zpráv, 3 příznaků, 3 nenulových hodnot, peak RSS 1.50 GB."
    ]
    assert all(
        token not in caplog.text for token in ("quartz", "amber", "opal", "jasper")
    )


def test_featurizer_applies_its_normalization() -> None:
    """Rozvinutí COPD vytvoří čtyři slovní příznaky místo jediného."""
    train, _ = TfidfFeaturizer(
        TextNormalization(
            acronyms={"COPD": "chronic obstructive pulmonary disease"}, variants={}
        )
    ).fit_transform(
        VectorizerSpec(name="word", analyzer="word", ngram_range=(1, 1)),
        ["COPD"],
        ["COPD"],
    )
    expected_features = 4
    assert train.sparse.shape[1] == expected_features


@pytest.mark.parametrize(
    ("analyzer", "ngram_range", "expected_features"),
    [
        ("char_wb", (3, 3), 3),
        ("word", (1, 1), 1),
    ],
)
def test_featurizer_uses_spec_analyzer(
    analyzer: Literal["word", "char_wb"],
    ngram_range: tuple[int, int],
    expected_features: int,
) -> None:
    """Abc má tři znakové trigramy s hranicemi slova, ale jediný slovní unigram."""
    train, _ = TfidfFeaturizer(None).fit_transform(
        VectorizerSpec(name="test", analyzer=analyzer, ngram_range=ngram_range),
        ["abc"],
        ["abc"],
    )
    assert train.sparse.shape[1] == expected_features


def test_tuner_logs_family_and_peak_rss(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dokončená buňka zaznamená jen rodinu a pevný celoživotní peak procesu."""
    features, labels, subjects = _word_training()
    monkeypatch.setattr(
        "mediparse.infrastructure.sklearn_screening.peak_rss_gb",
        lambda: "peak RSS 1.50 GB",
    )
    caplog.clear()

    with caplog.at_level("INFO", logger="mediparse.infrastructure.sklearn_screening"):
        SklearnThresholdTuner().fit(
            ClassifierFamily.LOGREG, features, labels, subjects, 3, 7
        )

    assert caplog.messages == ["Rodina logreg: peak RSS 1.50 GB."]


def test_tuner_passes_seed_to_estimator_factory() -> None:
    """Seed 13 předá továrně estimatoru, nezůstane jen u vnitřního dělení pacientů."""
    features, labels, subjects = _word_training()
    received: list[tuple[ClassifierFamily, Seed]] = []

    def recording(family: ClassifierFamily, seed: Seed) -> BaseEstimator:
        received.append((family, seed))
        return LogisticRegression(random_state=seed)

    SklearnThresholdTuner(estimator=recording).fit(
        ClassifierFamily.LOGREG, features, labels, subjects, 3, 13
    )

    assert received == [(ClassifierFamily.LOGREG, 13)]


@pytest.mark.parametrize(
    ("families", "expected"),
    [
        ((ClassifierFamily.LOGREG, ClassifierFamily.LIGHTGBM_LSA), Lsa(7, 13)),
        ((ClassifierFamily.COMPLEMENT_NB, ClassifierFamily.NB_SVM), None),
    ],
)
def test_screening_lsa_is_prepared_only_for_families_that_need_it(
    families: tuple[ClassifierFamily, ...],
    expected: Lsa | None,
) -> None:
    """LightGBM ve směsi rodin dostane komponenty a seed; ostatní rodiny LSA nepotřebují."""
    assert screening_lsa(families, 7, 13) == expected
