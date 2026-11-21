"""Adaptéry scikit-learn: trénink, klasifikace a uložení modelu."""

from __future__ import annotations

import functools
import re
from math import log
from typing import TYPE_CHECKING, Final
from unittest.mock import Mock

import pytest
import skops.io as sio
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import FixedThresholdClassifier, TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

from mediparse.domain.clinical_text import (
    NUMBER_PLACEHOLDER,
    TextNormalization,
    expand_diagnosis_acronyms,
    normalize_clinical_text,
    rewrite_diagnosis_variants,
)
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.section_headers import strip_diagnostic_sections
from mediparse.infrastructure import sklearn_classifier
from mediparse.infrastructure.diagnosis_candidates_file import load_text_normalization
from mediparse.infrastructure.sklearn_classifier import (
    TRUSTED_FUNCTIONS,
    LinearSvmTrainer,
    SkopsModelFile,
    tfidf_preprocessor,
    tfidf_vectorizer,
)
from mediparse.infrastructure.sklearn_split import threshold_splits
from tests.model_support import (
    ACRONYM_VOCABULARY,
    FIXED_SVM_PARAMS,
    NEGATIVE_CONTROL,
    SPELLING_VOCABULARY,
    VARIANT_VOCABULARY,
    separable_notes,
)
from tests.section_support import REPOSITORY_LEXICON, mimic_shaped_notes
from tests.support import REPOSITORY_DIAGNOSIS_CANDIDATES

if TYPE_CHECKING:
    from pathlib import Path

    from mediparse.domain.tuning import ParamValue

PATIENTS: Final = 12


def _corpus() -> tuple[list[str], list[bool], list[int]]:
    notes = separable_notes(PATIENTS)
    return (
        [note.text for note in notes],
        [Diagnosis.DIABETES in note.labels for note in notes],
        [note.subject_id for note in notes],
    )


def _noisy_corpus(flip_every: int = 5) -> tuple[list[str], list[bool], list[int]]:
    notes = separable_notes(24)
    return (
        [note.text for note in notes],
        [
            note.has_label(Diagnosis.DIABETES) != (note.subject_id % flip_every == 0)
            for note in notes
        ],
        [note.subject_id for note in notes],
    )


def _threshold_oracle(
    corpus: tuple[list[str], list[bool], list[int]], *, seed: int, scoring: str = "f1"
) -> TunedThresholdClassifierCV:
    texts, labels, subjects = corpus
    return TunedThresholdClassifierCV(
        Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    sublinear_tf=True, ngram_range=(1, 1), min_df=1, max_df=1.0
                ),
            ),
            ("svm", LinearSVC(C=1.0, class_weight=None, random_state=seed)),
        ]),
        scoring=scoring,
        cv=threshold_splits(labels, subjects, 3, seed),
    ).fit(texts, labels)


def test_classify_returns_label_at_threshold_with_decision_score() -> None:
    """Label odpovídá predikci modelu a prahu na škále rozhodovacího skóre."""
    texts, labels, subjects = _corpus()

    classifier = _trainer(None).fit(texts, labels, subjects)
    classifications = classifier.classify(texts)

    assert tuple(item.predicted for item in classifications) == tuple(labels)
    assert tuple(item.predicted for item in classifications) == tuple(
        bool(label) for label in classifier.model.predict(texts)
    )
    assert all(
        (item.score >= classifier.threshold) == item.predicted
        for item in classifications
    )


def test_smaller_regularization_parameter_reduces_score_range() -> None:
    """Menší C zesílí regularizaci a zmenší rozpětí rozhodovacích skóre."""
    texts, labels, subjects = _corpus()
    score_ranges = []

    for regularization in (0.001, 1000.0):
        classifier = LinearSvmTrainer(
            parameters=FIXED_SVM_PARAMS | {"c": regularization},
            seed=0,
            normalization=None,
            threshold_folds=3,
        ).fit(texts, labels, subjects)
        scores = [item.score for item in classifier.classify(texts)]
        score_ranges.append(max(scores) - min(scores))

    assert score_ranges[0] < score_ranges[1]


def test_sublinear_tf_weight_ratio_matches_definition() -> None:
    """Poměr vah odpovídá ručnímu výpočtu sublineárního TF a vyhlazeného IDF."""
    vectorizer = tfidf_vectorizer(None)

    weights = vectorizer.fit_transform([
        "alpha alpha alpha beta",
        "beta gamma",
    ]).toarray()

    alpha = weights[0, vectorizer.vocabulary_["alpha"]]
    beta = weights[0, vectorizer.vocabulary_["beta"]]
    assert alpha / beta == pytest.approx((1 + log(3)) * (log(3 / 2) + 1) / 1)


def test_model_file_round_trip(tmp_path: Path) -> None:
    """Uložený model se načte a klasifikuje stejně."""
    texts, labels, subjects = _corpus()
    classifier = _trainer(None).fit(texts, labels, subjects)
    model_file = SkopsModelFile(tmp_path / "out" / "model.skops")

    model_file.save(classifier)
    loaded = model_file.load()

    assert loaded.classify(texts) == classifier.classify(texts)


def test_model_file_creates_nested_parents(tmp_path: Path) -> None:
    """Uložení založí oba chybějící rodiče a model lze načíst."""
    texts, labels, subjects = _corpus()
    classifier = _trainer(None).fit(texts, labels, subjects)
    model_file = SkopsModelFile(tmp_path / "a" / "b" / "model.skops")

    model_file.save(classifier)

    assert model_file.load().classify(texts) == classifier.classify(texts)


def test_repeated_training_keeps_predictions_after_saving(tmp_path: Path) -> None:
    """Dva stejné tréninky i po uložení dávají přesně stejné labely a skóre."""
    texts, labels, subjects = _noisy_corpus()
    first = _trainer(None).fit(texts, labels, subjects)
    second = _trainer(None).fit(texts, labels, subjects)
    first_file = SkopsModelFile(tmp_path / "first.skops")
    second_file = SkopsModelFile(tmp_path / "second.skops")
    first_file.save(first)
    second_file.save(second)
    evaluation = [*texts, "diabetes treatment", "a new synthetic note"]

    first_predictions = first_file.load().classify(evaluation)
    second_predictions = second_file.load().classify(evaluation)

    assert [item.predicted for item in first_predictions] == [
        item.predicted for item in second_predictions
    ]
    assert [item.score for item in first_predictions] == [
        item.score for item in second_predictions
    ]


def test_model_file_missing_is_invalid_input(tmp_path: Path) -> None:
    """Chybějící soubor modelu je neplatný vstup a jmenuje cestu."""
    path = tmp_path / "model.skops"

    with pytest.raises(InvalidInputError, match=r"model\.skops"):
        SkopsModelFile(path).load()


class _Unsafe:
    pass


def test_model_file_corrupt_is_invalid_input(tmp_path: Path) -> None:
    """Poškozený soubor modelu je neplatný vstup a jmenuje cestu."""
    path = tmp_path / "bad.skops"
    path.write_text("garbage", encoding="utf-8")

    with pytest.raises(InvalidInputError, match=r"bad\.skops"):
        SkopsModelFile(path).load()


def test_model_file_untrusted_type_is_invalid_input(tmp_path: Path) -> None:
    """Typ mimo seznam důvěryhodných je neplatný vstup a jmenuje cestu."""
    path = tmp_path / "unsafe.skops"
    sio.dump(_Unsafe(), path)

    with pytest.raises(InvalidInputError, match=r"unsafe\.skops"):
        SkopsModelFile(path).load()


_NORMALIZATION: Final = TextNormalization(
    variants={
        "T2DM": "diabetes",
        "NIDDM": "diabetes",
        "IDDM": "diabetes",
        "PNA": "pneumonia",
        "afib": "atrial fibrillation",
        "a-fib": "atrial fibrillation",
        "a fib": "atrial fibrillation",
        "kindey": "kidney",
    },
    acronyms={
        "COPD": "chronic obstructive pulmonary disease",
        "AF": "atrial fibrillation",
    },
)


def _trainer(normalization: TextNormalization | None) -> LinearSvmTrainer:
    return LinearSvmTrainer(
        FIXED_SVM_PARAMS, seed=0, normalization=normalization, threshold_folds=3
    )


def _paired_corpus(
    vocabulary: tuple[str, str],
) -> tuple[list[str], list[bool], list[int]]:
    return (
        [vocabulary[0]] * 4 + [vocabulary[1]] * 4 + [NEGATIVE_CONTROL] * 8,
        [True] * 4 + [False] * 12,
        list(range(16)),
    )


def test_preprocessor_lowercases_after_normalization() -> None:
    """Vlastní preprocessor vypíná lowercase vectorizeru, snížení písmen nese funkce."""
    text = "CAST 500mg ___yo"

    assert tfidf_preprocessor(text, {}, {}) == normalize_clinical_text(text).lower()
    assert "cast" in tfidf_preprocessor(text, {}, {})


def test_preprocessor_expands_acronym_next_to_marker_and_lowercases() -> None:
    """Zkratka za značkou se rozvine před snížením písmen."""
    result = tfidf_preprocessor(
        "___COPD", _NORMALIZATION.acronyms, _NORMALIZATION.variants
    )

    assert "chronic obstructive pulmonary disease" in result
    assert "copd" not in result


def test_trainer_with_acronyms_scores_abbreviation_and_full_name_alike() -> None:
    """Zkratka a vypsaný název dostanou se slovníkem stejné skóre, bez něj různé."""
    texts, labels, subjects = _paired_corpus(ACRONYM_VOCABULARY)

    expanded = (
        _trainer(_NORMALIZATION)
        .fit(texts, labels, subjects)
        .classify(ACRONYM_VOCABULARY)
    )
    raw = _trainer(None).fit(texts, labels, subjects).classify(ACRONYM_VOCABULARY)

    assert expanded[0].score == expanded[1].score
    assert raw[0].score != raw[1].score


def test_model_file_round_trip_keeps_acronyms(tmp_path: Path) -> None:
    """Načtený model skóruje shodně a jeho preprocessor nese slovník jako data."""
    texts, labels, subjects = _paired_corpus(ACRONYM_VOCABULARY)
    classifier = _trainer(_NORMALIZATION).fit(texts, labels, subjects)
    model_file = SkopsModelFile(tmp_path / "model.skops")

    model_file.save(classifier)
    loaded = model_file.load()

    assert loaded.classify(ACRONYM_VOCABULARY) == classifier.classify(
        ACRONYM_VOCABULARY
    )
    preprocessor = loaded.model.estimator_["tfidf"].preprocessor
    assert preprocessor.keywords == {
        "acronyms": dict(_NORMALIZATION.acronyms),
        "variants": dict(_NORMALIZATION.variants),
    }
    assert type(preprocessor.keywords["variants"]) is dict
    assert type(preprocessor.keywords["acronyms"]) is dict


def test_model_file_partial_around_foreign_function_is_invalid_input(
    tmp_path: Path,
) -> None:
    """Důvěra v ``functools.partial`` nepustí cizí funkci uvnitř."""
    texts, labels, _ = _corpus()
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                preprocessor=functools.partial(_foreign_preprocessor, acronyms={})
            ),
        ),
        ("svm", LinearSVC()),
    ]).fit(texts, labels)
    path = tmp_path / "foreign_partial.skops"
    sio.dump(pipeline, path)

    with pytest.raises(
        InvalidInputError, match=r"foreign_partial\.skops.*_foreign_preprocessor"
    ):
        SkopsModelFile(path).load()


def test_trainer_with_normalization_scores_spellings_alike() -> None:
    """Se zapnutou normalizací dostanou dva zápisy téže dávky a věku stejné skóre."""
    texts, labels, subjects = _paired_corpus(SPELLING_VOCABULARY)

    normalized = (
        _trainer(_NORMALIZATION)
        .fit(texts, labels, subjects)
        .classify(SPELLING_VOCABULARY)
    )
    raw = (
        _trainer(normalization=None)
        .fit(texts, labels, subjects)
        .classify(SPELLING_VOCABULARY)
    )

    assert normalized[0].score == normalized[1].score
    assert raw[0].score != raw[1].score


def test_model_file_round_trip_with_preprocessor(tmp_path: Path) -> None:
    """Model s preprocessorem se načte a skóre se shodují."""
    texts, labels, subjects = _paired_corpus(SPELLING_VOCABULARY)
    classifier = _trainer(_NORMALIZATION).fit(texts, labels, subjects)
    model_file = SkopsModelFile(tmp_path / "model.skops")

    model_file.save(classifier)

    assert model_file.load().classify(SPELLING_VOCABULARY) == classifier.classify(
        SPELLING_VOCABULARY
    )


def _foreign_preprocessor(text: str, acronyms: object = None) -> str:
    del acronyms
    return text.lower()


def test_model_file_foreign_preprocessor_is_invalid_input(tmp_path: Path) -> None:
    """Funkce mimo seznam důvěryhodných jmen se nenačte."""
    texts, labels, _ = _corpus()
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=_foreign_preprocessor)),
        ("svm", LinearSVC()),
    ]).fit(texts, labels)
    path = tmp_path / "foreign.skops"
    sio.dump(pipeline, path)

    with pytest.raises(
        InvalidInputError, match=r"foreign\.skops.*_foreign_preprocessor"
    ):
        SkopsModelFile(path).load()


def test_trusted_functions_are_exact_full_names() -> None:
    """Seznam nese přesné plné jméno funkce, přejmenování spadne tady, ne při načtení."""
    assert (
        "functools.partial",
        f"{tfidf_preprocessor.__module__}.{tfidf_preprocessor.__qualname__}",
    ) == TRUSTED_FUNCTIONS


@pytest.mark.parametrize("name", sorted(mimic_shaped_notes()))
def test_analyzer_sees_normalized_stripped_fixture(name: str) -> None:
    """Analyzátor modelu vidí text po stripu a normalizaci tokeny výchozího vzoru."""
    text = mimic_shaped_notes()[name]
    texts, labels, subjects = _paired_corpus(SPELLING_VOCABULARY)
    analyzer = (
        _trainer(_NORMALIZATION)
        .fit(texts, labels, subjects)
        .model.estimator_["tfidf"]
        .build_analyzer()
    )
    stripped = strip_diagnostic_sections(text, REPOSITORY_LEXICON)

    expected = TfidfVectorizer().build_analyzer()(
        rewrite_diagnosis_variants(
            expand_diagnosis_acronyms(
                normalize_clinical_text(stripped), _NORMALIZATION.acronyms
            ),
            _NORMALIZATION.variants,
        ).lower()
    )

    tokens = analyzer(stripped)
    assert tokens == expected
    assert not any("___" in token for token in tokens)
    assert not any(token.isdigit() for token in tokens)
    if re.search(r"(?<![A-Za-z0-9_])[0-9]", stripped):
        assert NUMBER_PLACEHOLDER in tokens


def test_analyzer_tokens_after_strip_normalization_and_expansion() -> None:
    """Doslovný seznam tokenů po stripu, normalizaci a rozvinutí, bez produkčních funkcí."""
    texts, labels, subjects = _paired_corpus(SPELLING_VOCABULARY)
    analyzer = (
        _trainer(_NORMALIZATION).fit(texts, labels, subjects).model.estimator_["tfidf"]
    )
    text = "Discharge Diagnosis:\nCOPD\n\nHistory of Present Illness:\nPt ___yo with COPD and AF, 500 mg."

    stripped = strip_diagnostic_sections(text, REPOSITORY_LEXICON)

    assert analyzer.build_analyzer()(stripped) == [
        "history",
        "of",
        "present",
        "illness",
        "pt",
        "__age__",
        "with",
        "chronic",
        "obstructive",
        "pulmonary",
        "disease",
        "and",
        "atrial",
        "fibrillation",
        "__num__",
        "mg",
    ]


def test_vectorizer_without_normalization_has_no_preprocessor_and_lowercases() -> None:
    """Ablace nonorm nemá preprocessor, písmena snižuje sám vectorizer."""
    vectorizer = tfidf_vectorizer(None)

    assert vectorizer.preprocessor is None
    assert vectorizer.build_analyzer()("CAST 5mg") == ["cast", "5mg"]


def test_vectorizer_with_normalization_carries_preprocessor_with_dictionary() -> None:
    """Preprocessor je partial s oběma slovníky jako holými dicty."""
    vectorizer = tfidf_vectorizer(_NORMALIZATION)

    assert isinstance(vectorizer.preprocessor, functools.partial)
    assert vectorizer.preprocessor.func is tfidf_preprocessor
    assert vectorizer.preprocessor.keywords == {
        "acronyms": dict(_NORMALIZATION.acronyms),
        "variants": dict(_NORMALIZATION.variants),
    }


def test_trainer_builds_vectorizer_through_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Trenér bere vectorizer z továrny: odlišný token_pattern se projeví v modelu."""
    monkeypatch.setattr(
        "mediparse.infrastructure.sklearn_classifier.tfidf_vectorizer",
        lambda _normalization, **_params: TfidfVectorizer(
            token_pattern=r"[a-z]{3,}"  # ruff: ignore[hardcoded-password-func-arg]
        ),
    )
    texts, labels, subjects = _paired_corpus(SPELLING_VOCABULARY)

    model = _trainer(None).fit(texts, labels, subjects)

    assert model.model.estimator_["tfidf"].build_analyzer()("ab abc 12 abcd") == [
        "abc",
        "abcd",
    ]


def test_vectorizer_variants_differ_only_in_preprocessor() -> None:
    """Obě varianty z jednoho konstruktoru: liší se jen klíčem preprocessor."""
    plain = tfidf_vectorizer(None).get_params()
    prepared = tfidf_vectorizer(_NORMALIZATION).get_params()

    assert plain.pop("preprocessor") is None
    assert prepared.pop("preprocessor") is not None
    assert plain == prepared


def test_analyzer_rewrites_variant_after_deid_normalization() -> None:
    """Podtržítka se odstraní dřív, než hranice varianty hledá T2DM."""
    analyzer = tfidf_vectorizer(_NORMALIZATION).build_analyzer()
    assert analyzer("___T2DM") == ["__deid__", "diabetes"]


def test_preprocessor_expands_acronyms_and_variants_before_lowercase() -> None:
    """Zkratka i varianta dají stejný název diagnózy v ručně sestavené větě."""
    assert (
        tfidf_preprocessor(
            "h/o AF, afib", _NORMALIZATION.acronyms, _NORMALIZATION.variants
        )
        == "h/o atrial fibrillation, atrial fibrillation"
    )


def test_variant_model_is_self_contained_after_round_trip(tmp_path: Path) -> None:
    """Přepis sjednotí skóre i po načtení skops bez přístupu ke configu."""
    texts, labels, subjects = _paired_corpus(VARIANT_VOCABULARY)
    with_variants = _trainer(_NORMALIZATION).fit(texts, labels, subjects)
    without_variants = _trainer(
        TextNormalization(acronyms=_NORMALIZATION.acronyms, variants={})
    ).fit(texts, labels, subjects)
    model_file = SkopsModelFile(tmp_path / "variant.skops")
    model_file.save(with_variants)
    loaded = model_file.load()
    scores = loaded.classify(VARIANT_VOCABULARY)
    assert scores[0].score == scores[1].score
    assert loaded.classify(VARIANT_VOCABULARY) == with_variants.classify(
        VARIANT_VOCABULARY
    )
    raw_scores = without_variants.classify(VARIANT_VOCABULARY)
    assert raw_scores[0].score != raw_scores[1].score


def test_preprocessor_repairs_misspelling_inside_multiword_variant() -> None:
    """Oprava překlepu odhalí víceslovnou variantu už v témže předzpracování."""
    normalization = load_text_normalization(
        REPOSITORY_DIAGNOSIS_CANDIDATES, variants=True
    )
    assert (
        tfidf_preprocessor(
            "chronic obstuctive lung disease",
            normalization.acronyms,
            normalization.variants,
        )
        == "chronic obstructive pulmonary disease"
    )


def test_threshold_matches_library_cross_validation() -> None:
    """Zašuměný korpus rozliší seedy a práh odpovídá nezávisle složenému TTCV."""
    corpus = _noisy_corpus()
    texts, labels, subjects = corpus
    oracle = _threshold_oracle(corpus, seed=0)
    other_seed = _threshold_oracle(corpus, seed=1)
    assert oracle.best_threshold_ != other_seed.best_threshold_

    classifier = _trainer(None).fit(texts, labels, subjects)

    assert classifier.threshold == oracle.best_threshold_
    assert [item.score for item in classifier.classify(texts)] == pytest.approx(
        oracle.estimator_.decision_function(texts)
    )


def test_threshold_optimizes_f1_instead_of_balanced_accuracy() -> None:
    """Při převrácení každého třetího pacienta zvolí F1 jiný práh než výchozí metrika."""
    corpus = _noisy_corpus(flip_every=3)
    f1 = _threshold_oracle(corpus, seed=0)
    balanced = _threshold_oracle(corpus, seed=0, scoring="balanced_accuracy")
    assert f1.best_threshold_ != balanced.best_threshold_

    classifier = _trainer(None).fit(*corpus)

    assert classifier.threshold == f1.best_threshold_


def test_trainer_passes_seed_to_svm_and_threshold_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seed běhu dostane SVM i dělení pacientů, i když řešič seed nepotřebuje."""
    seed = 13
    estimator = Mock(wraps=LinearSVC)
    splitter = Mock(wraps=threshold_splits)
    monkeypatch.setattr(sklearn_classifier, "LinearSVC", estimator)
    monkeypatch.setattr(sklearn_classifier, "threshold_splits", splitter)

    LinearSvmTrainer(
        FIXED_SVM_PARAMS, seed=seed, normalization=None, threshold_folds=3
    ).fit(*_noisy_corpus())

    assert estimator.call_args.kwargs["random_state"] == seed
    assert splitter.call_args.args[-1] == seed


@pytest.mark.parametrize(
    "parameters",
    [
        FIXED_SVM_PARAMS | {"c": 0.0},
        FIXED_SVM_PARAMS | {"sublinear_tf": "true"},
        FIXED_SVM_PARAMS | {"unknown": 1},
        {},
    ],
)
def test_invalid_svm_parameters_are_invalid_input(
    parameters: dict[str, ParamValue],
) -> None:
    """Typové meze i neznámé nebo chybějící parametry trenér odmítne doménovou chybou."""
    texts, labels, subjects = _corpus()
    with pytest.raises(InvalidInputError, match="SVM"):
        LinearSvmTrainer(parameters, 0, None, 3).fit(texts, labels, subjects)


def test_nonconvergence_is_reported_after_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Varování ze skutečného fitu určuje konvergenci výsledku tréninku."""
    original = sklearn_classifier.svm_estimator
    monkeypatch.setattr(
        sklearn_classifier,
        "svm_estimator",
        lambda params, seed: original(params, seed).set_params(max_iter=1),
    )
    texts, labels, subjects = _corpus()
    classifier = _trainer(None).fit(texts, labels, subjects)
    assert classifier.converged is False


def test_model_file_loads_fixed_threshold_archive_without_training_metadata(
    tmp_path: Path,
) -> None:
    """Archiv knihovny bez metadat tréninku lze načíst a reprodukovat jeho predikce."""
    texts, labels, _ = _corpus()
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(sublinear_tf=True)),
        ("svm", LinearSVC(random_state=0)),
    ]).fit(texts, labels)
    model = FixedThresholdClassifier(
        FrozenEstimator(pipeline), threshold=0.25, response_method="decision_function"
    ).fit(texts, labels)
    path = tmp_path / "model.skops"
    sio.dump(model, path)

    predictions = SkopsModelFile(path).load().classify(texts)

    assert [item.predicted for item in predictions] == list(model.predict(texts))
    assert [item.score for item in predictions] == pytest.approx(
        model.decision_function(texts)
    )


@pytest.mark.parametrize(
    "shape",
    [
        "bare",
        "unfrozen",
        "step_name",
        "vectorizer",
        "foreign",
        "extra_step",
        "not_estimator",
        "unfitted",
        "nested",
    ],
)
def test_model_file_of_other_shape_is_invalid_input(tmp_path: Path, shape: str) -> None:
    """Cizí i zastaralý tvar archivu končí doménovou chybou při načtení."""
    texts, labels, _ = _corpus()
    steps = [
        (
            "vec" if shape == "step_name" else "tfidf",
            CountVectorizer() if shape == "vectorizer" else TfidfVectorizer(),
        ),
        (
            "svm",
            LogisticRegression() if shape == "foreign" else LinearSVC(random_state=0),
        ),
    ]
    if shape == "extra_step":
        steps.insert(1, ("scale", MaxAbsScaler()))
    pipeline = Pipeline(steps)
    if shape != "unfitted":
        pipeline.fit(texts, labels)
    estimator = pipeline if shape == "unfrozen" else FrozenEstimator(pipeline)
    if shape == "nested":
        estimator = FixedThresholdClassifier(
            pipeline, threshold=0.0, response_method="decision_function"
        )
    model = (
        pipeline
        if shape == "bare"
        else FixedThresholdClassifier(
            estimator,
            threshold=0.0,
            response_method="decision_function",
        )
    )
    if shape not in {"bare", "unfitted"}:
        model.fit(texts, labels)
    path = tmp_path / f"{shape}.skops"
    sio.dump({"kind": "constructed"} if shape == "not_estimator" else model, path)

    with pytest.raises(InvalidInputError, match=rf"{shape}\.skops.*nemá tvar"):
        SkopsModelFile(path).load()
