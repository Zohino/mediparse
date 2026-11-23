"""Skutečné fity a překlad chyb klasického stupně."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from random import Random
from typing import TYPE_CHECKING, Literal
from unittest.mock import Mock, call

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from joblib import parallel_config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from mediparse.application.patient_cv import patient_holdouts
from mediparse.domain.clinical_text import TextNormalization
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.representation import Word2VecRepresentation
from mediparse.domain.split import Holdout
from mediparse.domain.stage_tuning import StageLsa
from mediparse.entrypoints.stage_tuning import stage_normalizations
from mediparse.infrastructure import sklearn_classifier, sklearn_stage
from mediparse.infrastructure.diagnosis_candidates_file import load_text_normalization
from mediparse.infrastructure.fasttext_classifier import FastTextClassifier
from mediparse.infrastructure.sklearn_classifier import (
    EMPTY_VOCABULARY,
    TfidfParameters,
    tfidf_vectorizer,
)
from mediparse.infrastructure.sklearn_split import StratifiedGroupFolds
from mediparse.infrastructure.sklearn_stage import (
    SklearnStageScorer,
    encoder_features,
    joined_tokens,
    ngram_counts,
    tfidf_features,
    token_documents,
    word2vec_features,
)
from mediparse.infrastructure.stage_config import load_stage_config
from tests.hf_encoder_support import SPEC
from tests.screening_support import notes
from tests.stage_support import stage_config, token_stage_config, word2vec_stage_config
from tests.support import REPOSITORY, REPOSITORY_DIAGNOSIS_CANDIDATES

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from scipy.sparse import csr_matrix
    from sklearn.base import BaseEstimator
    from sklearn.decomposition import TruncatedSVD

    from mediparse.domain.representation import ClassifierFamily
    from mediparse.domain.stage_tuning import StageTuningConfig
    from mediparse.domain.tuning import ParamValue


def test_logreg_parameters_change_scores(tmp_path: Path) -> None:
    """Silnější a slabší regularizace dávají různá skóre na stejné syntetice."""
    config = stage_config()
    scorer = SklearnStageScorer(
        config, dict.fromkeys(config.text_preprocessing), 7, tmp_path, Mock()
    )
    fold = patient_holdouts(notes(visits=10), 2, 7, StratifiedGroupFolds())[0]
    params = config.branches[0].default_params()
    low = scorer.score(params | {"logreg.C": 1e-4}, fold)
    high = scorer.score(params | {"logreg.C": 1e2}, fold)
    assert low.scores[Diagnosis.CKD] != high.scores[Diagnosis.CKD]


def _lsa_config(**changes: object) -> StageTuningConfig:
    config = stage_config()
    data = config.model_dump(mode="json")
    data["families"] = [
        {
            "family": "lightgbm_lsa",
            "space": {
                "params": [
                    {"kind": "categorical", "name": "n_estimators", "choices": [10]}
                ]
            },
            "default": {"n_estimators": 10},
        },
        data["families"][0],
    ]
    data.update({"lsa": {"components": 2, "max_features": 1000}} | changes)
    return type(config).model_validate_json(json.dumps(data))


def test_lightgbm_receives_lsa_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Estimator LightGBM vidí právě tři komponenty místo řídkého TF-IDF."""
    config = _lsa_config(lsa={"components": 3, "max_features": 1000})
    observed = []
    original = sklearn_stage.classifier_estimator

    def estimator(family: ClassifierFamily, seed: int, scratch: Path) -> BaseEstimator:
        model = original(family, seed, scratch).set_params()
        fit = model.fit

        def recording_fit(features: np.ndarray, labels: list[bool]) -> BaseEstimator:
            observed.append(features.shape[1])
            return fit(features, labels)

        monkeypatch.setattr(model, "fit", recording_fit)
        return model

    monkeypatch.setattr(sklearn_stage, "classifier_estimator", estimator)
    scorer = SklearnStageScorer(
        config, dict.fromkeys(config.text_preprocessing), 7, tmp_path, Mock()
    )
    fold = patient_holdouts(notes(visits=10), 2, 7, StratifiedGroupFolds())[0]
    scorer.score(config.branches[0].default_params(), fold)
    assert observed == [3] * 5


@pytest.mark.parametrize("lsa", [True, False])
def test_vocabulary_cap_applies_only_to_lsa(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, lsa: bool
) -> None:
    """Strop omezí vstup LSA na pět sloupců, rodina bez LSA vidí celé V."""
    config = _lsa_config(lsa={"components": 3, "max_features": 5})
    branch = next(
        b
        for b in config.branches
        if b.family.family.value == ("lightgbm_lsa" if lsa else "logreg")
    )
    fold = patient_holdouts(notes(visits=10), 2, 7, StratifiedGroupFolds())[0]
    vocabulary = (
        tfidf_vectorizer(None, ngram_range=(1, 2))
        .fit([n.text for n in fold.train])
        .vocabulary_
    )
    assert len(vocabulary) > 5
    observed = []
    original = sklearn_stage.TruncatedSVD

    def svd(**kwargs: int) -> TruncatedSVD:
        model = original(**kwargs)
        fit_transform = model.fit_transform

        def recording_fit(features: csr_matrix) -> np.ndarray:
            observed.append(features.shape[1])
            return fit_transform(features)

        monkeypatch.setattr(model, "fit_transform", recording_fit)
        return model

    monkeypatch.setattr(sklearn_stage, "TruncatedSVD", svd)
    result = SklearnStageScorer(
        config, dict.fromkeys(config.text_preprocessing), 7, tmp_path, Mock()
    ).score(branch.default_params(), fold)
    assert observed == ([5] if lsa else [])
    assert result.n_features == (5 if lsa else len(vocabulary))


def _clinical_fold() -> Holdout:
    corpus = tuple(
        replace(
            n,
            text=f"{n.text} {'___COPD COPD' if n.subject_id % 2 else 'renal CKD'} "
            + "repeated " * (i % 4 + 1),
        )
        for i, n in enumerate(notes(visits=10))
    )
    return patient_holdouts(corpus, 2, 7, StratifiedGroupFolds())[0]


def test_nondefault_char_tfidf_matches_manual_pipeline(tmp_path: Path) -> None:
    """Nevýchozí znakové TF-IDF s normalizací dává ručně složená skóre."""
    config = load_stage_config(REPOSITORY / "config/stage_classical.json").model_copy(
        update={"lsa": StageLsa(components=3, max_features=100_000)}
    )
    branch = next(b for b in config.branches if b.name == "char_wb/logreg")
    normalization = load_text_normalization(
        REPOSITORY_DIAGNOSIS_CANDIDATES, variants=True
    )
    fold = _clinical_fold()
    params = branch.default_params() | {
        "char_wb.ngram_min": 2,
        "char_wb.ngram_max": 4,
        "char_wb.min_df": 2,
        "char_wb.max_df": 0.9,
        "char_wb.sublinear_tf": False,
        "logreg.C": 0.25,
    }
    scored = SklearnStageScorer(
        config,
        dict.fromkeys(config.text_preprocessing, normalization),
        7,
        tmp_path,
        Mock(),
    ).score(params, fold)
    vectorizer = tfidf_vectorizer(
        normalization,
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=2,
        max_df=0.9,
        sublinear_tf=False,
    )
    train = vectorizer.fit_transform([n.text for n in fold.train])
    validation = vectorizer.transform([n.text for n in fold.test])
    for diagnosis in Diagnosis:
        model = LogisticRegression(max_iter=1000, C=0.25).fit(
            train, [n.has_label(diagnosis) for n in fold.train]
        )
        assert scored.scores[diagnosis] == pytest.approx(
            model.predict_proba(validation)[:, 1], abs=1e-12
        )


@pytest.mark.parametrize("family", ["lightgbm_lsa", "linear_svm", "nb_svm"])
def test_seeded_scores_repeat(tmp_path: Path, family: str) -> None:
    """Stejný seed opakuje skóre; různé seedy rozliší alespoň větev s LSA."""
    config = load_stage_config(REPOSITORY / "config/stage_classical.json").model_copy(
        update={"lsa": StageLsa(components=3, max_features=100_000)}
    )
    branch = next(b for b in config.branches if b.name == f"char_wb/{family}")
    random = Random(117)
    vocabulary = [f"synthetic{i}" for i in range(100)]
    corpus = tuple(
        replace(n, text=" ".join(random.choices(vocabulary, k=40)))
        for n in notes(visits=10)
    )
    fold = patient_holdouts(corpus, 2, 7, StratifiedGroupFolds())[0]
    first = SklearnStageScorer(
        config, dict.fromkeys(config.text_preprocessing), 7, tmp_path, Mock()
    ).score(branch.default_params(), fold)
    repeated = SklearnStageScorer(
        config, dict.fromkeys(config.text_preprocessing), 7, tmp_path, Mock()
    ).score(branch.default_params(), fold)
    other = SklearnStageScorer(
        config, dict.fromkeys(config.text_preprocessing), 29, tmp_path, Mock()
    ).score(branch.default_params(), fold)
    assert first.scores == repeated.scores
    if family == "lightgbm_lsa":
        assert first.scores != other.scores


@pytest.mark.parametrize("failed_call", [2, None])
def test_convergence_accumulates_all_diagnoses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_call: int | None
) -> None:
    """Jediná nekonvergující diagnóza zneplatní konvergenci celého foldu."""
    original = sklearn_stage.fit_converged
    calls = 0

    def fit_once(fit: Callable[[], object]) -> bool:
        nonlocal calls
        calls += 1
        return original(fit) and calls != failed_call

    monkeypatch.setattr(sklearn_stage, "fit_converged", fit_once)
    config = stage_config()
    scorer = SklearnStageScorer(
        config, dict.fromkeys(config.text_preprocessing), 7, tmp_path, Mock()
    )
    scored = scorer.score(config.branches[0].default_params(), _clinical_fold())
    assert scored.converged is (failed_call is None)
    assert calls == 5


def test_min_df_above_training_documents_is_refused(tmp_path: Path) -> None:
    """Sklearn odmítne min_df větší než počet trénovacích dokumentů."""
    config = stage_config()
    scorer = SklearnStageScorer(
        config, dict.fromkeys(config.text_preprocessing), 7, tmp_path, Mock()
    )
    params = config.branches[0].default_params() | {"r1.min_df": 100000}
    with pytest.raises(InvalidInputError, match="min_df"):
        scorer.score(params, _clinical_fold())


@pytest.mark.parametrize(
    ("family", "param", "value"),
    [("logreg", "C", 1.0), ("linear_svm", "C", 1.0), ("lightgbm", "n_estimators", 5)],
)
def test_encoder_branch_scores_cached_embeddings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    param: str,
    value: float,
) -> None:
    """Encoder poskytne pět vektorů skóre a šířku čtyř příznaků."""
    config = stage_config(
        representations=[SPEC.model_dump(mode="json")],
        families=[
            {
                "family": family,
                "space": {
                    "params": [
                        {"name": param, "kind": "categorical", "choices": [value]}
                    ]
                },
                "default": {param: value},
            }
        ],
    )
    observed = []
    original = sklearn_stage.classifier_estimator

    def estimator(family: ClassifierFamily, seed: int, scratch: Path) -> BaseEstimator:
        model = original(family, seed, scratch).set_params()
        fit = model.fit

        def recording_fit(features: np.ndarray, labels: list[bool]) -> BaseEstimator:
            observed.append(features.shape[1])
            return fit(features, labels)

        monkeypatch.setattr(model, "fit", recording_fit)
        return model

    monkeypatch.setattr(sklearn_stage, "classifier_estimator", estimator)
    fold = _clinical_fold()
    vectors = {
        n.text: [float(i % 3), float(i % 5), float(i % 7), float(i % 11)]
        for i, n in enumerate((*fold.train, *fold.test))
    }
    embeddings = Mock()
    embeddings.embed.side_effect = lambda _spec, texts: np.array([
        vectors[text] for text in texts
    ])
    scored = SklearnStageScorer(
        config, dict.fromkeys(config.text_preprocessing), 7, tmp_path, embeddings
    ).score(config.branches[0].default_params(), fold)
    assert len(scored.scores) == 5
    assert all(len(scores) == len(fold.test) for scores in scored.scores.values())
    assert scored.n_features == 4
    assert observed == [4] * 5
    assert embeddings.embed.call_args_list == [
        call(SPEC, [n.text for n in (*fold.train, *fold.test)])
    ]


@pytest.mark.parametrize(
    ("text", "normalization", "features"),
    [
        ("Alpha, beta alpha", None, 3),
        (
            "COPD beta",
            TextNormalization(
                acronyms={"COPD": "chronic obstructive pulmonary disease"}, variants={}
            ),
            6,
        ),
    ],
)
def test_tokens_branch_counts_distinct_analyzer_tokens(
    tmp_path: Path, text: str, normalization: TextNormalization | None, features: int
) -> None:
    """Analyzátor počítá jedinečné tokeny včetně rozvinuté zkratky."""
    original = notes(visits=10)
    fold = Holdout(
        0,
        tuple(
            replace(n, text=text if i % 2 else "beta gamma")
            for i, n in enumerate(original[:-1])
        ),
        (replace(original[-1], text="alpha"),),
    )
    config = token_stage_config()
    scored = SklearnStageScorer(
        config,
        dict.fromkeys(config.text_preprocessing, normalization),
        7,
        tmp_path,
        Mock(),
    ).score(config.branches[0].default_params(), fold)
    assert scored.n_features == features


def test_fasttext_family_params_reach_estimator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Každý fit dostane hyperparametry aktivního pokusu."""
    original = sklearn_stage.classifier_estimator
    observed = []

    def estimator(family: ClassifierFamily, seed: int, scratch: Path) -> BaseEstimator:
        model = original(family, seed, scratch)
        assert isinstance(model, FastTextClassifier)
        fit = model.fit

        def recording_fit(documents: np.ndarray, labels: list[bool]) -> BaseEstimator:
            observed.append({
                name: model.get_params()[name]
                for name in ("lr", "epoch", "word_ngrams", "dim")
            })
            return fit(documents, labels)

        monkeypatch.setattr(model, "fit", recording_fit)
        return model

    monkeypatch.setattr(sklearn_stage, "classifier_estimator", estimator)
    config = token_stage_config()
    SklearnStageScorer(
        config, dict.fromkeys(config.text_preprocessing), 7, tmp_path, Mock()
    ).score(config.branches[0].default_params(), _clinical_fold())
    assert observed == [{"lr": 0.5, "epoch": 1, "word_ngrams": 2, "dim": 2}] * 5


def test_encoder_scaler_fits_only_outer_train() -> None:
    """Průměr 1 a populační std 1 z tréninku převedou test 4 na 3."""
    embeddings = Mock()
    embeddings.embed.side_effect = lambda _spec, texts: np.array([
        [{"a": 0.0, "b": 2.0, "c": 4.0}[text]] for text in texts
    ])
    train, test = encoder_features(embeddings, SPEC, ["a", "b"], ["c"])
    assert embeddings.embed.call_args_list == [call(SPEC, ["a", "b", "c"])]
    np.testing.assert_array_equal(train, [[-1.0], [1.0]])
    np.testing.assert_array_equal(test, [[3.0]])


def test_joined_tokens_drops_punctuation_and_single_letters() -> None:
    """Slovní analyzátor odstraní interpunkci a jednopísmenná slova."""
    prepare = joined_tokens(None)
    assert prepare("Insulin, glucose!\nA b") == "insulin glucose"
    assert prepare("Fracture.") == "fracture"


def test_joined_tokens_applies_normalization() -> None:
    """Rozvinutí COPD dodá stejná slova jako TF-IDF."""
    prepare = joined_tokens(
        TextNormalization(
            acronyms={"COPD": "chronic obstructive pulmonary disease"}, variants={}
        )
    )
    assert prepare("COPD") == "chronic obstructive pulmonary disease"


def test_token_documents_preserves_prepared_texts() -> None:
    """Hotové tokeny zůstanou řetězci v polích s dtype object."""
    train, test = token_documents(["insulin glucose"], ["fracture"])
    assert train.dtype == test.dtype == object
    np.testing.assert_array_equal(train, ["insulin glucose"])
    np.testing.assert_array_equal(test, ["fracture"])


@pytest.mark.parametrize(
    "config", [stage_config(), token_stage_config(), word2vec_stage_config()]
)
def test_empty_vocabulary_is_invalid_input(
    tmp_path: Path, config: StageTuningConfig
) -> None:
    """TF-IDF, tokenová větev i word2vec odmítnou trénink bez tokenů."""
    fold = replace(
        _clinical_fold(),
        train=tuple(
            replace(note, text=text)
            for note, text in zip(_clinical_fold().train[:2], ["!", "?"], strict=True)
        ),
    )
    with pytest.raises(
        InvalidInputError,
        match=re.escape(
            "Trénovací texty neobsahují žádný token, slovník by byl prázdný."
        ),
    ):
        SklearnStageScorer(
            config, dict.fromkeys(config.text_preprocessing), 7, tmp_path, Mock()
        ).score(config.branches[0].default_params(), fold)


@pytest.fixture
def word2vec_options() -> dict[str, ParamValue]:
    """Vrátí parametry malého deterministického modelu.

    Returns:
        Sedm hyperparametrů a provozní parametry gensim.
    """
    spec = word2vec_stage_config().representations[0]
    assert isinstance(spec, Word2VecRepresentation)
    return spec.default | {"workers": 1, "seed": 7}


@pytest.fixture
def word2vec_train() -> list[str]:
    """Vrátí syntetický trénink s jediným vzácným slovem.

    Returns:
        Opakované věty s jedním výskytem rare.
    """
    return ["alpha beta gamma delta"] * 20 + ["alpha beta"] * 20 + ["rare alpha"]


def test_word2vec_mean_repeats_tokens(
    word2vec_options: dict[str, ParamValue], word2vec_train: list[str]
) -> None:
    """Afinní standardizace zachová průměr s opakováním tokenů."""
    _, matrix = word2vec_features(
        word2vec_options, word2vec_train, ["alpha", "beta", "alpha alpha beta"]
    )
    np.testing.assert_allclose(
        matrix[2], (2 * matrix[0] + matrix[1]) / 3, rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("unknown", ["zzzneznamy", "rare"])
def test_word2vec_ignores_unknown_tokens(
    word2vec_options: dict[str, ParamValue], word2vec_train: list[str], unknown: str
) -> None:
    """Neznámé i podprahové slovo se z průměru vynechá."""
    _, matrix = word2vec_features(
        word2vec_options, word2vec_train, ["alpha", f"alpha {unknown}"]
    )
    np.testing.assert_array_equal(matrix[0], matrix[1])


def test_word2vec_empty_and_unknown_are_zero(
    word2vec_options: dict[str, ParamValue], word2vec_train: list[str]
) -> None:
    """Prázdný text a samé neznámé tokeny mají stejný konečný vektor."""
    _, matrix = word2vec_features(word2vec_options, word2vec_train, ["zzzneznamy", ""])
    np.testing.assert_array_equal(matrix[0], matrix[1])
    assert np.isfinite(matrix).all()


def test_word2vec_trains_only_on_train(
    word2vec_options: dict[str, ParamValue], word2vec_train: list[str]
) -> None:
    """Změna validace nijak nezmění trénovací matici."""
    first, _ = word2vec_features(word2vec_options, word2vec_train, ["alpha"])
    second, _ = word2vec_features(
        word2vec_options, word2vec_train, ["gamma delta", "beta"]
    )
    assert np.array_equal(first, second)


def test_word2vec_splits_long_documents(
    word2vec_options: dict[str, ParamValue], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Všech 25 000 tokenů dorazí k tréninku v kusech do 10 000."""
    original = sklearn_stage.Word2Vec.train
    observed = []

    def train(
        model: sklearn_stage.Word2Vec, sentences: list[list[str]], **kwargs: object
    ) -> object:
        observed.extend(len(sentence) for sentence in sentences)
        return original(model, sentences, **kwargs)

    monkeypatch.setattr(sklearn_stage.Word2Vec, "train", train)
    word2vec_features(word2vec_options, ["alpha " * 25000], ["alpha"])
    assert max(observed) <= 10000
    assert sum(observed) == 25000


def test_word2vec_empty_vocabulary_is_invalid_input(
    word2vec_options: dict[str, ParamValue], word2vec_train: list[str]
) -> None:
    """Příliš vysoké min_count vyvolá srozumitelnou doménovou chybu."""
    with pytest.raises(InvalidInputError, match="min_count"):
        word2vec_features(
            word2vec_options | {"min_count": 1000}, word2vec_train, ["alpha"]
        )


@pytest.fixture
def trained_word2vec(monkeypatch: pytest.MonkeyPatch) -> list[sklearn_stage.Word2Vec]:
    """Zachytí modely vytvořené featurizérem.

    Returns:
        Modely v pořadí volání konstruktoru.
    """
    original = sklearn_stage.Word2Vec
    models = []

    def model(**kwargs: ParamValue) -> sklearn_stage.Word2Vec:
        fitted = original(**kwargs)
        models.append(fitted)
        return fitted

    monkeypatch.setattr(sklearn_stage, "Word2Vec", model)
    return models


def test_word2vec_params_reach_gensim(
    word2vec_options: dict[str, ParamValue],
    word2vec_train: list[str],
    trained_word2vec: list[sklearn_stage.Word2Vec],
) -> None:
    """Model dostane všechny hyperparametry i workers a seed beze změny."""
    word2vec_features(word2vec_options, word2vec_train, ["alpha"])
    assert len(trained_word2vec) == 1
    assert {
        name: getattr(trained_word2vec[0], name) for name in word2vec_options
    } == word2vec_options


def test_word2vec_features_are_standardized_word_means(
    word2vec_options: dict[str, ParamValue],
    word2vec_train: list[str],
    trained_word2vec: list[sklearn_stage.Word2Vec],
) -> None:
    """Příznaky odpovídají průměru surových slovních vektorů a populačnímu škálování."""
    train, _ = word2vec_features(word2vec_options, word2vec_train, ["alpha"])
    wv = trained_word2vec[0].wv
    raw = np.array([
        np.mean([wv[word] for word in text.split() if word in wv], axis=0)
        for text in word2vec_train
    ])
    np.testing.assert_allclose(
        train, (raw - raw.mean(axis=0)) / raw.std(axis=0), rtol=1e-4, atol=1e-5
    )


def test_word2vec_branch_scores(
    tmp_path: Path, trained_word2vec: list[sklearn_stage.Word2Vec]
) -> None:
    """Větev word2vec vrátí pět skóre a sedm příznaků."""
    config = word2vec_stage_config()
    fold = _clinical_fold()
    normalization = TextNormalization(
        acronyms={"COPD": "chronic obstructive pulmonary disease"}, variants={}
    )
    result = SklearnStageScorer(
        config,
        dict.fromkeys(config.text_preprocessing, normalization),
        7,
        tmp_path,
        Mock(),
    ).score(config.branches[0].default_params(), fold)
    assert len(result.scores) == 5
    assert all(len(scores) == len(fold.test) for scores in result.scores.values())
    assert result.n_features == 7
    model = trained_word2vec[0]
    assert "pulmonary" in model.wv.key_to_index
    assert "copd" not in model.wv.key_to_index
    assert model.workers == 1
    assert model.seed == 7


def test_stage_inputs_follow_representation(
    tmp_path: Path, trained_word2vec: list[sklearn_stage.Word2Vec]
) -> None:
    """Větve encoderu i word2vec dostanou text podle úrovně reprezentace."""
    config = load_stage_config(REPOSITORY / "config/stage_embeddings.json")
    fold = _clinical_fold()
    fold = replace(
        fold,
        train=tuple(
            replace(n, text=n.text + " Pt with COPD and DM2 noted.") for n in fold.train
        ),
        test=tuple(
            replace(n, text=n.text + " Pt with COPD and DM2 noted.") for n in fold.test
        ),
    )
    embeddings = Mock()
    embeddings.embed.side_effect = lambda _spec, texts: np.arange(
        len(texts) * 4, dtype=float
    ).reshape(len(texts), 4)
    scorer = SklearnStageScorer(
        config,
        stage_normalizations(REPOSITORY_DIAGNOSIS_CANDIDATES, config),
        7,
        tmp_path,
        embeddings,
    )
    encoder = next(b for b in config.branches if b.name == "mmbert-base-mean/logreg")
    scorer.score(encoder.default_params(), fold)
    assert embeddings.embed.call_args_list == [
        call(config.encoders[0], [n.text for n in (*fold.train, *fold.test)])
    ]
    dictionary_encoder = next(
        b for b in config.branches if b.name == "mmbert-base-mean-dictionary/logreg"
    )
    scorer.score(dictionary_encoder.default_params(), fold)
    assert embeddings.embed.call_args_list[1].args[0] == config.encoders[1]
    normalized_texts = embeddings.embed.call_args_list[1].args[1]
    assert len(normalized_texts) == len(fold.train) + len(fold.test)
    for text in normalized_texts:
        assert "chronic obstructive pulmonary disease" in text
        assert "diabetes" in text
        assert all(acronym not in text for acronym in ("COPD", "copd", "DM2", "dm2"))
        assert text == text.lower()
    word2vec = next(b for b in config.branches if b.name == "word2vec/logreg")
    scorer.score(
        word2vec.default_params()
        | {"word2vec.min_count": 1, "word2vec.vector_size": 7, "word2vec.epochs": 3},
        fold,
    )
    vocabulary = trained_word2vec[0].wv.key_to_index
    assert {
        "chronic",
        "obstructive",
        "pulmonary",
        "disease",
        "diabetes",
    } <= vocabulary.keys()
    assert "copd" not in vocabulary
    assert "dm2" not in vocabulary


def _golden_scores(scratch: Path, jobs: int = 1) -> dict:
    configurations = {
        "classical": load_stage_config(
            REPOSITORY / "config/stage_classical.json"
        ).model_copy(update={"lsa": StageLsa(components=3, max_features=100_000)}),
        "stage": stage_config(),
        "tokens": token_stage_config(),
        "word2vec": word2vec_stage_config(),
        "embeddings": load_stage_config(REPOSITORY / "config/stage_embeddings.json"),
    }
    result = {}
    embeddings = Mock()
    embeddings.embed.side_effect = lambda _spec, texts: np.array(
        [
            [len(text), text.count("e"), text.count(" "), sum(map(ord, text)) % 997]
            for text in texts
        ],
        dtype=float,
    )
    for name, config in configurations.items():
        scorer = SklearnStageScorer(
            config,
            stage_normalizations(REPOSITORY_DIAGNOSIS_CANDIDATES, config),
            7,
            scratch,
            embeddings,
            jobs=jobs,
        )
        for branch in config.branches:
            if name == "embeddings" and branch.name not in {
                "mmbert-base-mean-dictionary/logreg",
                "mmbert-base-mean/logreg",
            }:
                continue
            scored = scorer.score(branch.default_params(), _clinical_fold())
            result[f"{name}/{branch.name}"] = {
                "n_features": scored.n_features,
                "converged": scored.converged,
                "scores": {
                    diagnosis.value: list(values)
                    for diagnosis, values in scored.scores.items()
                },
            }
    return result


@pytest.mark.parametrize("jobs", [1, 3])
@pytest.mark.usefixtures("loky_executor")
def test_scores_match_golden_values(tmp_path: Path, jobs: int) -> None:
    """Hlídá skóre proti kódu před optimalizací.

    Po upgradu scikit-learn, LightGBM, gensimu, fastTextu nebo numpy se vzor
    přegeneruje funkcí _golden_scores na main před jakoukoli změnou kódu,
    samostatným commitem test(stage): aktualizuj zlatý vzor skóre stupně.
    """
    expected = json.loads(
        (REPOSITORY / "tests/fixtures/stage_golden.json").read_text(encoding="utf-8")
    )
    with parallel_config(max_nbytes=0):
        actual = _golden_scores(tmp_path, jobs)
    assert actual.keys() == expected.keys()
    for name, scored in actual.items():
        assert scored["n_features"] == expected[name]["n_features"]
        assert scored["converged"] == expected[name]["converged"]
        for diagnosis, values in scored["scores"].items():
            tolerance = (
                {"rel": 1e-4, "abs": 1e-9}
                if name.startswith("word2vec/")
                else {"rel": 1e-9, "abs": 1e-12}
            )
            assert values == pytest.approx(
                expected[name]["scores"][diagnosis],
                rel=tolerance["rel"],
                abs=tolerance["abs"],
            )


def test_preprocessing_runs_once_per_text_and_representation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Každý text normalizuje právě jednou napříč foldy a rodinami."""
    preprocess = Mock(wraps=sklearn_classifier.tfidf_preprocessor)
    monkeypatch.setattr(sklearn_classifier, "tfidf_preprocessor", preprocess)
    config = load_stage_config(REPOSITORY / "config/stage_classical.json").model_copy(
        update={"lsa": StageLsa(components=3, max_features=100_000)}
    )
    scorer = SklearnStageScorer(
        config,
        stage_normalizations(REPOSITORY_DIAGNOSIS_CANDIDATES, config),
        7,
        tmp_path,
        Mock(),
    )
    original = _clinical_fold()
    corpus = (*original.train, *original.test)
    for fold in patient_holdouts(corpus, 2, 7, StratifiedGroupFolds()):
        for name in ("word/logreg", "word/complement_nb"):
            branch = next(branch for branch in config.branches if branch.name == name)
            scorer.score(branch.default_params(), fold)
    assert preprocess.call_count == len({note.text for note in corpus})


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.usefixtures("loky_executor")
def test_parallel_fits_report_nonconvergence(tmp_path: Path, jobs: int) -> None:
    """Limit jedné iterace liblinear hlásí nekonvergenci i z procesu."""
    config = stage_config(
        families=[
            {
                "family": "linear_svm",
                "space": {
                    "params": [
                        {"kind": "categorical", "name": "C", "choices": [1.0]},
                        {"kind": "categorical", "name": "max_iter", "choices": [1]},
                    ]
                },
                "default": {"C": 1.0, "max_iter": 1},
            }
        ]
    )
    scored = SklearnStageScorer(
        config, dict.fromkeys(config.text_preprocessing), 7, tmp_path, Mock(), jobs=jobs
    ).score(config.branches[0].default_params(), _clinical_fold())
    assert scored.converged is False


@pytest.mark.usefixtures("loky_executor")
def test_parallel_memmap_stays_in_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicitní scratch má přednost před neplatným výchozím adresářem."""
    invalid = tmp_path / "file"
    invalid.touch()
    monkeypatch.setenv("JOBLIB_TEMP_FOLDER", str(invalid))
    config = stage_config()
    with parallel_config(max_nbytes=0):
        scored = SklearnStageScorer(
            config,
            dict.fromkeys(config.text_preprocessing),
            7,
            tmp_path / "scratch",
            Mock(),
            jobs=2,
        ).score(config.branches[0].default_params(), _clinical_fold())
    assert len(scored.scores) == len(Diagnosis)
    assert (tmp_path / "scratch").is_dir()
    assert not any((tmp_path / "scratch").iterdir())


@given(
    analyzer=st.sampled_from(["word", "char_wb"]),
    bounds=st.tuples(st.integers(1, 6), st.integers(1, 6)),
    document_frequency=st.tuples(
        st.integers(1, 3), st.floats(min_value=0.5, max_value=1.0)
    ),
    sublinear_tf=st.booleans(),
    max_features=st.one_of(st.none(), st.integers(1, 50)),
)
def test_tfidf_features_match_vectorizer(
    analyzer: Literal["word", "char_wb"],
    bounds: tuple[int, int],
    document_frequency: tuple[int, float],
    *,
    sublinear_tf: bool,
    max_features: int | None,
) -> None:
    """Výřezy a váhy odpovídají nezávislému vektorizéru včetně remíz stropu."""
    min_df, max_df = document_frequency
    upper = 3 if analyzer == "word" else 6
    low, high = sorted(min(value, upper) for value in bounds)
    params = TfidfParameters(
        ngram_min=low,
        ngram_max=high,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
    )
    fold = _clinical_fold()
    train = [note.text for note in fold.train]
    validation = [*[note.text for note in fold.test], "zzzz zzzz"]
    counts = ngram_counts(analyzer, (1, upper), [*train, *validation])
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=(low, high),
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        max_features=max_features,
        lowercase=False,
    )
    try:
        expected_train = vectorizer.fit_transform(train)
    except ValueError as error:
        with pytest.raises(ValueError, match=re.escape(str(error))):
            tfidf_features(counts, train, validation, params, max_features)
        return
    expected_validation = vectorizer.transform(validation)
    actual = tfidf_features(counts, train, validation, params, max_features)
    for ours, expected in zip(
        actual, (expected_train, expected_validation), strict=True
    ):
        assert ours.shape == expected.shape
        np.testing.assert_allclose(
            ours.toarray(), expected.toarray(), rtol=0, atol=1e-12
        )


@pytest.mark.parametrize("case", ["min_df", "pruning", "empty"])
def test_tfidf_features_errors_match_vectorizer(case: str) -> None:
    """Chyby mezí a prořezání zachovají hlášku sklearn, prázdný trénink sdílenou."""
    fold = _clinical_fold()
    train = [note.text for note in fold.train] if case != "empty" else ["!", "?"]
    validation = [note.text for note in fold.test]
    # Ve 101 trénovacích zprávách mají termy DF 10, 50, 51 nebo 101; interval [2, 2.02] je prázdný.
    params = TfidfParameters(
        ngram_max=1,
        min_df=len(train) + 1 if case == "min_df" else 2,
        max_df=0.02 if case == "pruning" else 1.0,
        sublinear_tf=True,
    )
    counts = ngram_counts("word", (1, 1), [*train, *validation])
    if case == "empty":
        with pytest.raises(InvalidInputError, match=re.escape(EMPTY_VOCABULARY)):
            tfidf_features(counts, train, validation, params, None)
        return
    vectorizer = TfidfVectorizer(
        lowercase=False,
        min_df=params.min_df,
        max_df=params.max_df,
    )
    with pytest.raises(ValueError, match=r"max_df|After pruning") as error:
        vectorizer.fit_transform(train)
    with pytest.raises(ValueError, match=re.escape(str(error.value))):
        tfidf_features(counts, train, validation, params, None)


@pytest.fixture
def counts_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[SklearnStageScorer, Mock]:
    """Vrátí klasický scorer a záznam staveb matice četností.

    Returns:
        Scorer se syntetickými vstupy a průchozí mock stavby matice.
    """
    build = Mock(wraps=sklearn_stage.ngram_counts)
    monkeypatch.setattr(sklearn_stage, "ngram_counts", build)
    config = load_stage_config(REPOSITORY / "config/stage_classical.json").model_copy(
        update={"lsa": StageLsa(components=3, max_features=100_000)}
    )
    return SklearnStageScorer(
        config, dict.fromkeys(config.text_preprocessing), 7, tmp_path, Mock()
    ), build


def test_ngram_counts_built_once_per_widest_range(
    counts_scorer: tuple[SklearnStageScorer, Mock],
) -> None:
    """Druhý fold i užší rozsahy využijí společnou matici všech textů."""
    scorer, build = counts_scorer
    original = _clinical_fold()
    for fold in patient_holdouts(
        (*original.train, *original.test), 2, 7, StratifiedGroupFolds()
    ):
        for name, bounds in [
            ("char_wb", (3, 5)),
            ("char_wb", (2, 6)),
            ("char_wb", (2, 4)),
            ("word", (1, 3)),
            ("word", (1, 1)),
        ]:
            branch = next(
                b for b in scorer.config.branches if b.name == f"{name}/logreg"
            )
            scorer.score(
                branch.default_params()
                | {f"{name}.ngram_min": bounds[0], f"{name}.ngram_max": bounds[1]},
                fold,
            )
    assert [c.args[:2] for c in build.call_args_list] == [
        ("char_wb", (3, 5)),
        ("char_wb", (2, 6)),
        ("word", (1, 3)),
    ]


def test_ngram_counts_range_grows_to_union(
    counts_scorer: tuple[SklearnStageScorer, Mock],
) -> None:
    """Přestavba ponechá horní mez předchozího rozsahu."""
    scorer, build = counts_scorer
    branch = next(b for b in scorer.config.branches if b.name == "char_wb/logreg")
    for low, high in [(3, 5), (2, 4), (3, 5)]:
        scorer.score(
            branch.default_params()
            | {"char_wb.ngram_min": low, "char_wb.ngram_max": high},
            _clinical_fold(),
        )
    assert [c.args[:2] for c in build.call_args_list] == [
        ("char_wb", (3, 5)),
        ("char_wb", (2, 5)),
    ]


def test_ngram_counts_rebuild_for_unseen_text(
    counts_scorer: tuple[SklearnStageScorer, Mock],
) -> None:
    """Nový text rozšíří korpus cache a skóre odpovídá čerstvému scoreru."""
    scorer, build = counts_scorer
    branch = next(b for b in scorer.config.branches if b.name == "word/logreg")
    original = _clinical_fold()
    scorer.score(branch.default_params(), original)
    new_text = "novel synthetic validation text"
    fold = replace(
        original, test=(replace(original.test[0], text=new_text), *original.test[1:])
    )
    actual = scorer.score(branch.default_params(), fold)
    assert build.call_count == 2
    assert set(build.call_args.args[2]) == {
        n.text.lower() for n in (*original.train, *original.test)
    } | {new_text}
    expected = replace(scorer).score(branch.default_params(), fold)
    for diagnosis in Diagnosis:
        assert actual.scores[diagnosis] == pytest.approx(
            expected.scores[diagnosis], abs=1e-12
        )
