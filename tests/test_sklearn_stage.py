"""Skutečné fity a překlad chyb klasického stupně."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from random import Random
from typing import TYPE_CHECKING
from unittest.mock import Mock, call

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from mediparse.application.patient_cv import patient_holdouts
from mediparse.domain.clinical_text import TextNormalization
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.split import Holdout
from mediparse.domain.stage_tuning import StageLsa
from mediparse.infrastructure import sklearn_stage
from mediparse.infrastructure.diagnosis_candidates_file import load_text_normalization
from mediparse.infrastructure.fasttext_classifier import FastTextClassifier
from mediparse.infrastructure.sklearn_classifier import tfidf_vectorizer
from mediparse.infrastructure.sklearn_split import StratifiedGroupFolds
from mediparse.infrastructure.sklearn_stage import (
    SklearnStageScorer,
    encoder_features,
    token_documents,
)
from mediparse.infrastructure.stage_config import load_stage_config
from tests.hf_encoder_support import SPEC
from tests.screening_support import notes
from tests.stage_support import stage_config, token_stage_config
from tests.support import REPOSITORY, REPOSITORY_DIAGNOSIS_CANDIDATES

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from scipy.sparse import csr_matrix
    from sklearn.base import BaseEstimator
    from sklearn.decomposition import TruncatedSVD

    from mediparse.domain.representation import ClassifierFamily
    from mediparse.domain.stage_tuning import StageTuningConfig


def test_logreg_parameters_change_scores(tmp_path: Path) -> None:
    """Silnější a slabší regularizace dávají různá skóre na stejné syntetice."""
    config = stage_config()
    scorer = SklearnStageScorer(config, None, 7, tmp_path, Mock())
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
    scorer = SklearnStageScorer(config, None, 7, tmp_path, Mock())
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
    result = SklearnStageScorer(config, None, 7, tmp_path, Mock()).score(
        branch.default_params(), fold
    )
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
    scored = SklearnStageScorer(config, normalization, 7, tmp_path, Mock()).score(
        params, fold
    )
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
    first = SklearnStageScorer(config, None, 7, tmp_path, Mock()).score(
        branch.default_params(), fold
    )
    repeated = SklearnStageScorer(config, None, 7, tmp_path, Mock()).score(
        branch.default_params(), fold
    )
    other = SklearnStageScorer(config, None, 29, tmp_path, Mock()).score(
        branch.default_params(), fold
    )
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
    scorer = SklearnStageScorer(config, None, 7, tmp_path, Mock())
    scored = scorer.score(config.branches[0].default_params(), _clinical_fold())
    assert scored.converged is (failed_call is None)
    assert calls == 5


def test_min_df_above_training_documents_is_refused(tmp_path: Path) -> None:
    """Sklearn odmítne min_df větší než počet trénovacích dokumentů."""
    config = stage_config()
    scorer = SklearnStageScorer(config, None, 7, tmp_path, Mock())
    params = config.branches[0].default_params() | {"r1.min_df": 100000}
    with pytest.raises(InvalidInputError, match="min_df"):
        scorer.score(params, _clinical_fold())


def test_encoder_branch_scores_cached_embeddings(tmp_path: Path) -> None:
    """Encoder poskytne pět vektorů skóre a šířku čtyř příznaků."""
    base = stage_config().model_dump(mode="json")
    config = stage_config(
        representations=[SPEC.model_dump(mode="json")], families=base["families"][:1]
    )
    fold = _clinical_fold()
    vectors = {
        n.text: [float(i % 3), float(i % 5), float(i % 7), float(i % 11)]
        for i, n in enumerate((*fold.train, *fold.test))
    }
    embeddings = Mock()
    embeddings.embed.side_effect = lambda _spec, texts: np.array([
        vectors[text] for text in texts
    ])
    scored = SklearnStageScorer(config, None, 7, tmp_path, embeddings).score(
        config.branches[0].default_params(), fold
    )
    assert len(scored.scores) == 5
    assert all(len(scores) == len(fold.test) for scores in scored.scores.values())
    assert scored.n_features == 4
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
    scored = SklearnStageScorer(config, normalization, 7, tmp_path, Mock()).score(
        config.branches[0].default_params(), fold
    )
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
    SklearnStageScorer(config, None, 7, tmp_path, Mock()).score(
        config.branches[0].default_params(), _clinical_fold()
    )
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


def test_token_documents_joins_analyzer_tokens() -> None:
    """Slovní analyzátor odstraní interpunkci a jednopísmenná slova."""
    train, test = token_documents(None, ["Insulin, glucose!\nA b"], ["Fracture."])
    assert train.dtype == object
    assert test.dtype == object
    np.testing.assert_array_equal(train, ["insulin glucose"])
    np.testing.assert_array_equal(test, ["fracture"])


def test_token_documents_applies_its_normalization() -> None:
    """Rozvinutí COPD dodá stejná slova jako TF-IDF."""
    train, test = token_documents(
        TextNormalization(
            acronyms={"COPD": "chronic obstructive pulmonary disease"}, variants={}
        ),
        ["COPD"],
        ["COPD"],
    )
    np.testing.assert_array_equal(train, ["chronic obstructive pulmonary disease"])
    np.testing.assert_array_equal(test, ["chronic obstructive pulmonary disease"])


@pytest.mark.parametrize("config", [stage_config(), token_stage_config()])
def test_empty_vocabulary_is_invalid_input(
    tmp_path: Path, config: StageTuningConfig
) -> None:
    """TF-IDF i tokenová větev odmítnou trénink bez tokenů."""
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
        SklearnStageScorer(config, None, 7, tmp_path, Mock()).score(
            config.branches[0].default_params(), fold
        )
