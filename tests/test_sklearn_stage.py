"""Skutečné fity a překlad chyb klasického stupně."""

from __future__ import annotations

import json
from dataclasses import replace
from random import Random
from typing import TYPE_CHECKING

import pytest
from sklearn.linear_model import LogisticRegression

from mediparse.application.patient_cv import patient_holdouts
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure import sklearn_stage
from mediparse.infrastructure.diagnosis_candidates_file import load_text_normalization
from mediparse.infrastructure.sklearn_classifier import tfidf_vectorizer
from mediparse.infrastructure.sklearn_split import StratifiedGroupFolds
from mediparse.infrastructure.sklearn_stage import SklearnStageScorer
from mediparse.infrastructure.stage_config import load_stage_config
from tests.screening_support import notes
from tests.stage_support import stage_config
from tests.support import REPOSITORY, REPOSITORY_DIAGNOSIS_CANDIDATES

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import numpy as np
    from sklearn.base import BaseEstimator

    from mediparse.domain.screening import ClassifierFamily
    from mediparse.domain.split import Holdout


def test_logreg_parameters_change_scores(tmp_path: Path) -> None:
    """Silnější a slabší regularizace dávají různá skóre na stejné syntetice."""
    config = stage_config()
    scorer = SklearnStageScorer(config, None, 2, 7, tmp_path)
    fold = patient_holdouts(notes(visits=10), 2, 7, StratifiedGroupFolds())[0]
    params = config.branches[0].default_params()
    low = scorer.score(params | {"logreg.C": 1e-4}, fold)
    high = scorer.score(params | {"logreg.C": 1e2}, fold)
    assert low.scores[Diagnosis.CKD] != high.scores[Diagnosis.CKD]


def test_lightgbm_receives_lsa_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Estimator LightGBM vidí právě tři komponenty místo řídkého TF-IDF."""
    config = stage_config()
    data = config.model_dump(mode="json")
    data["families"][0] = {
        "family": "lightgbm_lsa",
        "space": {
            "params": [{"kind": "categorical", "name": "n_estimators", "choices": [10]}]
        },
        "default": {"n_estimators": 10},
    }
    config = type(config).model_validate_json(json.dumps(data))
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
    scorer = SklearnStageScorer(config, None, 3, 7, tmp_path)
    fold = patient_holdouts(notes(visits=10), 2, 7, StratifiedGroupFolds())[0]
    scorer.score(config.branches[0].default_params(), fold)
    assert observed == [3] * 5


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
    config = load_stage_config(REPOSITORY / "config/stage_classical.json")
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
    scored = SklearnStageScorer(config, normalization, 3, 7, tmp_path).score(
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
    config = load_stage_config(REPOSITORY / "config/stage_classical.json")
    branch = next(b for b in config.branches if b.name == f"char_wb/{family}")
    random = Random(117)
    vocabulary = [f"synthetic{i}" for i in range(100)]
    corpus = tuple(
        replace(n, text=" ".join(random.choices(vocabulary, k=40)))
        for n in notes(visits=10)
    )
    fold = patient_holdouts(corpus, 2, 7, StratifiedGroupFolds())[0]
    first = SklearnStageScorer(config, None, 3, 7, tmp_path).score(
        branch.default_params(), fold
    )
    repeated = SklearnStageScorer(config, None, 3, 7, tmp_path).score(
        branch.default_params(), fold
    )
    other = SklearnStageScorer(config, None, 3, 29, tmp_path).score(
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
    scorer = SklearnStageScorer(config, None, 2, 7, tmp_path)
    scored = scorer.score(config.branches[0].default_params(), _clinical_fold())
    assert scored.converged is (failed_call is None)
    assert calls == 5


def test_min_df_above_training_documents_is_refused(tmp_path: Path) -> None:
    """Sklearn odmítne min_df větší než počet trénovacích dokumentů."""
    config = stage_config()
    scorer = SklearnStageScorer(config, None, 2, 7, tmp_path)
    params = config.branches[0].default_params() | {"r1.min_df": 100000}
    with pytest.raises(InvalidInputError, match="min_df"):
        scorer.score(params, _clinical_fold())
