"""Skóre SVM proti nezávislému sestavení příznaků a klasifikátorů."""

from __future__ import annotations

import functools
from dataclasses import replace
from typing import TYPE_CHECKING, Literal
from unittest.mock import Mock

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from mediparse.domain.clinical_text import TextNormalization
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.split import Holdout
from mediparse.domain.tuning import CategoricalParam
from mediparse.infrastructure import sklearn_tuning
from mediparse.infrastructure.sklearn_classifier import tfidf_preprocessor
from mediparse.infrastructure.sklearn_tuning import SklearnLinearSvmScorer
from mediparse.infrastructure.tuning_config import load_tuning_config
from tests.model_support import separable_notes
from tests.support import REPOSITORY
from tests.tuning_support import SVM_PARAMS

if TYPE_CHECKING:
    from collections.abc import Callable

    from mediparse.domain.tuning import ParamValue


def _fold() -> Holdout:
    train_patients = 20
    notes = [
        replace(
            note,
            labels=frozenset(Diagnosis) if note.labels else frozenset(),
            text=f"COPD COPD {note.text}",
        )
        for note in separable_notes(30)
    ]
    return Holdout(
        0,
        tuple(note for note in notes if note.subject_id <= train_patients),
        tuple(note for note in notes if note.subject_id > train_patients),
    )


@pytest.mark.parametrize(
    "normalization",
    [
        None,
        TextNormalization(
            acronyms={"COPD": "chronic obstructive pulmonary disease"}, variants={}
        ),
    ],
)
@pytest.mark.parametrize(
    ("min_df", "max_df", "sublinear_tf"), [(1, 1.0, True), (2, 0.9, False)]
)
@pytest.mark.parametrize("analyzer", ["word", "char_wb"])
def test_svm_scores_match_independent_word_bigram_pipeline(
    normalization: TextNormalization | None,
    analyzer: Literal["word", "char_wb"],
    min_df: int,
    max_df: float,
    *,
    sublinear_tf: bool,
) -> None:
    """Oba analyzátory a opakovaný token ověří všechny volby TF-IDF s přesností 1e-12."""
    fold, seed = _fold(), 7
    actual = SklearnLinearSvmScorer(normalization, seed).score(
        SVM_PARAMS
        | {
            "analyzer": analyzer,
            "min_df": min_df,
            "max_df": max_df,
            "sublinear_tf": sublinear_tf,
        },
        fold,
    )
    preprocessor = (
        None
        if normalization is None
        else functools.partial(
            tfidf_preprocessor,
            acronyms=dict(normalization.acronyms),
            variants=dict(normalization.variants),
        )
    )
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        preprocessor=preprocessor,
    )
    train = vectorizer.fit_transform([note.text for note in fold.train])
    test = vectorizer.transform([note.text for note in fold.test])
    assert set(actual.scores) == set(Diagnosis)
    for diagnosis in Diagnosis:
        model = LinearSVC(C=0.5, class_weight="balanced", random_state=seed)
        model.fit(train, [note.has_label(diagnosis) for note in fold.train])
        np.testing.assert_allclose(
            actual.scores[diagnosis], model.decision_function(test), rtol=0, atol=1e-12
        )
    assert actual.converged


def test_repository_space_matches_svm_parameters() -> None:
    """Scorer nepoužije výchozí hodnotu místo překlepu ve veřejném configu."""
    assert (
        load_tuning_config(REPOSITORY / "config/tuning.json").space.names
        == SklearnLinearSvmScorer(None, 0).parameters
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"c": -1.0},
        {"ngram_max": True},
        {"analyzer": "invalid"},
        {"min_df": 1000},
        {"class_weight": "typo"},
    ],
)
def test_invalid_svm_parameters_or_vocabulary_are_domain_errors(
    changes: dict[str, ParamValue],
) -> None:
    """Obecný prostor nemůže obejít meze konkrétního estimatoru."""
    with pytest.raises(InvalidInputError):
        SklearnLinearSvmScorer(None, 7).score(SVM_PARAMS | changes, _fold())


def test_svm_passes_seed_to_every_diagnosis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Všech pět estimatorů dostane seed scoreru i při deterministickém řešiči."""
    seed = 13
    factory = Mock(wraps=LinearSVC)
    monkeypatch.setattr(sklearn_tuning, "LinearSVC", factory)
    SklearnLinearSvmScorer(None, seed).score(SVM_PARAMS, _fold())
    assert [call.kwargs.get("random_state") for call in factory.call_args_list] == [
        seed
    ] * 5


def test_svm_keeps_nonconvergence_from_first_diagnosis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pozdější konvergence čtyř diagnóz nepřepíše nekonvergenci první."""
    convergence = Mock(side_effect=[False, True, True, True, True])

    def fit_converged(fit: Callable[[], object]) -> bool:
        fit()
        return convergence()

    monkeypatch.setattr(sklearn_tuning, "fit_converged", fit_converged)
    assert not SklearnLinearSvmScorer(None, 7).score(SVM_PARAMS, _fold()).converged


def test_invalid_svm_message_includes_parameter_cause() -> None:
    """Odmítnutí záporného C zachová příčinu validace bez textu zpráv."""
    with pytest.raises(InvalidInputError, match="Input should be greater than 0"):
        SklearnLinearSvmScorer(None, 7).score(SVM_PARAMS | {"c": -1.0}, _fold())


def _repository_parameter_cases() -> list[tuple[str, ParamValue]]:
    space = load_tuning_config(REPOSITORY / "config/tuning.json").space
    cases = []
    for param in space.params:
        values = (
            param.choices
            if isinstance(param, CategoricalParam)
            else (param.low, param.high)
        )
        cases.extend((param.name, value) for value in values)
    return cases


@pytest.mark.parametrize(("name", "value"), _repository_parameter_cases())
def test_repository_space_values_produce_finite_svm_scores(
    name: str, value: ParamValue
) -> None:
    """Každá volba a krajní hodnota veřejného prostoru funguje se skutečným scorerem."""
    space = load_tuning_config(REPOSITORY / "config/tuning.json").space
    params = {
        param.name: param.choices[0]
        if isinstance(param, CategoricalParam)
        else param.low
        for param in space.params
    }
    result = SklearnLinearSvmScorer(None, 7).score(params | {name: value}, _fold())
    assert all(np.isfinite(scores).all() for scores in result.scores.values())
