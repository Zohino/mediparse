"""Továrna rodin klasifikátorů a skóre pozitivní třídy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol, assert_never, cast

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from mediparse.domain.representation import ClassifierFamily
from mediparse.infrastructure.complement_nb import ComplementNbLogOdds
from mediparse.infrastructure.fasttext_classifier import FastTextClassifier
from mediparse.infrastructure.nb_svm import NbSvmClassifier

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from sklearn.base import BaseEstimator

    from mediparse.domain.seed import Seed


# Pevný počet vláken zachovává výsledky mezi stroji i počty procesů.
LIGHTGBM_THREADS: Final = 4


def classifier_estimator(
    family: ClassifierFamily, seed: Seed, scratch: Path
) -> BaseEstimator:
    """Vytvoří estimator rodiny; LSA provádí scorer stupně.

    Returns:
        Nenatrénovaný estimator s výchozími hyperparametry rodiny.
    """
    match family:
        case ClassifierFamily.COMPLEMENT_NB:
            return ComplementNbLogOdds()
        case ClassifierFamily.LOGREG:
            return LogisticRegression()
        case ClassifierFamily.LINEAR_SVM:
            return LinearSVC(random_state=seed)
        case ClassifierFamily.NB_SVM:
            return NbSvmClassifier(random_state=seed)
        case ClassifierFamily.LIGHTGBM | ClassifierFamily.LIGHTGBM_LSA:
            # LightGBM doporučuje force_row_wise pro stabilitu mezi počty vláken.
            return LGBMClassifier(
                random_state=seed,
                deterministic=True,
                n_jobs=LIGHTGBM_THREADS,
                force_row_wise=True,
                verbose=-1,
            )
        case ClassifierFamily.FASTTEXT:
            return FastTextClassifier(scratch, random_state=seed)
        case _:
            assert_never(family)


def positive_scores(model: BaseEstimator, matrix: object) -> np.ndarray:
    """Vybere skóre pozitivní třídy na škále odezvy modelu.

    Returns:
        Pravděpodobnosti, má-li je model, jinak rozhodovací skóre.
    """
    if callable(predict_proba := getattr(model, "predict_proba", None)):
        return predict_proba(matrix)[:, 1]
    return cast("_DecisionModel", model).decision_function(matrix)


class _DecisionModel(Protocol):
    def decision_function(self, matrix: object) -> np.ndarray: ...
