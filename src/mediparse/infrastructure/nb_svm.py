"""Binární NB-SVM s log-count poměrem a interpolací Wang a Manning 2012."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted, validate_data

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy import ndarray


def _binarized(features: csr_matrix | ndarray) -> csr_matrix:
    binary = csr_matrix(features, dtype=np.float64, copy=True)
    binary.sum_duplicates()
    # Rovnice 2 používá přítomnost f>0; velikost TF-IDF vah do NB-SVM nevstupuje.
    binary.data[:] = binary.data > 0
    return binary


class NbSvmClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn estimator NB-SVM; alpha vyhlazuje četnosti, beta interpoluje váhy SVM."""

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.25,
        # get_params vystavuje C pod stejným jménem jako LR a SVM pro ladění S39j.
        C: float = 1.0,  # ruff: ignore[invalid-argument-name]
        random_state: int | None = None,
    ) -> None:
        """Uloží parametry článku bez naučeného stavu pro sklearn clone."""
        self.alpha = alpha
        self.beta = beta
        self.C = C
        self.random_state = random_state

    def fit(self, features: csr_matrix | ndarray, labels: Sequence[bool]) -> Self:
        """Naučí poměr četností tříd a interpolované váhy nad binárním vstupem.

        Returns:
            Tentýž estimator s naučenými r_, coef_, intercept_ a classes_.

        Raises:
            ValueError: Vstup neobsahuje právě dvě třídy nebo má neplatný tvar.
        """
        features, labels = validate_data(self, features, labels, accept_sparse="csr")
        self.classes_ = np.unique(labels)
        if len(self.classes_) != 2:  # ruff: ignore[magic-value-comparison]
            msg = "NB-SVM vyžaduje právě dvě třídy."
            raise ValueError(msg)
        binary = _binarized(features)
        positive = self.alpha + binary.T @ (labels == self.classes_[1])
        negative = self.alpha + binary.T @ (labels == self.classes_[0])
        self.r_ = np.log((positive / positive.sum()) / (negative / negative.sum()))
        # Škálování jediné kopie na místě šetří paměť matice znakových n-gramů.
        binary.data *= self.r_[binary.indices]
        svm = LinearSVC(C=self.C, random_state=self.random_state).fit(binary, labels)
        weights = svm.coef_
        self.coef_ = (1 - self.beta) * np.mean(np.abs(weights)) + self.beta * weights
        self.intercept_ = svm.intercept_
        return self

    def decision_function(self, features: csr_matrix | ndarray) -> ndarray:
        """Vrátí rozhodovací skóre s původním biasem SVM.

        Returns:
            Jedno skóre pro každý řádek vstupu.
        """
        check_is_fitted(self, ("r_", "coef_", "intercept_"))
        features = validate_data(self, features, accept_sparse="csr", reset=False)
        return (
            _binarized(features) @ (self.r_ * self.coef_.ravel()) + self.intercept_[0]
        )

    def predict(self, features: csr_matrix | ndarray) -> ndarray:
        """Přiřadí třídu podle znaménka rozhodovacího skóre.

        Returns:
            Labely ve stejném pořadí jako vstupní řádky.
        """
        scores = self.decision_function(features)
        return self.classes_[(scores > 0).astype(int)]
