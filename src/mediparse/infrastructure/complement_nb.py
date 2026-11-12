"""Binární Complement NB s log-odds bez saturace pravděpodobností."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import ComplementNB
from sklearn.utils.validation import check_is_fitted, validate_data

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy import ndarray
    from scipy.sparse import csr_matrix


class ComplementNbLogOdds(ClassifierMixin, BaseEstimator):
    """Sklearn obal CNB zachovávající pořadí skóre i při saturaci pravděpodobností."""

    def __init__(self, alpha: float = 1.0) -> None:
        """Uloží vyhlazení bez naučeného stavu pro sklearn clone."""
        self.alpha = alpha

    def fit(
        self, features: csr_matrix | ndarray, labels: Sequence[bool] | ndarray
    ) -> Self:
        """Naučí binární CNB se stejnými výchozími parametry jako knihovna.

        Returns:
            Tentýž estimator s naučeným modelem a třídami.

        Raises:
            ValueError: Vstup nemá právě dvě třídy nebo platný tvar.
        """
        features, labels = validate_data(self, features, labels, accept_sparse="csr")
        self.classes_ = np.unique(labels)
        if len(self.classes_) != 2:  # ruff: ignore[magic-value-comparison]
            msg = "Complement NB vyžaduje právě dvě třídy."
            raise ValueError(msg)
        self.model_ = ComplementNB(alpha=self.alpha).fit(features, labels)
        return self

    # predict_proba se záměrně nevystavuje: auto tuneru i classify by vzaly saturované skóre.
    def decision_function(self, features: csr_matrix | ndarray) -> ndarray:
        """Vrátí rozdíl společných log-věrohodností pozitivní a negativní třídy.

        Returns:
            Jedno log-odds skóre na řádek vstupu.
        """
        check_is_fitted(self, "model_")
        features = validate_data(self, features, accept_sparse="csr", reset=False)
        likelihood = self.model_.predict_joint_log_proba(features)
        return likelihood[:, 1] - likelihood[:, 0]

    def predict(self, features: csr_matrix | ndarray) -> ndarray:
        """Přiřadí třídu podle znaménka skóre; rovnost rozhodne jako knihovní argmax.

        Returns:
            Labely v pořadí vstupních řádků.
        """
        scores = self.decision_function(features)
        return self.classes_[(scores > 0).astype(int)]
