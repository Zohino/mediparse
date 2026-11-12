"""Log-odds Complement NB proti knihovnímu orákulu a regrese saturace tuneru."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.base import clone
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.naive_bayes import ComplementNB

from mediparse.domain.classical_screening import ClassifierFamily
from mediparse.infrastructure.complement_nb import ComplementNbLogOdds
from mediparse.infrastructure.sklearn_screening import (
    ScreeningFeatures,
    SklearnThresholdTuner,
    threshold_splits,
)


def test_log_odds_matches_library_probabilities_and_predictions() -> None:
    """Knihovní orákulum platí i pro nulové skóre a malé kladné log-odds pod jedničkou."""
    features = csr_matrix([[1.0, 3.0], [2.0, 4.0], [3.0, 1.0], [4.0, 2.0]])
    labels = [False, False, True, True]
    reference = ComplementNB().fit(features, labels)
    model = ComplementNbLogOdds().fit(features, labels)
    evaluated = csr_matrix([
        [1.0, 3.0],
        [2.0, 4.0],
        [3.0, 1.0],
        [4.0, 2.0],
        [0.0, 0.0],
        [0.1, 0.0],
    ])
    probabilities = reference.predict_proba(evaluated)

    np.testing.assert_array_equal(
        model.predict(evaluated), reference.predict(evaluated)
    )
    np.testing.assert_allclose(
        model.decision_function(evaluated),
        np.log(probabilities[:, 1]) - np.log(probabilities[:, 0]),
    )
    assert not hasattr(model, "predict_proba")


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ([False, False], "právě dvě třídy"),
        ([False, True, False], "inconsistent numbers"),
    ],
)
def test_fit_rejects_one_class_and_invalid_shape(
    labels: list[bool], message: str
) -> None:
    """Jedna třída i nesoulad počtu řádků jsou neplatný vstup estimatoru."""
    with pytest.raises(ValueError, match=message):
        ComplementNbLogOdds().fit(csr_matrix([[1.0], [2.0]]), labels)


def test_fit_rejects_more_than_two_classes() -> None:
    """Obal je binární i pro labely mimo bool."""
    with pytest.raises(ValueError, match="právě dvě třídy"):
        ComplementNbLogOdds().fit(
            csr_matrix([[1.0], [2.0], [3.0]]), np.array([0, 1, 2])
        )


def test_clone_preserves_alpha() -> None:
    """Sklearn clone zachová vyhlazení pro vnitřní fity tuneru."""
    assert clone(ComplementNbLogOdds(alpha=0.3)).alpha == pytest.approx(0.3)


def test_log_odds_resolves_constant_probability_regression() -> None:
    """Sonda 3: 90 řádků, 15 pozitivních a 5000 nulových příznaků saturuje CNB."""
    positives = 15
    labels = np.arange(90) < positives
    random = np.random.default_rng(0)
    features = csr_matrix(
        np.column_stack([
            np.full(90, 50.0),
            np.where(labels, 2.0, 1.0) + random.uniform(0, 1, 90),
            np.zeros((90, 5000)),
        ])
    )
    subjects = list(range(90))
    cv = threshold_splits(labels.tolist(), subjects, 3, 7)

    with pytest.raises(ValueError, match="constant predictions"):
        TunedThresholdClassifierCV(ComplementNB(), scoring="f1", cv=cv).fit(
            features, labels
        )

    model = SklearnThresholdTuner().fit(
        ClassifierFamily.COMPLEMENT_NB,
        ScreeningFeatures(features, None),
        labels.tolist(),
        subjects,
        3,
        7,
    )

    assert (
        len({item.score for item in model.classify(ScreeningFeatures(features, None))})
        > 1
    )
