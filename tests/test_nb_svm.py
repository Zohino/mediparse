"""NB-SVM proti rovnicím Wang a Manning 2012 a invariantům řídkého vstupu."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy.sparse import csc_matrix, csr_matrix
from sklearn.base import clone
from sklearn.svm import LinearSVC

from mediparse.infrastructure.nb_svm import NbSvmClassifier

if TYPE_CHECKING:
    from collections.abc import Callable


def _features() -> csr_matrix:
    return csr_matrix([[2.0, 0.0], [0.5, 3.0], [0.0, 1.0]])


@pytest.mark.parametrize("representation", [csr_matrix, csc_matrix, np.array])
def test_log_count_ratio_matches_manual_equation_two(representation: Callable) -> None:
    """Binarizace a alpha=1 dají p=[3,2], q=[1,2], tedy r=[log 1,8, log 0,6]."""
    features = representation([[2.0, 0.0], [0.5, 3.0], [0.0, 1.0]])
    model = NbSvmClassifier(random_state=7).fit(features, [True, True, False])

    assert model.r_ == pytest.approx(np.log([1.8, 0.6]), rel=1e-9)


def test_interpolation_changes_weights_and_preserves_bias() -> None:
    """Rovnice 4 interpoluje pouze váhy s β=0,25, bias zůstane z LinearSVC."""
    weighted = np.array([[1, 0], [1, 1], [0, 1]]) * np.log([1.8, 0.6])
    reference = LinearSVC(C=1, random_state=7).fit(weighted, [True, True, False])
    model = NbSvmClassifier(random_state=7).fit(_features(), [True, True, False])

    expected = 0.75 * np.mean(np.abs(reference.coef_)) + 0.25 * reference.coef_
    np.testing.assert_allclose(model.coef_, expected)
    np.testing.assert_allclose(model.intercept_, reference.intercept_, rtol=1e-9)


def test_beta_one_matches_svm_scores_and_predictions() -> None:
    """Při β=1 je rozhodovací funkce i predikce shodná s neinterpolovanou SVM."""
    weighted = np.array([[1, 0], [1, 1], [0, 1]]) * np.log([1.8, 0.6])
    reference = LinearSVC(C=1, random_state=7).fit(weighted, [True, True, False])
    model = NbSvmClassifier(beta=1.0, random_state=7).fit(
        _features(), [True, True, False]
    )

    np.testing.assert_allclose(
        model.decision_function(_features()), reference.decision_function(weighted)
    )
    np.testing.assert_array_equal(
        model.predict(_features()), reference.predict(weighted)
    )


def test_dual_solver_preserves_regularization_and_seed() -> None:
    """Při 12 řádcích a 80 příznacích odpovídá skóre CSR orákulu s C=0,01 a seedem 13."""
    features = csr_matrix(np.random.default_rng(31).integers(0, 4, size=(12, 80)))
    labels = [True, False] * 6
    binary = (features.toarray() > 0).astype(float)
    positive = 1 + binary[np.array(labels)].sum(axis=0)
    negative = 1 + binary[~np.array(labels)].sum(axis=0)
    ratio = np.log((positive / positive.sum()) / (negative / negative.sum()))
    weighted = csr_matrix(binary * ratio)
    reference = LinearSVC(C=0.01, random_state=13).fit(weighted, labels)
    model = NbSvmClassifier(beta=1.0, C=0.01, random_state=13).fit(features, labels)

    np.testing.assert_allclose(
        model.decision_function(features), reference.decision_function(weighted)
    )


@given(st.lists(st.floats(0.01, 100, allow_nan=False), min_size=4, max_size=4))
def test_positive_rescaling_cannot_change_scores(scales: list[float]) -> None:
    """Kladné škálování nenulových hodnot při fitu i predikci nemění skóre."""
    original = _features()
    scaled = original.copy()
    scaled.data *= scales
    first = NbSvmClassifier(random_state=7).fit(original, [True, True, False])
    second = NbSvmClassifier(random_state=7).fit(scaled, [True, True, False])

    np.testing.assert_array_equal(
        first.decision_function(original), second.decision_function(scaled)
    )


def test_fit_and_scoring_preserve_input_bytes() -> None:
    """Binarizace a škálování nesmí přepsat žádné pole vstupní CSR matice."""
    features = _features()
    before = tuple(
        part.tobytes() for part in (features.data, features.indices, features.indptr)
    )

    model = NbSvmClassifier(random_state=7).fit(features, [True, True, False])
    model.decision_function(features)

    assert (
        tuple(
            part.tobytes()
            for part in (features.data, features.indices, features.indptr)
        )
        == before
    )


def test_explicit_sparse_zero_cannot_change_ratio_or_scores() -> None:
    """Uložená nula nepřidává výskyt příznaku ani příspěvek k rozhodovacímu skóre."""
    without_zero = _features()
    with_zero = csr_matrix(
        ([2.0, 0.0, 0.5, 3.0, 1.0], [0, 1, 0, 1, 1], [0, 2, 4, 5]),
        shape=(3, 2),
    )
    reference = NbSvmClassifier(random_state=7).fit(without_zero, [True, True, False])
    model = NbSvmClassifier(random_state=7).fit(with_zero, [True, True, False])

    np.testing.assert_allclose(model.r_, reference.r_)
    np.testing.assert_allclose(
        model.decision_function(with_zero), reference.decision_function(without_zero)
    )


def test_integer_counts_match_floating_point_scores() -> None:
    """Celočíselné četnosti a jejich float verze dávají shodná skóre."""
    features = csr_matrix([[2, 0], [1, 3], [0, 1]])
    floating = features.astype(float)
    model = NbSvmClassifier(random_state=7).fit(features, [True, True, False])
    reference = NbSvmClassifier(random_state=7).fit(floating, [True, True, False])

    np.testing.assert_allclose(
        model.decision_function(features), reference.decision_function(floating)
    )


@pytest.mark.parametrize("label", [False, True])
def test_single_class_is_rejected(*, label: bool) -> None:
    """Jediná třída je odmítnutá stráží NB-SVM před výpočtem poměru četností."""
    with pytest.raises(ValueError, match=r"^NB-SVM vyžaduje právě dvě třídy\.$"):
        NbSvmClassifier().fit(_features(), [label] * 3)


def test_clone_exposes_parameters_without_fitted_state() -> None:
    """Tuner může estimator klonovat a budoucí ladění používá jméno C ze sklearn."""
    model = NbSvmClassifier(alpha=2.0, beta=0.5, C=3.0, random_state=7).fit(
        _features(), [True, True, False]
    )
    copied = clone(model)

    assert copied.get_params() == {
        "alpha": 2.0,
        "beta": 0.5,
        "C": 3.0,
        "random_state": 7,
    }
    assert not hasattr(copied, "r_")
    assert not hasattr(copied, "predict_proba")
