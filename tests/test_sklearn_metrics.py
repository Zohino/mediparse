"""Metriky z knihovního výpočtu: známé hodnoty a nedefinované případy."""

from __future__ import annotations

import pytest

from mediparse.infrastructure.sklearn_metrics import SklearnScorer


def test_scorer_on_known_example() -> None:
    """Dvě ze tří pozitivních, jeden falešný poplach."""
    truth = [True, True, True, False, False]
    predicted = [True, True, False, True, False]
    scores = [2.0, 1.0, -0.5, 0.5, -1.0]

    metrics = SklearnScorer().score(truth, predicted, scores)

    assert metrics.precision == pytest.approx(2 / 3)
    assert metrics.recall == pytest.approx(2 / 3)
    assert metrics.f1 == pytest.approx(2 / 3)
    assert metrics.accuracy == pytest.approx(3 / 5)


def test_scorer_auc_on_hand_computable_example() -> None:
    """Ze čtyř párů pozitivní a negativní jsou tři seřazené správně."""
    truth = [True, True, False, False]
    scores = [0.9, 0.4, 0.5, 0.1]

    metrics = SklearnScorer().score(truth, [True, False, False, False], scores)

    assert metrics.roc_auc == pytest.approx(3 / 4)
    assert metrics.pr_auc == pytest.approx(0.5 * 1.0 + 0.5 * 2 / 3)


def test_scorer_without_predicted_positives_is_zero() -> None:
    """Bez predikovaných pozitivních je přesnost 0.0 a bez varování."""
    metrics = SklearnScorer().score([True, False], [False, False], [-1.0, -2.0])

    assert (metrics.precision, metrics.recall, metrics.f1) == (0.0, 0.0, 0.0)
    assert metrics.accuracy == pytest.approx(0.5)
