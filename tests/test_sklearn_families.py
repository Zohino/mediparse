"""Parametry a determinismus továren rodin klasifikátorů."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mediparse.domain.representation import ClassifierFamily
from mediparse.infrastructure.fasttext_classifier import FastTextClassifier
from mediparse.infrastructure.sklearn_families import classifier_estimator

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "family", [ClassifierFamily.LINEAR_SVM, ClassifierFamily.NB_SVM]
)
def test_svm_factory_preserves_seed(tmp_path: Path, family: ClassifierFamily) -> None:
    """Obě rodiny s duálním řešičem dostanou seed ladění."""
    assert classifier_estimator(family, 13, tmp_path).get_params()["random_state"] == 13


def test_lightgbm_factory_preserves_determinism_parameters(
    tmp_path: Path,
) -> None:
    """Veřejné parametry LightGBM drží seed a determinismus; LSA provádí scorer stupně."""
    estimator = classifier_estimator(ClassifierFamily.LIGHTGBM_LSA, 13, tmp_path)
    params = estimator.get_params()

    expected = {
        "random_state": 13,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }
    assert {key: params[key] for key in expected} == expected


def test_fasttext_factory_preserves_seed_and_scratch(tmp_path: Path) -> None:
    """Továrna předá fastTextu seed i chráněný adresář."""
    estimator = classifier_estimator(ClassifierFamily.FASTTEXT, 11, tmp_path)
    assert isinstance(estimator, FastTextClassifier)
    assert estimator.random_state == 11
    assert estimator.scratch == tmp_path
