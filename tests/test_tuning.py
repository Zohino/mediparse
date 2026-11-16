"""Nezávislá orákula metriky a validace obecného prostoru ladění."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.tuning import (
    ParamValue,
    SearchSpace,
    ValidationScores,
    macro_max_f1,
    max_f1,
)
from mediparse.infrastructure.tuning_config import load_tuning_config
from tests.screening_support import note
from tests.support import REPOSITORY

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("labels", "scores", "expected"),
    [
        (
            [True, True, False, True, False, False],
            [0.9, 0.8, 0.7, 0.6, 0.2, 0.1],
            6 / 7,
        ),
        ([True, False], [0.5, 0.5], 2 / 3),
    ],
)
def test_max_f1_matches_manual_thresholds(
    labels: list[bool], scores: list[float], expected: float
) -> None:
    """Ruční maxima 6/7 a 2/3 nesmějí rozdělit shodné skóre na dva prahy."""
    assert max_f1(labels, scores) == pytest.approx(expected)


@given(st.lists(st.tuples(st.booleans(), st.integers(-5, 5)), min_size=1, max_size=30))
def test_max_f1_matches_brute_force(rows: list[tuple[bool, int]]) -> None:
    """Hrubá síla počítá 2 TP / (2 TP + FP + FN) přímo z každého prahu."""
    labels = [True, *[label for label, _ in rows]]
    scores = [0.0, *[float(score) for _, score in rows]]
    candidates = []
    for threshold in [float("inf"), *scores]:
        predicted = [score >= threshold for score in scores]
        true_positive = sum(
            label and prediction
            for label, prediction in zip(labels, predicted, strict=True)
        )
        false_positive = sum(
            not label and prediction
            for label, prediction in zip(labels, predicted, strict=True)
        )
        false_negative = sum(
            label and not prediction
            for label, prediction in zip(labels, predicted, strict=True)
        )
        candidates.append(
            2 * true_positive / (2 * true_positive + false_positive + false_negative)
        )
    assert max_f1(labels, scores) == pytest.approx(max(candidates))


@pytest.mark.parametrize(
    ("labels", "scores"),
    [
        ([False], [0.0]),
        ([True], []),
        ([True], [float("nan")]),
        ([True], [float("inf")]),
    ],
)
def test_max_f1_rejects_invalid_scores(labels: list[bool], scores: list[float]) -> None:
    """Bez pozitivních, konečných skóre a shodných délek cíl nevznikne."""
    with pytest.raises(InvalidInputError):
        max_f1(labels, scores)


def test_macro_f1_averages_five_diagnoses() -> None:
    """Čtyři diagnózy mají F1=1, CKD při konstantním skóre F1=2/3: průměr 14/15."""
    notes = (note(1, 0, positive=True), note(2, 0, positive=False))
    scores = dict.fromkeys(Diagnosis, (1.0, 0.0))
    scores[Diagnosis.CKD] = (0.5, 0.5)
    assert macro_max_f1(
        notes, ValidationScores(scores, converged=True)
    ) == pytest.approx(14 / 15)


def test_macro_f1_refuses_missing_diagnosis() -> None:
    """Chybějící diagnóza nesmí potichu změnit jmenovatel macro metriky."""
    with pytest.raises(InvalidInputError, match="pět diagnóz"):
        macro_max_f1((), ValidationScores({}, converged=True))


@pytest.mark.parametrize(
    "params",
    [
        [{"kind": "float", "name": "c", "low": 1.0, "high": 1.0, "log": False}],
        [{"kind": "float", "name": "c", "low": 2.0, "high": 1.0, "log": False}],
        [{"kind": "float", "name": "c", "low": 0.0, "high": 1.0, "log": True}],
        [{"kind": "int", "name": "c", "low": 2, "high": 1, "log": False}],
        [{"kind": "int", "name": "c", "low": 0, "high": 1, "log": True}],
        [{"kind": "categorical", "name": "a", "choices": [None, None]}],
        [{"kind": "categorical", "name": "a", "choices": []}],
        [{"kind": "unknown", "name": "a"}],
        [{"kind": "categorical", "name": "a", "choices": [True], "extra": 0}],
        [{"kind": "int", "name": "c", "low": "1", "high": 2, "log": False}],
        [{"kind": "int", "name": "c", "low": True, "high": 2, "log": False}],
        [{"kind": "categorical", "name": "a", "choices": [None]}] * 2,
        [],
    ],
)
def test_space_rejects_invalid_json(params: list[dict[str, object]]) -> None:
    """Neplatné intervaly, duplicity, cizí klíč i typ se odmítnou na hranici JSON."""
    with pytest.raises(ValidationError):
        SearchSpace.model_validate_json(json.dumps({"params": params}))


def test_repository_tuning_config_and_loader(tmp_path: Path) -> None:
    """Config z repa má úplný prostor a loader převádí chyby na doménové odmítnutí."""
    config = load_tuning_config(REPOSITORY / "config/tuning.json")
    assert config.row_id == "en-svm"
    assert config.space.names == {
        "analyzer",
        "ngram_max",
        "c",
        "class_weight",
        "min_df",
        "max_df",
        "sublinear_tf",
    }
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"row_id": "wrong"}', encoding="utf-8")
    with pytest.raises(InvalidInputError, match=r"invalid\.json"):
        load_tuning_config(invalid)


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"c": 1.0, "n": 2, "flag": True}, True),
        ({"c": 0.1, "n": 2, "flag": True}, True),
        ({"c": 0.5, "n": 1, "flag": True}, True),
        ({"c": 0.5, "n": 3, "flag": True}, True),
        ({"c": 0.0, "n": 2, "flag": True}, False),
        ({"c": 2.0, "n": 2, "flag": True}, False),
        ({"c": 0.5, "n": True, "flag": True}, False),
        ({"c": True, "n": 2, "flag": True}, False),
        ({"c": float("nan"), "n": 2, "flag": True}, False),
        ({"c": 0.5, "n": 2, "flag": 1}, False),
        ({"c": 0.5, "n": 4, "flag": True}, False),
        ({"c": 0.5, "n": 2}, False),
        ({"c": 0.5, "n": 2, "flag": True, "extra": 1}, False),
    ],
)
def test_space_membership_is_complete_and_typed(
    params: dict[str, ParamValue], *, expected: bool
) -> None:
    """Příslušnost rozlišuje bool, int i kategorii a vyžaduje přesnou sadu jmen."""
    space = SearchSpace.model_validate_json(
        json.dumps({
            "params": [
                {"kind": "float", "name": "c", "low": 0.1, "high": 1.0, "log": True},
                {"kind": "int", "name": "n", "low": 1, "high": 3, "log": False},
                {"kind": "categorical", "name": "flag", "choices": [True, False]},
            ]
        })
    )
    assert space.contains(params) is expected


def test_tuning_config_refuses_single_inner_fold(tmp_path: Path) -> None:
    """Vnitřní CV potřebuje alespoň dva foldy, jeden musí odmítnout loader."""
    config = load_tuning_config(REPOSITORY / "config/tuning.json")
    invalid = tmp_path / "invalid.json"
    invalid.write_text(
        json.dumps(config.model_dump(mode="json") | {"inner_folds": 1}),
        encoding="utf-8",
    )
    with pytest.raises(InvalidInputError, match="inner_folds"):
        load_tuning_config(invalid)
