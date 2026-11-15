"""Předem schválené pravidlo non-inferiority a cenového srovnání."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from mediparse.domain.decision_rule import DecisionRule
from mediparse.domain.inputs import InvalidInputError


@pytest.fixture
def rule() -> DecisionRule:
    """Načte preregistrované pravidlo přímo z configu repozitáře.

    Returns:
        Validované pravidlo používané akceptačními příklady.
    """
    path = Path(__file__).parents[1] / "config/decision_rule.json"
    return DecisionRule.model_validate_json(path.read_text(encoding="utf-8"))


def test_repository_config_matches_protocol(rule: DecisionRule) -> None:
    """Soubor obsahuje přesně hodnoty schválené před evaluací."""
    assert rule.model_dump() == {
        "margin": 0.02,
        "min_latency_ratio": 10.0,
        "min_training_cost_ratio": 10.0,
        "confidence": 0.95,
        "bootstrap_resamples": 2000,
        "min_evaluation_notes": 25000,
        "seed": 0,
    }


@pytest.mark.parametrize(
    ("ci_low", "ci_high", "notes", "expected"),
    [
        (-0.015, -0.005, 276000, "non_inferior"),
        (0.001, 0.004, 55000, "non_inferior"),
        (-0.035, -0.025, 276000, "inferior"),
        (-0.025, -0.010, 276000, "inconclusive"),
        (-0.020, -0.010, 276000, "inconclusive"),
        (-0.030, -0.020, 276000, "inconclusive"),
        (-0.015, -0.005, 25000, "non_inferior"),
        (0.0, 0.0, 276000, "non_inferior"),
        (-1.0, 1.0, 276000, "inconclusive"),
    ],
)
def test_verdict(
    rule: DecisionRule, ci_low: float, ci_high: float, notes: int, expected: str
) -> None:
    """Meze intervalu určují tři výsledky s ostrou hranicí minus delta."""
    assert rule.verdict(ci_low, ci_high, notes) == expected


@pytest.mark.parametrize(
    ("ci_low", "ci_high", "notes", "message"),
    [
        (-0.010, -0.015, 276000, "meze"),
        (-0.015, -0.005, 24999, "zpráv"),
        (math.nan, -0.01, 276000, "meze"),
        (-0.03, math.nan, 276000, "meze"),
        (math.inf, math.inf, 276000, "meze"),
        (-math.inf, -0.01, 276000, "meze"),
        (-1.01, -0.01, 276000, "meze"),
        (-0.01, 1.01, 276000, "meze"),
    ],
)
def test_invalid_evaluation_is_rejected(
    rule: DecisionRule, ci_low: float, ci_high: float, notes: int, message: str
) -> None:
    """Neplatné meze a evaluace pod minimem odmítnou výpočet verdiktu."""
    with pytest.raises(InvalidInputError, match=message):
        rule.verdict(ci_low, ci_high, notes)


@pytest.mark.parametrize(
    ("latency_ratio", "training_cost_ratio", "expected"),
    [(12.0, 1.0, True), (1.0, 10.0, True), (9.9, 9.9, False)],
)
def test_is_cheaper(
    rule: DecisionRule,
    latency_ratio: float,
    training_cost_ratio: float,
    *,
    expected: bool,
) -> None:
    """Stačí dosáhnout alespoň jednoho prahu poměru transformer ku klasice."""
    assert rule.is_cheaper(latency_ratio, training_cost_ratio) is expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("margin", 0),
        ("margin", 1),
        ("min_latency_ratio", 0.5),
        ("min_training_cost_ratio", 0.5),
        ("confidence", 0),
        ("confidence", 1),
        ("bootstrap_resamples", 999),
        ("min_evaluation_notes", 0),
        ("seed", -1),
        ("unknown", 0),
        ("margin", "0.02"),
    ],
)
def test_invalid_config_is_rejected(
    rule: DecisionRule, field: str, value: object
) -> None:
    """Config odmítne neplatné meze, neznámé pole i převod řetězce na číslo."""
    with pytest.raises(ValidationError):
        DecisionRule.model_validate_json(json.dumps(rule.model_dump() | {field: value}))
