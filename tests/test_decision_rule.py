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
        "min_inference_energy_ratio": 10.0,
        "confidence": 0.95,
        "bootstrap_resamples": 2000,
        "min_evaluation_notes": 25000,
        "seed": 0,
        "seed_runs": 3,
        "seed_sensitivity_sd_multiple": 2.0,
        "latency": {"warmup": 10, "samples": 1000},
        "reference_cpu": "Intel(R) Core(TM) Ultra 9 185H",
        "reference_gpu": "NVIDIA GeForce RTX 4050 Laptop GPU",
        "cpu_watts_per_core": 2.8125,
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
    ("latency_ratio", "energy_ratio", "expected"),
    [(12.0, 1.0, True), (1.0, 10.0, True), (10.0, 1.0, True), (9.9, 9.9, False)],
)
def test_is_cheaper(
    rule: DecisionRule,
    latency_ratio: float,
    energy_ratio: float,
    *,
    expected: bool,
) -> None:
    """Stačí dosáhnout alespoň jednoho prahu poměru transformer ku klasice."""
    assert rule.is_cheaper(latency_ratio, energy_ratio) is expected


@pytest.mark.parametrize(
    ("latency_ratio", "energy_ratio"),
    [(math.nan, 1.0), (1.0, math.inf), (-1.0, 1.0), (1.0, -1.0)],
)
def test_is_cheaper_rejects_invalid_ratios(
    rule: DecisionRule, latency_ratio: float, energy_ratio: float
) -> None:
    """Neplatný poměr nákladů se odmítne, ne tiše porovná."""
    with pytest.raises(InvalidInputError, match="poměr"):
        rule.is_cheaper(latency_ratio, energy_ratio)


def test_rule_is_frozen(rule: DecisionRule) -> None:
    """Preregistrované pravidlo nelze po načtení měnit."""
    with pytest.raises(ValidationError):
        rule.margin = 0.05  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bootstrap_resamples", 1000),
        ("seed_runs", 2),
        ("min_latency_ratio", 1.0),
        ("min_inference_energy_ratio", 1.0),
    ],
)
def test_config_boundary_is_accepted(
    rule: DecisionRule, field: str, value: float
) -> None:
    """Schéma přijímá inkluzivní dolní hranice preregistrovaných parametrů."""
    assert (
        DecisionRule.model_validate_json(
            json.dumps(rule.model_dump() | {field: value})
        ).model_dump()[field]
        == value
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("margin", 0),
        ("margin", 1),
        ("min_latency_ratio", 0.5),
        ("min_inference_energy_ratio", 0.5),
        ("confidence", 0),
        ("confidence", 1),
        ("bootstrap_resamples", 999),
        ("min_evaluation_notes", 0),
        ("seed", -1),
        ("seed", 2**32),
        ("unknown", 0),
        ("margin", "0.02"),
        ("seed_runs", 1),
        ("seed_sensitivity_sd_multiple", 0),
        ("cpu_watts_per_core", 0),
        ("reference_cpu", ""),
        ("reference_gpu", ""),
        ("latency", {"warmup": 10, "samples": 0}),
        ("min_training_cost_ratio", 10.0),
    ],
)
def test_invalid_config_is_rejected(
    rule: DecisionRule, field: str, value: object
) -> None:
    """Config odmítne neplatné meze, neznámé pole i převod řetězce na číslo."""
    with pytest.raises(ValidationError):
        DecisionRule.model_validate_json(json.dumps(rule.model_dump() | {field: value}))


@pytest.mark.parametrize(
    ("difference", "seed_sd", "expected"),
    [
        (-0.015, 0.003, True),
        (-0.015, 0.002, False),
        (-0.014, 0.003, False),
        (-0.030, 0.006, True),
        (0.0, 0.0, False),
        (-1.0, 0.0, False),
        (1.0, 0.0, False),
    ],
)
def test_is_seed_sensitive(
    rule: DecisionRule, difference: float, seed_sd: float, *, expected: bool
) -> None:
    """Vzdálenost od hranice se ostře porovná s násobkem SD přes seedy."""
    assert rule.is_seed_sensitive(difference, seed_sd) is expected


@pytest.mark.parametrize(
    ("difference", "seed_sd", "message"),
    [
        (0.0, -0.001, "konečné a nezáporné"),
        (0.0, math.nan, "konečné a nezáporné"),
        (1.5, 0.003, "Rozdíl macro-F1"),
        (-1.5, 0.003, "Rozdíl macro-F1"),
    ],
)
def test_seed_sensitivity_rejects_invalid_inputs(
    rule: DecisionRule, difference: float, seed_sd: float, message: str
) -> None:
    """Neplatný rozdíl nebo SD nesmí dát značku citlivosti na seed."""
    with pytest.raises(InvalidInputError, match=message):
        rule.is_seed_sensitive(difference, seed_sd)


@pytest.mark.parametrize(
    ("cpu_seconds", "gpu_joules", "expected"),
    [(100.0, 0.0, 281.25), (100.0, 500.0, 781.25), (0.0, 0.0, 0.0)],
)
def test_inference_energy_joules(
    rule: DecisionRule, cpu_seconds: float, gpu_joules: float, expected: float
) -> None:
    """CPU-sekundy krát výkon na jádro a NVML dávají ručně spočtenou energii."""
    assert rule.inference_energy_joules(cpu_seconds, gpu_joules) == expected


@pytest.mark.parametrize(
    ("cpu_seconds", "gpu_joules"),
    [(-1.0, 0.0), (math.inf, 0.0), (0.0, -1.0), (0.0, math.nan)],
)
def test_inference_energy_rejects_invalid_inputs(
    rule: DecisionRule, cpu_seconds: float, gpu_joules: float
) -> None:
    """Energie inference odmítá záporné a nekonečné vstupy i NaN."""
    with pytest.raises(InvalidInputError):
        rule.inference_energy_joules(cpu_seconds, gpu_joules)


@pytest.mark.parametrize(
    ("transformer_f1", "baseline_f1", "expected"),
    [
        (0.80, 0.70, True),
        (0.70, 0.70, False),
        (0.60, 0.70, False),
        (1.0, 0.0, True),
        (0.0, 1.0, False),
    ],
)
def test_is_credible(
    rule: DecisionRule, transformer_f1: float, baseline_f1: float, *, expected: bool
) -> None:
    """Transformer musí ostře překonat neladěný TF-IDF + LR na téže sadě."""
    assert rule.is_credible(transformer_f1, baseline_f1) is expected


@pytest.mark.parametrize(
    ("transformer_f1", "baseline_f1"),
    [(1.2, 0.7), (math.nan, 0.7), (0.8, math.nan), (0.8, -0.1), (0.8, 1.2)],
)
def test_credibility_rejects_invalid_scores(
    rule: DecisionRule, transformer_f1: float, baseline_f1: float
) -> None:
    """Obě macro-F1 musí být konečná čísla v intervalu [0, 1]."""
    with pytest.raises(InvalidInputError, match="Macro-F1"):
        rule.is_credible(transformer_f1, baseline_f1)


@pytest.mark.parametrize(
    ("ci", "latency_ratio", "energy_ratio", "expected"),
    [
        ((-0.015, -0.005), 2.0, 1.0, True),
        ((-0.015, -0.005), 1.0, 1.0, False),
        ((-0.015, -0.005), 2.0, 0.9, False),
        ((-0.025, -0.010), 5.0, 5.0, False),
        ((-0.035, -0.025), 5.0, 5.0, False),
    ],
)
def test_dominates(
    rule: DecisionRule,
    ci: tuple[float, float],
    latency_ratio: float,
    energy_ratio: float,
    *,
    expected: bool,
) -> None:
    """NI a náklady nedražší v obou osách s jednou ostrou úsporou určují dominanci."""
    assert rule.dominates(*ci, 276000, latency_ratio, energy_ratio) is expected


@pytest.mark.parametrize(
    ("latency_ratio", "energy_ratio"),
    [(math.nan, 1.0), (1.0, math.nan), (-1.0, 1.0), (1.0, math.inf)],
)
def test_dominance_rejects_invalid_ratios(
    rule: DecisionRule, latency_ratio: float, energy_ratio: float
) -> None:
    """Dominance odmítá neplatné poměry stejným kontraktem jako cenový verdikt."""
    with pytest.raises(InvalidInputError):
        rule.dominates(-0.015, -0.005, 276000, latency_ratio, energy_ratio)
