"""Percentily nákladů podle ručního výpočtu nejbližšího pořadí."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from mediparse.domain.costs import LatencyPlan, latency, nearest_rank

if TYPE_CHECKING:
    from collections.abc import Sequence


def test_latency_plan_requires_at_least_one_sample() -> None:
    """Plán bez měřených vzorků nemá definovanou latenci."""
    with pytest.raises(ValidationError, match="samples"):
        LatencyPlan(warmup=0, samples=0)


def test_nearest_rank_of_three_unsorted_values() -> None:
    """ceil(1,5) = 2: medián je druhá nejmenší skutečně naměřená hodnota."""
    expected = 0.2
    assert nearest_rank([0.3, 0.1, 0.2], 50) == expected


@pytest.mark.parametrize(
    ("values", "percent", "expected"),
    [(range(1, 101), 7, 7), ([0.1, 0.2], 0.5, 0.1)],
)
def test_nearest_rank_keeps_exact_and_fractional_ranks(
    values: Sequence[float], percent: float, expected: float
) -> None:
    """Sedm procent ze sta má pořadí 7, půl procenta ze dvou má pořadí 1."""
    assert nearest_rank(values, percent) == expected


@pytest.mark.parametrize(
    ("durations", "p50", "p95"),
    [
        ([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], 0.5, 1.0),
        (list(range(1, 21)), 10, 19),
        ([0.7], 0.7, 0.7),
    ],
)
def test_latency_uses_nearest_rank(
    durations: list[float], p50: float, p95: float
) -> None:
    """Ruční pořadí jsou 5 a 10 z deseti, 10 a 19 z dvaceti, 1 z jedné."""
    result = latency(durations)

    assert result.samples == len(durations)
    assert result.p50_seconds == p50
    assert result.p95_seconds == p95


@pytest.mark.parametrize(("values", "percent"), [([], 50), ([0.1], 0), ([0.1], 101)])
def test_undefined_percentile_is_rejected(values: list[float], percent: float) -> None:
    """Prázdný soubor a percentil mimo (0, 100] nemají definované pořadí."""
    with pytest.raises(ValueError, match=r"neprázdný vstup.*percent"):
        nearest_rank(values, percent)


@given(st.lists(st.floats(min_value=0, max_value=1e6), min_size=1, max_size=100))
def test_latency_percentiles_are_ordered_observed_values(
    durations: list[float],
) -> None:
    """Percentil neinterpoluje a vyšší percentil nemůže být menší."""
    result = latency(durations)

    assert result.p50_seconds in durations
    assert result.p95_seconds in durations
    assert result.p50_seconds <= result.p95_seconds


@given(st.lists(st.floats(min_value=0, max_value=1e6), min_size=1, max_size=100))
def test_full_percentile_is_maximum(values: list[float]) -> None:
    """Stý percentil má pořadí n a je největší hodnotou neprázdného vstupu."""
    assert nearest_rank(values, 100) == max(values)
