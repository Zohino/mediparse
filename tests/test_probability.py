"""Součet pravděpodobností rozdělení."""

import pytest

from mediparse.domain.probability import sums_to_one


@pytest.mark.parametrize("values", [[0.5, 0.5], [0.1] * 10])
def test_sum_within_tolerance_is_one(values: list[float]) -> None:
    """Součet 1 projde i s odchylkou plovoucí čárky."""
    assert sums_to_one(values)


@pytest.mark.parametrize("values", [[0.5, 0.4], []], ids=["short", "empty"])
def test_other_sum_is_not_one(values: list[float]) -> None:
    """Jiný součet včetně prázdného vstupu (0) neprojde."""
    assert not sums_to_one(values)
