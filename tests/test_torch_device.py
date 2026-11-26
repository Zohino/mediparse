"""Přiřazení nezávislých foldů výpočetním zařízením."""

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.torch_device import fold_device


def test_fold_device_cycles_over_gpus() -> None:
    """Pět foldů se dvěma GPU má střídavé zařízení."""
    assert [fold_device("cuda", i, 2) for i in range(5)] == [
        "cuda:0",
        "cuda:1",
        "cuda:0",
        "cuda:1",
        "cuda:0",
    ]
    assert fold_device("cpu", 3, 0) == "cpu"
    with pytest.raises(InvalidInputError, match="GPU"):
        fold_device("cuda", 0, 0)
