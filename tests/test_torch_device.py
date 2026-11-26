"""Přiřazení nezávislých foldů výpočetním zařízením."""

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.torch_device import fold_device, fold_groups


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


def test_fold_groups_keep_each_gpu_exclusive() -> None:
    """Každá GPU dostane jediný úkol s cyklicky přiřazenými foldy."""
    jobs, groups = fold_groups("cuda", 5, 3, 2)
    assert jobs == 2
    assert groups == (
        ((0, "cuda:0"), (2, "cuda:0"), (4, "cuda:0")),
        ((1, "cuda:1"), (3, "cuda:1")),
    )


def test_fold_groups_schedule_cpu_folds_individually() -> None:
    """CPU nemá vazbu na zařízení a každý fold je samostatný úkol."""
    jobs, groups = fold_groups("cpu", 3, 2, 0)
    assert jobs == 2
    assert groups == (((0, "cpu"),), ((1, "cpu"),), ((2, "cpu"),))


def test_fold_groups_refuse_missing_gpu() -> None:
    """CUDA bez viditelné GPU nevytvoří rozvrh."""
    with pytest.raises(InvalidInputError, match="GPU"):
        fold_groups("cuda", 1, 1, 0)
