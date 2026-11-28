"""Přiřazení nezávislých foldů výpočetním zařízením."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure import torch_device
from mediparse.infrastructure.torch_device import DevicePool, fold_device, fold_groups


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


def test_device_pool_refuses_insufficient_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Čtyři pokusy se třemi GPU jsou odmítnuty před tréninkem."""
    monkeypatch.setattr(torch_device, "cuda_device_count", lambda: 3)
    with pytest.raises(InvalidInputError, match="GPU"):
        DevicePool.of("cuda", 4)


def test_device_pool_leases_exclusively(monkeypatch: pytest.MonkeyPatch) -> None:
    """Třetí vlákno čeká, dokud se jedna ze dvou GPU nevrátí."""
    monkeypatch.setattr(torch_device, "cuda_device_count", lambda: 2)
    pool = DevicePool.of("cuda", 2)
    waiting, acquired = Event(), Event()

    def third() -> str:
        waiting.set()
        with pool.lease() as device:
            acquired.set()
            return device

    with ThreadPoolExecutor(max_workers=1) as executor:
        with pool.lease() as first:
            with pool.lease() as second:
                assert {first, second} == {"cuda:0", "cuda:1"}
                future = executor.submit(third)
                assert waiting.wait(10)
                assert not acquired.wait(0.1)
            assert future.result(timeout=10) == second
        assert pool.free.qsize() == 2


def test_device_pool_returns_lease_after_error() -> None:
    """Výjimka neodebere zařízení dalším pokusům; CPU má dvě místa."""
    pool = DevicePool.of("cpu", 2)
    with pool.lease() as first, pool.lease() as second:
        assert first == second == "cpu"

    def fail() -> None:
        """Vyvolá chybu během zápůjčky.

        Raises:
            RuntimeError: Syntetické selhání pokusu.
        """
        with pool.lease():
            raise RuntimeError

    with pytest.raises(RuntimeError):
        fail()
    assert pool.free.qsize() == 2
