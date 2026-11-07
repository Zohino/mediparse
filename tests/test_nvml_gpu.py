"""Viditelnost CUDA a měření GPU podle ruční tabulky a podvrženého NVML."""

from __future__ import annotations

from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pynvml
import pytest

from mediparse.domain.costs import GpuUsage
from mediparse.domain.environment import GpuDevice
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure import nvml_gpu
from mediparse.infrastructure.nvml_gpu import gpu_monitor, visible_indices


@pytest.mark.parametrize(
    ("visible", "order", "expected"),
    [
        (None, None, (0, 1)),
        ("", None, ()),
        ("-1", None, ()),
        ("1", "PCI_BUS_ID", (1,)),
        (" 1 , 0 ", "PCI_BUS_ID", (1, 0)),
        ("GPU-b", None, (1,)),
        ("GPU-b,GPU-a", None, (1, 0)),
    ],
)
def test_visible_indices_follow_cuda_order(
    visible: str | None, order: str | None, expected: tuple[int, ...]
) -> None:
    """Ruční tabulka mapuje indexy a přesná UUID v pořadí viditelnosti."""
    assert visible_indices(visible, order, ("GPU-a", "GPU-b")) == expected


@pytest.mark.parametrize(
    ("visible", "order"),
    [
        ("1", None),
        ("2", "PCI_BUS_ID"),
        ("0,0", "PCI_BUS_ID"),
        ("GPU-c", None),
        ("MIG-x", None),
        ("abc", None),
        ("GPU-a,0", "PCI_BUS_ID"),
        ("GPU-a,GPU-a", None),
        ("0,", "PCI_BUS_ID"),
    ],
)
def test_invalid_visibility_is_refused_without_disclosing_value(
    visible: str, order: str | None
) -> None:
    """Neznámé, opakované a nejednoznačné položky se neodhadují."""
    with pytest.raises(InvalidInputError) as caught:
        visible_indices(visible, order, ("GPU-a", "GPU-b"))
    assert visible not in str(caught.value)


def test_single_device_needs_no_bus_order() -> None:
    """U jediné karty se CUDA a NVML v pořadí nemohou lišit."""
    assert visible_indices("0", None, ("GPU-a",)) == (0,)


@pytest.fixture
def nvml(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Jedna syntetická karta s ručně zadaným čítačem energie a pamětí.

    Returns:
        Podvržené funkce NVML pro nastavení scénáře a kontrolu volání.
    """
    functions = {
        "nvmlInit": Mock(),
        "nvmlShutdown": Mock(),
        "nvmlDeviceGetCount": Mock(return_value=1),
        "nvmlDeviceGetHandleByIndex": Mock(side_effect=lambda index: index),
        "nvmlDeviceGetUUID": Mock(return_value="GPU-a"),
        "nvmlDeviceGetName": Mock(return_value="Fake GPU"),
        "nvmlDeviceGetMemoryInfo": Mock(
            return_value=SimpleNamespace(total=1000, used=100)
        ),
        "nvmlDeviceGetTotalEnergyConsumption": Mock(side_effect=[1000, 4500]),
    }
    for name, function in functions.items():
        monkeypatch.setattr(pynvml, name, function)
    return SimpleNamespace(**functions)


class _OneSampleEvent(Event):
    """První čekání pustí vzorek, další počká až na ukončení okna."""

    def __init__(self) -> None:
        super().__init__()
        self.waits = 0

    def wait(self, timeout: float | None = None) -> bool:
        self.waits += 1
        if self.waits == 1 and timeout is not None:
            return False
        return super().wait()


def test_monitor_measures_energy_and_middle_memory_peak(
    monkeypatch: pytest.MonkeyPatch, nvml: SimpleNamespace
) -> None:
    """(4500 - 1000) / 1000 = 3,5 J; max(100, 300, 200) = 300 B."""
    sampled = Event()
    middle_peak = 300
    expected_samples = 3
    readings = iter((100, 300, 200))

    def memory(_handle: object) -> SimpleNamespace:
        used = next(readings)
        if used == middle_peak:
            sampled.set()
        return SimpleNamespace(total=1000, used=used)

    monkeypatch.setattr(nvml_gpu, "Event", _OneSampleEvent)
    nvml.nvmlDeviceGetMemoryInfo.side_effect = memory
    with gpu_monitor(None, None) as monitor:
        assert sampled.wait(5), "Vzorkovací vlákno nepřečetlo paměť."
    assert monitor.measurement().devices == (
        GpuUsage(
            device=GpuDevice(name="Fake GPU", memory_total_bytes=1000),
            energy_joules=3.5,
            memory_used_start_bytes=100,
            memory_used_peak_bytes=300,
        ),
    )
    assert nvml.nvmlDeviceGetMemoryInfo.call_count == expected_samples
    nvml.nvmlShutdown.assert_called_once_with()


@pytest.mark.parametrize(
    "error",
    [
        pynvml.NVMLError(pynvml.NVML_ERROR_LIBRARY_NOT_FOUND),
        pynvml.NVMLError(pynvml.NVML_ERROR_DRIVER_NOT_LOADED),
    ],
)
def test_missing_driver_has_no_devices(nvml: SimpleNamespace, error: Exception) -> None:
    """Chybějící knihovna i nenačtený ovladač dovolí běh bez GPU."""
    nvml.nvmlInit.side_effect = error
    with gpu_monitor(None, None) as monitor:
        assert monitor.devices == ()
    assert monitor.measurement().devices == ()
    nvml.nvmlShutdown.assert_not_called()


@pytest.mark.parametrize("visible", ["", "-1"])
def test_hidden_gpu_shuts_down_without_sampling(
    nvml: SimpleNamespace, visible: str
) -> None:
    """Skrytá karta není měřena, dostupná knihovna se korektně ukončí."""
    with gpu_monitor(visible, None) as monitor:
        assert monitor.devices == ()
    assert monitor.measurement().devices == ()
    nvml.nvmlDeviceGetMemoryInfo.assert_not_called()
    nvml.nvmlShutdown.assert_called_once_with()


def test_unsupported_energy_preserves_memory(nvml: SimpleNamespace) -> None:
    """Karta bez čítače stále nese měření paměti."""
    nvml.nvmlDeviceGetTotalEnergyConsumption.side_effect = pynvml.NVMLError(
        pynvml.NVML_ERROR_NOT_SUPPORTED
    )
    with gpu_monitor(None, None) as monitor:
        pass
    (usage,) = monitor.measurement().devices
    assert usage.energy_joules is None
    expected_memory = 100
    assert usage.memory_used_start_bytes == expected_memory
    assert usage.memory_used_peak_bytes == expected_memory


def test_other_init_error_is_refused(nvml: SimpleNamespace) -> None:
    """Chyba oprávnění není neexistující GPU a nesmí se tiše přeskočit."""
    nvml.nvmlInit.side_effect = pynvml.NVMLError(pynvml.NVML_ERROR_NO_PERMISSION)
    with pytest.raises(InvalidInputError), gpu_monitor(None, None):
        pytest.fail("Tělo po chybě startu nesmí běžet.")


def test_start_read_error_shuts_down_before_body(nvml: SimpleNamespace) -> None:
    """Selhání prvního čtení odmítne krok a uvolní NVML."""
    nvml.nvmlDeviceGetMemoryInfo.side_effect = pynvml.NVMLError(
        pynvml.NVML_ERROR_UNKNOWN
    )
    with pytest.raises(InvalidInputError), gpu_monitor(None, None):
        pytest.fail("Tělo po chybě prvního čtení nesmí běžet.")
    nvml.nvmlShutdown.assert_called_once_with()


def test_body_exception_is_preserved_and_nvml_closed(nvml: SimpleNamespace) -> None:
    """Výjimka tréninku nebo inference projde ven jako stejný objekt."""
    error = RuntimeError("Synthetic body failure")
    with pytest.raises(RuntimeError) as caught, gpu_monitor(None, None):
        raise error
    assert caught.value is error
    nvml.nvmlShutdown.assert_called_once_with()


def test_cleanup_error_does_not_mask_body(nvml: SimpleNamespace) -> None:
    """Ani chyba posledního čtení a shutdownu nezakryje původní příčinu."""
    error = RuntimeError("Synthetic body failure")
    nvml.nvmlDeviceGetTotalEnergyConsumption.side_effect = [
        1000,
        pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN),
    ]
    nvml.nvmlShutdown.side_effect = pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN)
    with pytest.raises(RuntimeError) as caught, gpu_monitor(None, None):
        raise error
    assert caught.value is error
    nvml.nvmlShutdown.assert_called_once_with()


@pytest.mark.usefixtures("nvml")
def test_measurement_is_unavailable_inside_window() -> None:
    """Výsledek je úplný až po uzavření měřicího okna."""
    with gpu_monitor(None, None) as monitor, pytest.raises(RuntimeError):
        monitor.measurement()


def test_sampling_error_is_refused_after_body(
    monkeypatch: pytest.MonkeyPatch, nvml: SimpleNamespace
) -> None:
    """Chyba prostředního vzorku nesmí zmizet ve vlákně démona."""
    sampled = Event()

    sample_call = 2

    def memory(_handle: object) -> SimpleNamespace:
        if nvml.nvmlDeviceGetMemoryInfo.call_count == sample_call:
            sampled.set()
            raise pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN)
        return SimpleNamespace(total=1000, used=100)

    monkeypatch.setattr(nvml_gpu, "Event", _OneSampleEvent)
    nvml.nvmlDeviceGetMemoryInfo.side_effect = memory
    completed = Event()

    def measured_step() -> None:
        with gpu_monitor(None, None):
            assert sampled.wait(5), "Vzorkovací vlákno nepřečetlo paměť."
            completed.set()

    with pytest.raises(InvalidInputError):
        measured_step()
    assert completed.is_set()
    nvml.nvmlShutdown.assert_called_once_with()


def test_real_machine_has_consistent_measurement() -> None:
    """Skutečný stroj s GPU i bez něj splní invarianty bez pevných měřených hodnot."""
    with gpu_monitor(None, None) as monitor:
        devices = monitor.devices
    measurement = monitor.measurement()
    assert tuple(usage.device for usage in measurement.devices) == devices
    assert measurement.seconds >= 0
    assert all(
        usage.memory_used_peak_bytes >= usage.memory_used_start_bytes
        for usage in measurement.devices
    )


def test_monitor_preserves_uuid_selection_order(nvml: SimpleNamespace) -> None:
    """Dvě karty se měří podle UUID v pořadí CUDA, bez cizích zařízení."""
    nvml.nvmlDeviceGetCount.return_value = 2
    nvml.nvmlDeviceGetUUID.side_effect = lambda index: ("GPU-a", "GPU-b")[index]
    nvml.nvmlDeviceGetName.side_effect = lambda index: ("Fake GPU A", "Fake GPU B")[
        index
    ]
    nvml.nvmlDeviceGetTotalEnergyConsumption.side_effect = [1000, 2000, 4500, 7000]
    with gpu_monitor("GPU-b,GPU-a", None) as monitor:
        assert tuple(device.name for device in monitor.devices) == (
            "Fake GPU B",
            "Fake GPU A",
        )
    assert tuple(usage.energy_joules for usage in monitor.measurement().devices) == (
        3.5,
        5.0,
    )


def test_final_read_error_is_refused(nvml: SimpleNamespace) -> None:
    """Selhání čítače na konci nevyrobí neúplné náklady."""
    nvml.nvmlDeviceGetTotalEnergyConsumption.side_effect = [
        1000,
        pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN),
    ]
    with pytest.raises(InvalidInputError), gpu_monitor(None, None):
        pass
    nvml.nvmlShutdown.assert_called_once_with()


def test_shutdown_error_is_refused(nvml: SimpleNamespace) -> None:
    """Chyba při ukončení dostupného NVML se nesmí vydávat za úspěch."""
    nvml.nvmlShutdown.side_effect = pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN)
    with pytest.raises(InvalidInputError), gpu_monitor(None, None):
        pass


@pytest.mark.parametrize("interval", [0.0, -0.1, float("inf"), float("nan")])
def test_invalid_sampling_interval_is_rejected(interval: float) -> None:
    """Chybný interval programátora nemůže spustit nekonečnou smyčku vzorkování."""
    with (
        pytest.raises(ValueError, match="Interval"),
        gpu_monitor(None, None, interval_seconds=interval),
    ):
        pass
