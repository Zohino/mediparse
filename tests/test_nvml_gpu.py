"""Viditelnost CUDA a měření GPU podle ruční tabulky a podvrženého NVML."""

from __future__ import annotations

from threading import Event
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pynvml
import pytest

from mediparse.domain.costs import GpuUsage
from mediparse.domain.environment import GpuDevice
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure import nvml_gpu
from mediparse.infrastructure.nvml_gpu import gpu_monitor, visible_indices

if TYPE_CHECKING:
    from collections.abc import Callable


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


def _energy_readings(values: dict[int, list[int | Exception]]) -> Callable[[int], int]:
    def energy(handle: int) -> int:
        value = values[handle].pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    return energy


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
        "nvmlDeviceGetUUID": Mock(side_effect=lambda handle: {0: "GPU-a"}[handle]),
        "nvmlDeviceGetName": Mock(side_effect=lambda handle: {0: "Fake GPU"}[handle]),
        "nvmlDeviceGetMemoryInfo": Mock(
            side_effect=lambda handle: {0: SimpleNamespace(total=1000, used=100)}[
                handle
            ]
        ),
        "nvmlDeviceGetTotalEnergyConsumption": Mock(
            side_effect=_energy_readings({0: [1000, 4500]})
        ),
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
    readings = {0: iter((100, 300, 200))}

    def memory(handle: int) -> SimpleNamespace:
        used = next(readings[handle])
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
    nvml.nvmlDeviceGetTotalEnergyConsumption.side_effect = _energy_readings({
        0: [1000, pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN)],
    })
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


@pytest.mark.parametrize(
    "sampling_error",
    [
        pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN),
        ValueError("Synthetic sampler failure"),
    ],
)
def test_sampling_error_is_refused_after_body(
    monkeypatch: pytest.MonkeyPatch, nvml: SimpleNamespace, sampling_error: Exception
) -> None:
    """Chyba prostředního vzorku nesmí zmizet ve vlákně démona."""
    sampled = Event()

    sample_call = 2

    def memory(handle: int) -> SimpleNamespace:
        if nvml.nvmlDeviceGetMemoryInfo.call_count == sample_call:
            sampled.set()
            raise sampling_error
        return {0: SimpleNamespace(total=1000, used=100)}[handle]

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


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        (
            ("GPU-b,GPU-a", None, 0),
            (
                GpuUsage(
                    device=GpuDevice(name="Fake GPU B", memory_total_bytes=2000),
                    energy_joules=7.0,
                    memory_used_start_bytes=400,
                    memory_used_peak_bytes=800,
                ),
                GpuUsage(
                    device=GpuDevice(name="Fake GPU A", memory_total_bytes=1000),
                    energy_joules=3.5,
                    memory_used_start_bytes=100,
                    memory_used_peak_bytes=300,
                ),
            ),
        ),
        (
            ("1", "PCI_BUS_ID", 1),
            (
                GpuUsage(
                    device=GpuDevice(name="Fake GPU B", memory_total_bytes=2000),
                    energy_joules=7.0,
                    memory_used_start_bytes=400,
                    memory_used_peak_bytes=800,
                ),
            ),
        ),
    ],
)
def test_monitor_measures_selected_device_handles(
    monkeypatch: pytest.MonkeyPatch,
    nvml: SimpleNamespace,
    selection: tuple[str, str | None, int],
    expected: tuple[GpuUsage, ...],
) -> None:
    """A: (4500-1000)/1000=3,5 J, max=300 B; B: (9000-2000)/1000=7 J, max=800 B."""
    visible, order, last_sampled_handle = selection
    sampled = Event()
    readings = {0: iter((100, 300, 200)), 1: iter((400, 800, 500))}
    middle_peaks = {0: 300, 1: 800}

    def memory(handle: int) -> SimpleNamespace:
        used = next(readings[handle])
        if handle == last_sampled_handle and used == middle_peaks[handle]:
            sampled.set()
        return SimpleNamespace(total={0: 1000, 1: 2000}[handle], used=used)

    monkeypatch.setattr(nvml_gpu, "Event", _OneSampleEvent)
    nvml.nvmlDeviceGetCount.return_value = 2
    nvml.nvmlDeviceGetUUID.side_effect = lambda handle: {0: "GPU-a", 1: "GPU-b"}[handle]
    nvml.nvmlDeviceGetName.side_effect = lambda handle: {
        0: "Fake GPU A",
        1: "Fake GPU B",
    }[handle]
    nvml.nvmlDeviceGetMemoryInfo.side_effect = memory
    nvml.nvmlDeviceGetTotalEnergyConsumption.side_effect = _energy_readings({
        0: [1000, 4500],
        1: [2000, 9000],
    })
    with gpu_monitor(visible, order) as monitor:
        assert sampled.wait(5), "Vzorkovací vlákno nepřečetlo vybrané karty."
        assert monitor.devices == tuple(usage.device for usage in expected)
    assert monitor.measurement().devices == expected
    selected = {last_sampled_handle} if visible == "1" else {0, 1}
    assert {
        call.args[0] for call in nvml.nvmlDeviceGetMemoryInfo.call_args_list
    } == selected
    assert {
        call.args[0] for call in nvml.nvmlDeviceGetTotalEnergyConsumption.call_args_list
    } == selected


def test_final_read_error_is_refused(nvml: SimpleNamespace) -> None:
    """Selhání čítače na konci nevyrobí neúplné náklady."""
    nvml.nvmlDeviceGetTotalEnergyConsumption.side_effect = _energy_readings({
        0: [1000, pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN)],
    })
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


@pytest.mark.usefixtures("nvml")
def test_measurement_uses_elapsed_window_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Délka okna je ručně 12,5 - 10,0 = 2,5 sekundy."""
    monkeypatch.setattr(nvml_gpu, "perf_counter", iter((10.0, 12.5)).__next__)
    with gpu_monitor(None, None) as monitor:
        pass
    expected_seconds = 2.5
    assert monitor.measurement().seconds == expected_seconds


@pytest.mark.parametrize("phase", ["energy", "shutdown"])
def test_completion_inside_except_refuses_its_own_error(
    nvml: SimpleNamespace, phase: str
) -> None:
    """Obsluhovaná KeyError volajícího není výjimka těla měřicího okna."""
    if phase == "energy":
        nvml.nvmlDeviceGetTotalEnergyConsumption.side_effect = _energy_readings({
            0: [1000, pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN)]
        })
    else:
        nvml.nvmlShutdown.side_effect = pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN)
    empty: dict[str, int] = {}
    try:
        empty.pop("synthetic-missing-key")
    except KeyError:
        with pytest.raises(InvalidInputError), gpu_monitor(None, None):
            pass


@pytest.mark.parametrize(
    "code", [pynvml.NVML_ERROR_LIBRARY_NOT_FOUND, pynvml.NVML_ERROR_DRIVER_NOT_LOADED]
)
def test_missing_driver_does_not_chain_into_body(
    nvml: SimpleNamespace, code: int
) -> None:
    """Zpracovaná chyba ovladače se nepřipojí jako kontext chyby měřeného těla."""
    nvml.nvmlInit.side_effect = pynvml.NVMLError(code)
    error = RuntimeError("Synthetic body failure")
    with pytest.raises(RuntimeError) as caught, gpu_monitor(None, None):
        raise error
    assert caught.value is error
    assert caught.value.__context__ is None
