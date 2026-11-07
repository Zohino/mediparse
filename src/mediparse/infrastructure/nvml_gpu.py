"""Měření energie z čítače NVML a vzorkování paměti viditelných GPU."""

from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from math import isfinite
from threading import Event, Thread
from time import perf_counter
from typing import TYPE_CHECKING, Final

import pynvml
from pydantic import ValidationError

from mediparse.domain.costs import GpuMeasurement, GpuUsage
from mediparse.domain.environment import GpuDevice
from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

logger = logging.getLogger(__name__)
MILLIJOULES_PER_JOULE: Final = 1000
NVML_FAILURE: Final = "Měření GPU přes NVML selhalo"


def visible_indices(
    cuda_visible_devices: str | None,
    cuda_device_order: str | None,
    uuids: Sequence[str],
    pci_bus_ids: Sequence[str],
) -> tuple[int, ...]:
    """Přiřadí viditelnost CUDA k indexům NVML bez odhadu jiné karty.

    Args:
        cuda_visible_devices: Výběr zařízení CUDA nebo None pro všechna zařízení.
        cuda_device_order: Pořadí zařízení CUDA; číselný výběr více karet vyžaduje PCI_BUS_ID.
        uuids: UUID zařízení v pořadí výčtu NVML.
        pci_bus_ids: PCI adresy stejných zařízení ve stejném pořadí NVML.

    Returns:
        Indexy NVML v pořadí viditelnosti.

    Raises:
        InvalidInputError: Položka je neznámá, opakovaná nebo pořadí nejednoznačné.
    """
    if cuda_visible_devices is None:
        return tuple(range(len(uuids)))
    if cuda_visible_devices.strip() in {"", "-1"}:
        return ()
    pci_order = sorted(range(len(uuids)), key=lambda index: pci_bus_ids[index].lower())
    indices: list[int] = []
    # NVML ignoruje CUDA_VISIBLE_DEVICES, proto viditelnost mapujeme sami.
    # CUDA přijímá prefix UUID a při neplatné položce seznam zkrátí; přesná UUID
    # a odmítnutí celého seznamu brání tomu, aby překlep tiše zúžil měření.
    for item in cuda_visible_devices.split(","):
        token = item.strip()
        if token.isascii() and token.isdecimal():
            if len(uuids) > 1 and cuda_device_order != "PCI_BUS_ID":
                msg = "Číselný výběr GPU na stroji s více kartami vyžaduje CUDA_DEVICE_ORDER=PCI_BUS_ID."
                raise InvalidInputError(msg)
            position = int(token)
            if position >= len(uuids):
                msg = "Číselný index GPU je mimo rozsah zařízení NVML."
                raise InvalidInputError(msg)
            index = pci_order[position]
        elif token.startswith("GPU-") and token in uuids:
            index = uuids.index(token)
        else:
            msg = "Viditelnost GPU obsahuje neznámou položku."
            raise InvalidInputError(msg)
        if index in indices:
            msg = "Viditelnost GPU obsahuje opakovanou položku."
            raise InvalidInputError(msg)
        indices.append(index)
    return tuple(indices)


class GpuMonitor:
    """Viditelná zařízení a výsledek dostupný až po uzavření okna."""

    def __init__(self, devices: tuple[GpuDevice, ...]) -> None:
        """Připraví monitor viditelných zařízení bez dokončeného měření."""
        self.devices = devices
        self._measurement: GpuMeasurement | None = None

    def measurement(self) -> GpuMeasurement:
        """Vrátí dokončené měření.

        Returns:
            Agregované měření všech viditelných GPU.

        Raises:
            RuntimeError: Okno ještě nebylo úspěšně uzavřeno.
        """
        if self._measurement is None:
            msg = "Měření GPU je dostupné až po uzavření okna."
            raise RuntimeError(msg)
        return self._measurement

    def finish(
        self, sampler: _MemorySampler, thread: Thread | None, start: float
    ) -> None:
        """Zastaví vzorkování a uloží dokončené měření.

        Raises:
            InvalidInputError: Čtení nebo validace nákladů selhaly.
        """
        sampler.stop.set()
        if thread is not None:
            thread.join()
        try:
            usages = sampler.usages()
            self._measurement = GpuMeasurement(
                seconds=perf_counter() - start, devices=usages
            )
        except (pynvml.NVMLError, ValidationError) as error:
            raise InvalidInputError(_measurement_message(error)) from error


@dataclass(frozen=True)
class _DeviceStart:
    handle: object
    device: GpuDevice
    energy_millijoules: int | None
    memory_used_bytes: int


def _measurement_message(error: Exception) -> str:
    return (
        f"{NVML_FAILURE}: {error}"
        if isinstance(error, pynvml.NVMLError)
        else f"{NVML_FAILURE}."
    )


def _energy(handle: object) -> int | None:
    try:
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    except pynvml.NVMLError as error:
        # Binding tvoří podtřídy i atribut value dynamicky; jejich kód čteme bez vypnutí ty.
        if getattr(error, "value", None) == pynvml.NVML_ERROR_NOT_SUPPORTED:
            return None
        raise


def _read_devices(visible: str | None, order: str | None) -> tuple[_DeviceStart, ...]:
    try:
        handles = tuple(
            pynvml.nvmlDeviceGetHandleByIndex(index)
            for index in range(pynvml.nvmlDeviceGetCount())
        )
        uuids = tuple(pynvml.nvmlDeviceGetUUID(handle) for handle in handles)
        pci_bus_ids = tuple(
            pynvml.nvmlDeviceGetPciInfo(handle).busId for handle in handles
        )
        return tuple(
            _read_start(handles[index])
            for index in visible_indices(visible, order, uuids, pci_bus_ids)
        )
    except (pynvml.NVMLError, ValidationError) as error:
        raise InvalidInputError(_measurement_message(error)) from error


def _read_start(handle: object) -> _DeviceStart:
    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return _DeviceStart(
        handle=handle,
        device=GpuDevice(
            name=pynvml.nvmlDeviceGetName(handle), memory_total_bytes=memory.total
        ),
        energy_millijoules=_energy(handle),
        memory_used_bytes=memory.used,
    )


@dataclass
class _MemorySampler:
    starts: tuple[_DeviceStart, ...]
    peaks: list[int]
    stop: Event
    failure: Exception | None = None

    def sample(self) -> None:
        for index, start in enumerate(self.starts):
            memory = pynvml.nvmlDeviceGetMemoryInfo(start.handle)
            self.peaks[index] = max(self.peaks[index], memory.used)

    def run(self, interval_seconds: float) -> None:
        try:
            while not self.stop.wait(interval_seconds):
                self.sample()
        except Exception as error:
            self.failure = error
            logger.debug(
                "Vzorkování paměti GPU selhalo; chybu předá hlavní vlákno.",
                exc_info=True,
            )

    def usages(self) -> tuple[GpuUsage, ...]:
        self.sample()
        if self.failure is not None:
            raise InvalidInputError(
                _measurement_message(self.failure)
            ) from self.failure
        return tuple(
            _usage(start, peak)
            for start, peak in zip(self.starts, self.peaks, strict=True)
        )


def _usage(start: _DeviceStart, peak: int) -> GpuUsage:
    end = _energy(start.handle) if start.energy_millijoules is not None else None
    # Čítač zahrnuje i klidový příkon; PBS přiděluje kartu úloze výhradně.
    joules = (
        (end - start.energy_millijoules) / MILLIJOULES_PER_JOULE
        if end is not None and start.energy_millijoules is not None
        else None
    )
    return GpuUsage(
        device=start.device,
        energy_joules=joules,
        # Ve WSL nelze měřit po procesech; výchozí stav ukazuje i cizí alokace.
        memory_used_start_bytes=start.memory_used_bytes,
        memory_used_peak_bytes=peak,
    )


def _shutdown_nvml() -> None:
    try:
        pynvml.nvmlShutdown()
    except pynvml.NVMLError as error:
        raise InvalidInputError(_measurement_message(error)) from error


@contextmanager
def _nvml_session() -> Generator[bool]:
    available = True
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as error:
        if getattr(error, "value", None) not in {
            pynvml.NVML_ERROR_LIBRARY_NOT_FOUND,
            pynvml.NVML_ERROR_DRIVER_NOT_LOADED,
        }:
            raise InvalidInputError(_measurement_message(error)) from error
        available = False
    if not available:
        yield False
        return
    try:
        yield True
    except BaseException:
        try:
            _shutdown_nvml()
        except Exception as error:
            logger.warning(
                "Ukončení NVML selhalo při zpracování původní výjimky: %s",
                error,
                exc_info=True,
            )
        raise
    else:
        _shutdown_nvml()


@contextmanager
def gpu_monitor(
    cuda_visible_devices: str | None,
    cuda_device_order: str | None,
    *,
    interval_seconds: float = 0.1,
) -> Generator[GpuMonitor]:
    """Změří energii a špičku paměti GPU kolem celého měřeného kroku.

    Yields:
        Monitor s popisem zařízení; výsledek zpřístupní po výstupu z okna.

    Raises:
        InvalidInputError: Viditelnost nebo měření NVML selhaly; chybějící ovladač nevadí.
        ValueError: Interval vzorkování není konečný a kladný.
    """
    if not isfinite(interval_seconds) or interval_seconds <= 0:
        msg = "Interval vzorkování GPU musí být konečný a kladný."
        raise ValueError(msg)
    hidden = cuda_visible_devices is not None and cuda_visible_devices.strip() in {
        "",
        "-1",
    }
    session = nullcontext(enter_result=False) if hidden else _nvml_session()
    with session as available:
        starts = (
            _read_devices(cuda_visible_devices, cuda_device_order) if available else ()
        )
        monitor = GpuMonitor(tuple(start.device for start in starts))
        sampler = _MemorySampler(
            starts, [start.memory_used_bytes for start in starts], Event()
        )
        start = perf_counter()
        thread = (
            Thread(target=sampler.run, args=(interval_seconds,), daemon=True)
            if starts
            else None
        )
        if thread is not None:
            thread.start()
        try:
            yield monitor
        except BaseException:
            try:
                monitor.finish(sampler, thread, start)
            except Exception as error:
                logger.warning(
                    "Dokončení měření GPU selhalo při výjimce měřeného kroku: %s",
                    error,
                    exc_info=True,
                )
            raise
        else:
            monitor.finish(sampler, thread, start)
