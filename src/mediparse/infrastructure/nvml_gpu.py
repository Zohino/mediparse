"""Měření energie z čítače NVML a vzorkování paměti viditelných GPU."""

from __future__ import annotations

import logging
import sys
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, field
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
NVML_FAILURE: Final = "Měření GPU přes NVML selhalo."


def visible_indices(
    cuda_visible_devices: str | None,
    cuda_device_order: str | None,
    uuids: Sequence[str],
) -> tuple[int, ...]:
    """Přiřadí viditelnost CUDA k indexům NVML bez odhadu jiné karty.

    Returns:
        Indexy NVML v pořadí viditelnosti.

    Raises:
        InvalidInputError: Položka je neznámá, opakovaná nebo pořadí nejednoznačné.
    """
    if cuda_visible_devices is None:
        return tuple(range(len(uuids)))
    if cuda_visible_devices.strip() in {"", "-1"}:
        return ()
    indices: list[int] = []
    invalid_visibility = "Viditelnost GPU nelze jednoznačně přiřadit k zařízením NVML."
    # NVML ignoruje CUDA_VISIBLE_DEVICES, proto viditelnost mapujeme sami.
    for item in cuda_visible_devices.split(","):
        token = item.strip()
        if token.isascii() and token.isdecimal():
            # CUDA jinak řadí nejrychlejší kartu první, NVML podle sběrnice PCI.
            if len(uuids) > 1 and cuda_device_order != "PCI_BUS_ID":
                raise InvalidInputError(invalid_visibility)
            index = int(token)
        elif token.startswith("GPU-") and token in uuids:
            index = uuids.index(token)
        else:
            raise InvalidInputError(invalid_visibility)
        if index >= len(uuids) or index in indices:
            raise InvalidInputError(invalid_visibility)
        indices.append(index)
    return tuple(indices)


@dataclass(frozen=True)
class GpuMonitor:
    """Viditelná zařízení a výsledek dostupný až po uzavření okna."""

    devices: tuple[GpuDevice, ...]
    _result: Future[GpuMeasurement] = field(repr=False)

    def measurement(self) -> GpuMeasurement:
        """Vrátí dokončené měření.

        Returns:
            Agregované měření všech viditelných GPU.

        Raises:
            RuntimeError: Okno ještě nebylo úspěšně uzavřeno.
        """
        if not self._result.done():
            msg = "Měření GPU je dostupné až po uzavření okna."
            raise RuntimeError(msg)
        return self._result.result()


@dataclass(frozen=True)
class _DeviceStart:
    handle: object
    device: GpuDevice
    energy_millijoules: int | None
    memory_used_bytes: int


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
        return tuple(
            _read_start(handles[index])
            for index in visible_indices(visible, order, uuids)
        )
    except (pynvml.NVMLError, ValidationError) as error:
        raise InvalidInputError(NVML_FAILURE) from error


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


@dataclass(frozen=True)
class _MemorySampler:
    starts: tuple[_DeviceStart, ...]
    peaks: list[int]
    stop: Event
    failure: Future[None] = field(default_factory=Future)

    def sample(self) -> None:
        for index, start in enumerate(self.starts):
            memory = pynvml.nvmlDeviceGetMemoryInfo(start.handle)
            self.peaks[index] = max(self.peaks[index], memory.used)

    def run(self, interval_seconds: float) -> None:
        try:
            while not self.stop.wait(interval_seconds):
                self.sample()
        except pynvml.NVMLError as error:
            self.failure.set_exception(error)

    def usages(self) -> tuple[GpuUsage, ...]:
        self.sample()
        if self.failure.done():
            self.failure.result()
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


def _finish_window(
    result: Future[GpuMeasurement],
    sampler: _MemorySampler,
    thread: Thread | None,
    start: float,
) -> None:
    body_error = sys.exception()
    sampler.stop.set()
    if thread is not None:
        thread.join()
    try:
        usages = sampler.usages()
        result.set_result(
            GpuMeasurement(seconds=perf_counter() - start, devices=usages)
        )
    except (pynvml.NVMLError, ValidationError) as error:
        if body_error is not None:
            # Původní selhání kroku má přednost před chybou dokončení měření.
            logger.warning("Dokončení měření GPU selhalo při výjimce měřeného kroku.")
            return
        raise InvalidInputError(NVML_FAILURE) from error


@contextmanager
def _nvml_session() -> Generator[bool]:
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as error:
        if getattr(error, "value", None) in {
            pynvml.NVML_ERROR_LIBRARY_NOT_FOUND,
            pynvml.NVML_ERROR_DRIVER_NOT_LOADED,
        }:
            yield False
            return
        raise InvalidInputError(NVML_FAILURE) from error
    try:
        yield True
    finally:
        active_error = sys.exception()
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError as error:
            if active_error is None:
                raise InvalidInputError(NVML_FAILURE) from error
            logger.warning("Ukončení NVML selhalo při zpracování původní výjimky.")


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
    with _nvml_session() as available:
        starts = (
            _read_devices(cuda_visible_devices, cuda_device_order) if available else ()
        )
        result: Future[GpuMeasurement] = Future()
        monitor = GpuMonitor(tuple(start.device for start in starts), result)
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
        finally:
            _finish_window(result, sampler, thread, start)
