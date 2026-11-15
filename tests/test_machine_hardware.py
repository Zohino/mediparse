"""Údaje o stroji: zkonstruovaný cpuinfo a skutečné kapacity hostitele."""

import os
import platform
import resource
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mediparse.domain.environment import GpuDevice
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.machine_hardware import (
    cpu_model,
    machine_hardware,
    peak_rss_bytes,
    peak_rss_gb,
)


def test_first_cpu_model_is_stripped() -> None:
    """Dva zkonstruované procesory vrátí první model bez okolních mezer."""
    cpuinfo = (
        "processor : 0\nmodel name :  Synthetic CPU A  \n\n"
        "processor : 1\nmodel name : Synthetic CPU B\n"
    )

    assert cpu_model(cpuinfo, "fallback") == "Synthetic CPU A"


def test_missing_model_uses_fallback() -> None:
    """CPU bez položky model name použije architekturu stroje."""
    assert cpu_model("processor : 0\n", "fallback") == "fallback"


def test_available_cores_do_not_exceed_machine() -> None:
    """Proces nemá více dostupných jader než celý stroj."""
    hardware = machine_hardware(gpus=())

    assert 1 <= hardware.available_cores <= hardware.logical_cores


@pytest.mark.parametrize(
    ("page_size", "physical_pages"),
    [(-1, 4096), (4096, -1), (-1, -1), (0, 4096), (4096, 0)],
)
def test_unknown_memory_keeps_specific_reason(
    monkeypatch: pytest.MonkeyPatch, page_size: int, physical_pages: int
) -> None:
    """Nezjištěná velikost paměti zachová konkrétní důvod odmítnutí."""
    values = {"SC_PAGE_SIZE": page_size, "SC_PHYS_PAGES": physical_pages}
    monkeypatch.setattr(os, "sysconf", values.__getitem__)

    with pytest.raises(
        InvalidInputError,
        match=r"^Systém neuvádí kladnou velikost fyzické paměti\.$",
    ):
        machine_hardware(gpus=())


def test_unknown_cpu_count_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chybějící počet jader se přeloží na odmítnutí nezjistitelného HW."""
    monkeypatch.setattr(os, "cpu_count", lambda: None)

    with pytest.raises(
        InvalidInputError,
        match=r"^Hardware stroje nelze zjistit ze systémových údajů\.$",
    ):
        machine_hardware(gpus=())


def test_hardware_preserves_visible_gpu_order() -> None:
    """GPU dodá NVML adaptér, systémové údaje jejich pořadí nezmění."""
    gpus = (
        GpuDevice(name="Fake GPU B", memory_total_bytes=2000),
        GpuDevice(name="Fake GPU A", memory_total_bytes=1000),
    )
    assert machine_hardware(gpus).gpus == gpus


def test_unreadable_cpuinfo_records_architecture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OSError při čtení procfs zachová známou syntetickou architekturu CPU."""
    monkeypatch.setattr(
        Path, "read_text", Mock(side_effect=OSError("Synthetic procfs failure"))
    )
    monkeypatch.setattr(platform, "machine", lambda: "synthetic-arch")
    assert machine_hardware(gpus=()).cpu_model == "synthetic-arch"


def test_peak_rss_converts_linux_kib_to_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Linux ru_maxrss=1536 KiB znamená ručně spočtených 1572864 bajtů."""
    usage = Mock(return_value=SimpleNamespace(ru_maxrss=1536))
    monkeypatch.setattr(resource, "getrusage", usage)

    expected_bytes = 1_572_864
    assert peak_rss_bytes() == expected_bytes
    usage.assert_called_once_with(resource.RUSAGE_SELF)


def test_peak_rss_gb_uses_decimal_gigabytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Jeden a půl miliardy bajtů se formátuje jako 1,50 desítkového GB."""
    monkeypatch.setattr(
        "mediparse.infrastructure.machine_hardware.peak_rss_bytes",
        lambda: 1_500_000_000,
    )

    assert peak_rss_gb() == "peak RSS 1.50 GB"
