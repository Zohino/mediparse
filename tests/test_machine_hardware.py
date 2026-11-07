"""Údaje o stroji: zkonstruovaný cpuinfo a skutečné kapacity hostitele."""

import os

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.machine_hardware import cpu_model, machine_hardware


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


def test_machine_capacities_are_positive() -> None:
    """Hostitel má kladné kapacity a proces nemá více jader než celý stroj."""
    hardware = machine_hardware()

    assert hardware.cpu_model
    assert hardware.logical_cores >= 1
    assert 1 <= hardware.available_cores <= hardware.logical_cores
    assert hardware.memory_bytes >= 1


@pytest.mark.parametrize(
    ("page_size", "physical_pages"), [(-1, 4096), (4096, -1), (-1, -1)]
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
        machine_hardware()


def test_unknown_cpu_count_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chybějící počet jader se přeloží na odmítnutí nezjistitelného HW."""
    monkeypatch.setattr(os, "cpu_count", lambda: None)

    with pytest.raises(
        InvalidInputError,
        match=r"^Hardware stroje nelze zjistit ze systémových údajů\.$",
    ):
        machine_hardware()
