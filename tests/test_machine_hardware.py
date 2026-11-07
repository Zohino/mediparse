"""Údaje o stroji: zkonstruovaný cpuinfo a skutečné kapacity hostitele."""

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
