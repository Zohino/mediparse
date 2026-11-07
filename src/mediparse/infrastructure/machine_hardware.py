"""Hardware stroje ze systémových údajů Linuxu."""

from __future__ import annotations

import os
import platform
from pathlib import Path

from pydantic import ValidationError

from mediparse.domain.environment import GpuDevice, Hardware
from mediparse.domain.inputs import InvalidInputError


def cpu_model(cpuinfo: str, fallback: str) -> str:
    """Vybere první model CPU z cpuinfo, jinak předanou architekturu.

    Returns:
        Název modelu bez okolních mezer nebo fallback.
    """
    for line in cpuinfo.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "model name" and value.strip():
            return value.strip()
    return fallback


def machine_hardware(gpus: tuple[GpuDevice, ...]) -> Hardware:
    """Zjistí CPU, jádra systému i procesu a fyzickou paměť hostitele.

    Args:
        gpus: Viditelná GPU z adaptéru NVML, převezmou se beze změny.

    Returns:
        Hardware stroje, na kterém běží proces.

    Raises:
        InvalidInputError: Kapacity stroje nelze zjistit nebo nejsou kladné.
    """
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        # Mimo procfs je architektura jediný dostupný popis CPU.
        cpuinfo = ""
    unknown_hardware = "Hardware stroje nelze zjistit ze systémových údajů."
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        physical_pages = os.sysconf("SC_PHYS_PAGES")
        logical_cores = os.cpu_count()
        available_cores = os.process_cpu_count()
    except (OSError, ValueError) as error:
        raise InvalidInputError(unknown_hardware) from error
    if page_size <= 0 or physical_pages <= 0:
        msg = "Systém neuvádí kladnou velikost fyzické paměti."
        raise InvalidInputError(msg)
    try:
        return Hardware.model_validate({
            "cpu_model": cpu_model(cpuinfo, platform.machine()),
            "logical_cores": logical_cores,
            "available_cores": available_cores,
            "memory_bytes": page_size * physical_pages,
            "gpus": gpus,
        })
    except ValidationError as error:
        raise InvalidInputError(unknown_hardware) from error
