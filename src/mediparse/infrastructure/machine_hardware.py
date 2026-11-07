"""Hardware stroje ze systémových údajů Linuxu."""

from __future__ import annotations

import os
import platform
from pathlib import Path

from mediparse.domain.environment import Hardware
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


def machine_hardware() -> Hardware:
    """Zjistí CPU, jádra systému i procesu a fyzickou paměť hostitele.

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
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        physical_pages = os.sysconf("SC_PHYS_PAGES")
        if page_size <= 0 or physical_pages <= 0:
            msg = "Systém neuvádí kladnou velikost fyzické paměti."
            raise InvalidInputError(msg)
        return Hardware.model_validate({
            "cpu_model": cpu_model(cpuinfo, platform.machine()),
            "logical_cores": os.cpu_count(),
            "available_cores": os.process_cpu_count(),
            "memory_bytes": page_size * physical_pages,
        })
    except (OSError, ValueError) as error:
        msg = "Hardware stroje nelze zjistit ze systémových údajů."
        raise InvalidInputError(msg) from error
