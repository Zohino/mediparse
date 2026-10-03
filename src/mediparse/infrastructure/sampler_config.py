"""Načtení konfigurace vzorkovače plánů z JSON souboru."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from mediparse.domain.synthetic_plan import SamplerConfig
from mediparse.infrastructure.input_file import parse_file

SAMPLER_CONFIG_PATH: Final = Path("config/synthetic_plan.json")


def load_sampler_config(path: Path) -> SamplerConfig:
    """Přečte JSON konfiguraci a předá ji validaci.

    Returns:
        Konfigurace, která prošla schématem i invarianty.
    """
    return parse_file(path, SamplerConfig.model_validate_json)
