"""Načtení konfigurace vzorkovače plánů z JSON souboru."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from mediparse.domain.synthetic_plan import SamplerConfig

SAMPLER_CONFIG_PATH: Final = Path("config/synthetic_plan.json")


def load_sampler_config(path: Path) -> SamplerConfig:
    """Přečte JSON konfiguraci a předá ji validaci; neplatný obsah vyhodí ValidationError.

    Returns:
        Konfigurace, která prošla schématem i invarianty.
    """
    return SamplerConfig.model_validate_json(path.read_text(encoding="utf-8"))
