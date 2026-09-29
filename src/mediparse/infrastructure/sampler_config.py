"""Načtení konfigurace vzorkovače plánů z JSON souboru."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.domain.synthetic_plan import SamplerConfig

if TYPE_CHECKING:
    from pathlib import Path


def load_sampler_config(path: Path) -> SamplerConfig:
    """Přečte JSON konfiguraci a předá ji validaci; neplatný obsah vyhodí ValidationError.

    Returns:
        Konfigurace, která prošla schématem i invarianty.
    """
    return SamplerConfig.model_validate_json(path.read_text(encoding="utf-8"))
