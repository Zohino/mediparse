"""Načtení striktního JSON configu ladění stupně."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.domain.stage_tuning import StageTuningConfig
from mediparse.infrastructure.input_file import parse_file

if TYPE_CHECKING:
    from pathlib import Path


def load_stage_config(path: Path) -> StageTuningConfig:
    """Přečte a ověří config CASH stupně.

    Returns:
        Striktně validovaný config.
    """
    return parse_file(path, StageTuningConfig.model_validate_json)
