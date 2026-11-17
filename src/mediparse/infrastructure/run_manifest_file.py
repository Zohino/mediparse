"""Načtení manifestů tréninku a ladění z JSON souboru."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.domain.run_manifest import RunManifest, TuningManifest
from mediparse.infrastructure.input_file import parse_file

if TYPE_CHECKING:
    from pathlib import Path


def load_run_manifest(path: Path) -> RunManifest:
    """Přečte a přísně validuje manifest tréninku.

    Returns:
        Manifest, který prošel schématem.
    """
    return parse_file(path, RunManifest.model_validate_json)


def load_tuning_manifest(path: Path) -> TuningManifest:
    """Přečte a přísně validuje manifest ladění.

    Returns:
        Manifest, který prošel schématem.
    """
    return parse_file(path, TuningManifest.model_validate_json)
