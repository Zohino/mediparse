"""Načtení manifestu tréninku z JSON souboru."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.domain.run_manifest import RunManifest
from mediparse.infrastructure.input_file import parse_file

if TYPE_CHECKING:
    from pathlib import Path


def load_run_manifest(path: Path) -> RunManifest:
    """Přečte a přísně validuje manifest tréninku.

    Returns:
        Manifest, který prošel schématem.
    """
    return parse_file(path, RunManifest.model_validate_json)
