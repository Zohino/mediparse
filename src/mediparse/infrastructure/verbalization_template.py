"""Šablona instrukcí verbalizace v repu: text s místy pro údaje ze zprávy."""

from __future__ import annotations

from pathlib import Path
from typing import Final

VERBALIZATION_TEMPLATE_PATH: Final = Path("config/verbalization_template.md")


def load_verbalization_template(path: Path) -> str:
    """Přečte šablonu instrukcí.

    Returns:
        Text šablony v UTF-8.
    """
    return path.read_text(encoding="utf-8")
