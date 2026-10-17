"""Skládání pinu auditu z configu tabulek MIMIC a configu vzorkovače."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from mediparse.domain.corpus_audit import AuditPin
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.audit_pin import load_audit_pin
from mediparse.infrastructure.mimic_tables import load_reference_sha256
from tests.support import REPOSITORY_CONFIG, REPOSITORY_TABLES

if TYPE_CHECKING:
    from pathlib import Path

    from mediparse.infrastructure.sampler_config import SamplerConfig


def test_repository_pin_holds_note_reference_and_sampler_labels(
    sampler_config: SamplerConfig,
) -> None:
    """Pin z configů v repu nese otisky jen MIMIC-IV-Note a štítky struktury vzorkovače."""
    pin = load_audit_pin(REPOSITORY_TABLES, REPOSITORY_CONFIG)

    assert pin == AuditPin(
        load_reference_sha256(REPOSITORY_TABLES),
        sampler_config.structure.structure_labels,
    )


@pytest.mark.parametrize("missing", ["tables", "config"])
def test_missing_config_is_invalid_input(tmp_path: Path, missing: str) -> None:
    """Chybějící config tabulek nebo vzorkovače končí InvalidInputError."""
    tables = tmp_path / "none.json" if missing == "tables" else REPOSITORY_TABLES
    config = tmp_path / "none.json" if missing == "config" else REPOSITORY_CONFIG

    with pytest.raises(InvalidInputError):
        load_audit_pin(tables, config)
