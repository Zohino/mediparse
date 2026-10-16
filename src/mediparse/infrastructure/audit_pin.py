"""Pin auditu složený z configu tabulek MIMIC a configu vzorkovače."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.domain.corpus_audit import AuditPin
from mediparse.infrastructure.mimic_tables import load_reference_sha256
from mediparse.infrastructure.sampler_config import load_sampler_config

if TYPE_CHECKING:
    from pathlib import Path


def load_audit_pin(tables: Path, config: Path) -> AuditPin:
    """Načte pin auditu: otisky referenčních souborů a štítky struktury zprávy.

    Args:
        tables: Config tabulek MIMIC s otisky referenčních souborů.
        config: Config vzorkovače, z něhož pochází štítky struktury.

    Returns:
        Pin, proti kterému záznam auditu platí.
    """
    return AuditPin(
        load_reference_sha256(tables),
        load_sampler_config(config).structure.structure_labels,
    )
