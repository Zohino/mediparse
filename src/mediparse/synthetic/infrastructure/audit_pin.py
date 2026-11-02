"""Pin auditu složený z configu tabulek MIMIC a configu vzorkovače."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from mediparse.infrastructure.mimic_tables import load_mimic_tables
from mediparse.synthetic.domain.corpus_audit import AuditPin
from mediparse.synthetic.infrastructure.sampler_config import load_sampler_config

if TYPE_CHECKING:
    from pathlib import Path


REFERENCE_PROJECT: Final = "mimic-iv-note"


def load_reference_sha256(path: Path) -> dict[str, str]:
    """Přečte otisky tabulek projektu MIMIC-IV-Note, proti kterým běží audit.

    Returns:
        Slovník jméno souboru z URL → SHA-256 ze ``SHA256SUMS.txt`` na PhysioNetu.
    """
    return {
        name: table.sha256
        for name, table in load_mimic_tables(path).items()
        if REFERENCE_PROJECT in table.path.parts
    }


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
