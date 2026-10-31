"""Složení vstupní tabulky tréninku a inference podle přepínače stripu v configu."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.application.diagnostic_strip import DiagnosticStripTable
from mediparse.domain.run_manifest import DiagnosticStrip
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.section_lexicon_config import load_section_lexicon

if TYPE_CHECKING:
    from pathlib import Path

    from mediparse.application.ports.experiment import InputTable
    from mediparse.domain.training import TrainingConfig


def training_input(
    cohort: Path, training: TrainingConfig, lexicon: Path
) -> tuple[InputTable, DiagnosticStrip | None]:
    """Složí tabulku zpráv, kterou čte trénink i inference, a záznam stripu pro manifest.

    Trénink a inference volají tutéž funkci se stejným configem, aby klasifikovaly týž text.

    Args:
        cohort: Cesta k tabulce kohorty.
        training: Config tréninku s přepínačem stripu.
        lexicon: Cesta k lexikonu diagnostických sekcí.

    Returns:
        Tabulka beze změny a ``None``, když přepínač stripu je vypnutý. Jinak tabulka bez
        diagnostických sekcí a záznam s jejich seřazenými klíči a otiskem lexikonu.
    """
    table = ParquetNoteTable(cohort)
    if not training.strip_diagnostic_sections:
        return table, None
    sections = load_section_lexicon(lexicon)
    record = DiagnosticStrip(
        sections=tuple(sections.diagnostic_sections),
        lexicon_sha256=file_sha256(lexicon),
    )
    return DiagnosticStripTable(table, sections), record
