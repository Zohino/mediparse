"""Složení vstupní tabulky a normalizace textu ze sdílených voleb vstupu modelu."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.application.diagnostic_strip import DiagnosticStripTable
from mediparse.domain.run_manifest import DiagnosticStrip
from mediparse.infrastructure.diagnosis_candidates_file import load_text_normalization
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.note_table import ParquetNoteTable
from mediparse.infrastructure.section_lexicon_config import load_section_lexicon

if TYPE_CHECKING:
    from pathlib import Path

    from mediparse.application.ports.experiment import InputTable
    from mediparse.domain.clinical_text import TextNormalization
    from mediparse.domain.model_input import ModelInput


def training_input(
    cohort: Path, model_input: ModelInput, lexicon: Path
) -> tuple[InputTable, DiagnosticStrip | None]:
    """Složí tabulku zpráv, kterou čte trénink i inference, a záznam stripu pro manifest.

    Trénink a inference volají tutéž funkci se stejnými volbami, aby klasifikovaly týž text.

    Args:
        cohort: Cesta k tabulce kohorty.
        model_input: Volby vstupu s přepínačem stripu.
        lexicon: Cesta k lexikonu diagnostických sekcí.

    Returns:
        Tabulka beze změny a ``None``, když přepínač stripu je vypnutý. Jinak tabulka bez
        diagnostických sekcí a záznam s jejich seřazenými klíči a otiskem lexikonu.
    """
    table = ParquetNoteTable(cohort)
    if not model_input.strip_diagnostic_sections:
        return table, None
    sections = load_section_lexicon(lexicon)
    record = DiagnosticStrip(
        sections=tuple(sections.diagnostic_sections),
        lexicon_sha256=file_sha256(lexicon),
    )
    return DiagnosticStripTable(table, sections), record


def text_normalization(
    candidates: Path, model_input: ModelInput
) -> TextNormalization | None:
    """Načte pravidla normalizace podle úrovně předzpracování vstupu.

    Args:
        candidates: Cesta ke kandidátům diagnóz se zkratkami a variantami.
        model_input: Volby vstupu s úrovní předzpracování textu.

    Returns:
        Pravidla normalizace, nebo ``None`` pro původní text bez předzpracování.
    """
    if not model_input.text_preprocessing.normalizes:
        return None
    return load_text_normalization(
        candidates, variants=model_input.text_preprocessing.rewrites_variants
    )
