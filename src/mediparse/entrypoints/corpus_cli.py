"""Společné chování korpusových příkazů: volby se stejnou cestou a nápovědou."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, assert_never

from mediparse.application.corpus_provenance import (
    ProvenanceRefused,
    ProvenanceWritten,
)
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.mimic_tables import TABLES_PATH
from mediparse.infrastructure.plans_file import PLANS_PATH
from mediparse.infrastructure.sampler_config import SAMPLER_CONFIG_PATH
from mediparse.infrastructure.synthetic_corpus import CORPUS_ROOT
from mediparse.infrastructure.verbalization_template import (
    VERBALIZATION_TEMPLATE_PATH,
)

if TYPE_CHECKING:
    import argparse

    from mediparse.application.corpus_provenance import ProvenanceOutcome

logger = logging.getLogger(__name__)


def add_corpus_option(parser: argparse.ArgumentParser) -> None:
    """Přidá ``--corpus``, kořen syntetického korpusu."""
    parser.add_argument(
        "--corpus", type=Path, default=CORPUS_ROOT, help="kořen syntetického korpusu"
    )


def add_sampler_config_option(parser: argparse.ArgumentParser) -> None:
    """Přidá ``--config``, konfiguraci vzorkovače."""
    parser.add_argument(
        "--config",
        type=Path,
        default=SAMPLER_CONFIG_PATH,
        help="konfigurace vzorkovače",
    )


def add_tables_option(parser: argparse.ArgumentParser) -> None:
    """Přidá ``--tables``, config tabulek MIMIC s oficiálními otisky."""
    parser.add_argument(
        "--tables",
        type=Path,
        default=TABLES_PATH,
        help="config tabulek MIMIC s oficiálními otisky",
    )


def add_plans_option(parser: argparse.ArgumentParser) -> None:
    """Přidá ``--plans``, soubor plánů JSON Lines."""
    parser.add_argument(
        "--plans", type=Path, default=PLANS_PATH, help="soubor plánů JSON Lines"
    )


def add_template_option(parser: argparse.ArgumentParser) -> None:
    """Přidá ``--template``, šablonu instrukcí verbalizace."""
    parser.add_argument(
        "--template",
        type=Path,
        default=VERBALIZATION_TEMPLATE_PATH,
        help="šablona instrukcí verbalizace",
    )


def provenance_exit_code(outcome: ProvenanceOutcome, written: str) -> ExitCode:
    """Převede výsledek zápisu provenance na návratový kód a hlášku v logu.

    Args:
        outcome: Výsledek use case provenance.
        written: Hláška pro úspěšný zápis.

    Returns:
        OK po zápisu, REFUSED s důvodem odmítnutí v logu.
    """
    match outcome:
        case ProvenanceWritten():
            logger.info("%s", written)
            return ExitCode.OK
        case ProvenanceRefused(reason=reason):
            logger.error("%s", reason)
            return ExitCode.REFUSED
        case _:
            assert_never(outcome)
