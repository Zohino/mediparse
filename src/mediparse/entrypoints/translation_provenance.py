"""Vstupní bod provenance překladu: k provenance auditovaného korpusu doplní, čím a odkud vznikl český korpus.

Konzolový skript ``mediparse-translation-provenance`` čte pracovní adresář překladu
(požadavky, běhy a výstupy), ověří, že český korpus je jeho výsledkem pro současné
anglické originály, a zapíše část ``translation``. Provenance generování ponechá.
S daty MIMIC nepracuje, jen s jejich oficiálními otisky z configu.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, assert_never

from mediparse.application.corpus_provenance import (
    ProvenanceRefused,
    ProvenanceWritten,
)
from mediparse.application.translation_provenance import TranslationProvenance
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.mimic_tables import TABLES_PATH, load_reference_sha256
from mediparse.infrastructure.sampler_config import (
    SAMPLER_CONFIG_PATH,
    load_sampler_config,
)
from mediparse.infrastructure.synthetic_corpus import (
    CORPUS_ROOT,
    PROVENANCE_NAME,
    CorpusDirectory,
)
from mediparse.infrastructure.translation_workdir import (
    WORKDIR_PATH,
    TranslationWorkdir,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


def main() -> ExitCode:
    """Konzolový skript ``mediparse-translation-provenance``.

    Returns:
        Návratový kód.
    """
    configure_logging()
    return run(sys.argv[1:])


@refusing_invalid_input
def run(argv: Sequence[str]) -> ExitCode:
    """Složí provenance překladu z pracovního adresáře a zapíše ji ke korpusu.

    Returns:
        OK po zapsání, REFUSED při neplatném vstupním souboru, pracovním adresáři
        nebo korpusu, který neodpovídá auditu či překladu.
    """
    args = _parser().parse_args(argv)
    outcome = TranslationProvenance(
        CorpusDirectory(args.corpus), TranslationWorkdir(args.workdir)
    ).run(
        load_reference_sha256(args.tables),
        load_sampler_config(args.config).structure.structure_labels,
    )
    match outcome:
        case ProvenanceWritten():
            logger.info(
                "Provenance překladu zapsána do %s.", args.corpus / PROVENANCE_NAME
            )
            return ExitCode.OK
        case ProvenanceRefused(reason=reason):
            logger.error("%s", reason)
            return ExitCode.REFUSED
        case _:
            assert_never(outcome)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mediparse-translation-provenance",
        description="Provenance českého překladu syntetického korpusu.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path(WORKDIR_PATH),
        help="pracovní adresář překladu s requests.json, runs.jsonl a outputs.jsonl",
    )
    parser.add_argument(
        "--corpus", type=Path, default=CORPUS_ROOT, help="kořen syntetického korpusu"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=SAMPLER_CONFIG_PATH,
        help="konfigurace vzorkovače se štítky struktury zprávy",
    )
    parser.add_argument(
        "--tables",
        type=Path,
        default=TABLES_PATH,
        help="config tabulek MIMIC s oficiálními otisky",
    )
    return parser
