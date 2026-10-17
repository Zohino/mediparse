"""Vstupní bod provenance překladu: k provenance auditovaného korpusu doplní, čím a odkud vznikl český korpus.

Konzolový skript ``mediparse-translation-provenance`` čte pracovní adresář překladu
(požadavky, běhy a výstupy), ověří, že český korpus je jeho výsledkem pro současné
anglické originály, a zapíše část ``translation``. Provenance generování ponechá.
S daty MIMIC nepracuje, jen s jejich oficiálními otisky z configu.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from mediparse.application.translation_provenance import TranslationProvenance
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.corpus_cli import (
    add_corpus_option,
    add_sampler_config_option,
    add_tables_option,
    provenance_exit_code,
)
from mediparse.infrastructure.audit_pin import load_audit_pin
from mediparse.infrastructure.synthetic_corpus import (
    PROVENANCE_NAME,
    CorpusDirectory,
)
from mediparse.infrastructure.translation_workdir import (
    WORKDIR_PATH,
    TranslationWorkdir,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.entrypoints.exit_code import ExitCode


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
        load_audit_pin(args.tables, args.config),
    )
    return provenance_exit_code(
        outcome, f"Provenance překladu zapsána do {args.corpus / PROVENANCE_NAME}."
    )


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
    add_corpus_option(parser)
    add_sampler_config_option(parser)
    add_tables_option(parser)
    return parser
