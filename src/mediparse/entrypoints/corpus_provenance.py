"""Vstupní bod provenance syntetického korpusu: po čistém auditu zapíše, čím a podle čeho korpus vznikl.

Konzolový skript ``mediparse-corpus-provenance`` dostane model, verzi Claude Code,
den generování a commit specifikace jako argumenty; git ani Claude Code nevolá.
Seed a otisky configu vzorkovače a šablony instrukcí bere ze souborů v repu.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, assert_never

from mediparse.application.corpus_provenance import (
    CorpusProvenance,
    ProvenanceRefused,
    ProvenanceWritten,
)
from mediparse.domain.corpus_provenance import Generation
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.file_digest import file_sha256
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
from mediparse.infrastructure.verbalization_template import (
    VERBALIZATION_TEMPLATE_PATH,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def main() -> ExitCode:
    """Konzolový skript ``mediparse-corpus-provenance``.

    Returns:
        Návratový kód.
    """
    return run(sys.argv[1:])


def run(argv: Sequence[str]) -> ExitCode:
    """Složí provenance z argumentů a souborů v repu a zapíše ji ke korpusu.

    Returns:
        OK po zapsání provenance, REFUSED při neplatných údajích nebo korpusu bez čistého auditu.
    """
    args = _parser().parse_args(argv)
    seed = load_sampler_config(args.config).seed
    try:
        generation = Generation(
            seed=seed,
            sampler_config_sha256=file_sha256(args.config),
            verbalization_template_sha256=file_sha256(args.template),
            model=args.model,
            claude_code_version=args.claude_code_version,
            generated_on=args.date,
            specification_commit=args.specification_commit,
        )
    except ValueError as error:
        sys.stderr.write(f"Neplatné údaje o generování: {error}\n")
        return ExitCode.REFUSED
    outcome = CorpusProvenance(CorpusDirectory(args.corpus)).run(
        generation, load_reference_sha256(args.tables)
    )
    match outcome:
        case ProvenanceWritten():
            sys.stdout.write(
                f"Provenance zapsána do {args.corpus / PROVENANCE_NAME}.\n"
            )
            return ExitCode.OK
        case ProvenanceRefused(reason=reason):
            sys.stderr.write(f"{reason}\n")
            return ExitCode.REFUSED
        case _:
            assert_never(outcome)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mediparse-corpus-provenance",
        description="Provenance auditovaného syntetického korpusu.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="model, který zprávy napsal, např. claude-opus-5-5",
    )
    parser.add_argument(
        "--claude-code-version",
        required=True,
        help="verze Claude Code při generování, z `claude --version`",
    )
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        required=True,
        help="den dokončení generování ve tvaru RRRR-MM-DD",
    )
    parser.add_argument(
        "--specification-commit",
        required=True,
        help="commit docs/synthetic-corpus.md, podle kterého korpus vznikl",
    )
    parser.add_argument(
        "--corpus", type=Path, default=CORPUS_ROOT, help="kořen syntetického korpusu"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=SAMPLER_CONFIG_PATH,
        help="konfigurace vzorkovače",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=VERBALIZATION_TEMPLATE_PATH,
        help="šablona instrukcí verbalizace",
    )
    parser.add_argument(
        "--tables",
        type=Path,
        default=TABLES_PATH,
        help="config tabulek MIMIC s oficiálními otisky",
    )
    return parser
