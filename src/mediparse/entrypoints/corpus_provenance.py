"""Vstupní bod provenance syntetického korpusu: po čistém auditu zapíše, čím a podle čeho korpus vznikl.

Konzolový skript ``mediparse-corpus-provenance`` dostane model, verzi Claude Code,
den generování a commit specifikace jako argumenty; git ani Claude Code nevolá.
Seed, otisk configu vzorkovače a otisk zadání všech zpráv bere ze souborů v repu.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from typing import TYPE_CHECKING

from mediparse.application.corpus_provenance import (
    CorpusProvenance,
)
from mediparse.domain.corpus_provenance import (
    Generation,
    model_id,
    prompts_sha256,
    semantic_version,
)
from mediparse.domain.digest import commit_sha
from mediparse.entrypoints.cli import (
    argument,
    configure_logging,
    refusing_invalid_input,
)
from mediparse.entrypoints.corpus_cli import (
    add_corpus_option,
    add_plans_option,
    add_sampler_config_option,
    add_tables_option,
    add_template_option,
)
from mediparse.entrypoints.provenance_exit import provenance_exit_code
from mediparse.infrastructure.audit_pin import load_audit_pin
from mediparse.infrastructure.file_digest import file_sha256
from mediparse.infrastructure.plans_file import PlansFile
from mediparse.infrastructure.sampler_config import (
    load_sampler_config,
)
from mediparse.infrastructure.synthetic_corpus import (
    PROVENANCE_NAME,
    CorpusDirectory,
)
from mediparse.infrastructure.verbalization_template import (
    load_verbalization_template,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.entrypoints.exit_code import ExitCode


def main() -> ExitCode:
    """Konzolový skript ``mediparse-corpus-provenance``.

    Returns:
        Návratový kód.
    """
    configure_logging()
    return run(sys.argv[1:])


@refusing_invalid_input
def run(argv: Sequence[str]) -> ExitCode:
    """Složí provenance z argumentů a souborů v repu a zapíše ji ke korpusu.

    Returns:
        OK po zapsání provenance, REFUSED při neplatném vstupním souboru nebo korpusu bez čistého auditu.
    """
    args = _parser().parse_args(argv)
    config = load_sampler_config(args.config)
    prompts = prompts_sha256(
        load_verbalization_template(args.template),
        PlansFile(args.plans).load(),
        config,
    )
    generation = Generation(
        seed=config.seed,
        sampler_config_sha256=file_sha256(args.config),
        prompts_sha256=prompts,
        model=args.model,
        claude_code_version=args.claude_code_version,
        generated_on=args.date,
        specification_commit=args.specification_commit,
    )
    outcome = CorpusProvenance(CorpusDirectory(args.corpus)).run(
        generation,
        load_audit_pin(args.tables, args.config),
    )
    return provenance_exit_code(
        outcome, f"Provenance zapsána do {args.corpus / PROVENANCE_NAME}."
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mediparse-corpus-provenance",
        description="Provenance auditovaného syntetického korpusu.",
    )
    parser.add_argument(
        "--model",
        type=argument(model_id),
        required=True,
        help="model, který zprávy napsal, např. claude-sonnet-5-5",
    )
    parser.add_argument(
        "--claude-code-version",
        type=argument(semantic_version),
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
        type=argument(commit_sha),
        required=True,
        help="commit docs/synthetic-corpus.md, podle kterého korpus vznikl",
    )
    add_corpus_option(parser)
    add_sampler_config_option(parser)
    add_plans_option(parser)
    add_template_option(parser)
    add_tables_option(parser)
    return parser
