"""Sdílené volby korpusových příkazů: jedna výchozí cesta a jedna nápověda na volbu."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mediparse.application.corpus_provenance import (
    ProvenanceRefused,
    ProvenanceWritten,
)
from mediparse.entrypoints.corpus_cli import (
    add_corpus_option,
    add_plans_option,
    add_sampler_config_option,
    add_tables_option,
    add_template_option,
    provenance_exit_code,
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
    from collections.abc import Callable

type AddOption = Callable[[argparse.ArgumentParser], None]

DEFAULTS = [
    pytest.param(add_corpus_option, "corpus", CORPUS_ROOT, id="corpus"),
    pytest.param(add_sampler_config_option, "config", SAMPLER_CONFIG_PATH, id="config"),
    pytest.param(add_tables_option, "tables", TABLES_PATH, id="tables"),
    pytest.param(add_plans_option, "plans", PLANS_PATH, id="plans"),
    pytest.param(
        add_template_option, "template", VERBALIZATION_TEMPLATE_PATH, id="template"
    ),
]
HELPS = [
    pytest.param(add_corpus_option, "kořen syntetického korpusu", id="corpus"),
    pytest.param(add_sampler_config_option, "konfigurace vzorkovače", id="config"),
    pytest.param(
        add_tables_option, "config tabulek MIMIC s oficiálními otisky", id="tables"
    ),
    pytest.param(add_plans_option, "soubor plánů JSON Lines", id="plans"),
    pytest.param(add_template_option, "šablona instrukcí verbalizace", id="template"),
]


def _parser(add: AddOption) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nástroj")
    add(parser)
    return parser


@pytest.mark.parametrize(("add", "name", "default"), DEFAULTS)
def test_option_defaults_to_infrastructure_path(
    add: AddOption, name: str, default: Path
) -> None:
    """Bez volby platí cesta z konstanty infrastruktury."""
    args = _parser(add).parse_args([])

    assert getattr(args, name) == default


@pytest.mark.parametrize(("add", "name", "default"), DEFAULTS)
def test_option_can_be_overridden_and_is_a_path(
    add: AddOption, name: str, default: Path
) -> None:
    """Zadaná hodnota přebije výchozí a je typu ``Path``."""
    assert default != Path("/x")

    args = _parser(add).parse_args([f"--{name}=/x"])

    assert getattr(args, name) == Path("/x")


@pytest.mark.parametrize(("add", "help_text"), HELPS)
def test_option_help_is_in_parser_help(add: AddOption, help_text: str) -> None:
    """Nápověda volby je ve výpisu nápovědy parseru."""
    assert help_text in " ".join(_parser(add).format_help().split())


def test_written_provenance_is_logged_and_ok(caplog: pytest.LogCaptureFixture) -> None:
    """Zapsaná provenance skončí kódem OK a hláškou na úrovni INFO."""
    with caplog.at_level(logging.INFO):
        code = provenance_exit_code(ProvenanceWritten(), "zapsáno")

    assert code is ExitCode.OK
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (logging.INFO, "zapsáno")
    ]


def test_refused_provenance_logs_reason_and_is_refused(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Odmítnutá provenance skončí kódem REFUSED a důvodem na úrovni ERROR."""
    with caplog.at_level(logging.INFO):
        code = provenance_exit_code(ProvenanceRefused("důvod"), "zapsáno")

    assert code is ExitCode.REFUSED
    assert [(r.levelno, r.getMessage()) for r in caplog.records] == [
        (logging.ERROR, "důvod")
    ]
