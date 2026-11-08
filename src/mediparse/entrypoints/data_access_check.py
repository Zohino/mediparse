"""Konzolová stráž přístupu ke skutečným datům MIMIC."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from mediparse.domain.data_access import blocked_reason
from mediparse.entrypoints.cli import configure_logging, refusing_invalid_input
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.data_access_rules import load_data_access_rules

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


@refusing_invalid_input
def run(environ: Mapping[str, str]) -> ExitCode:
    """Ověří prostředí podle sdílených pravidel přístupu.

    Returns:
        OK bez blokace, BLOCKED s důvodem v logu, REFUSED při neplatných pravidlech.
    """
    reason = blocked_reason(environ, load_data_access_rules())
    if reason is not None:
        logger.error("%s", reason)
        return ExitCode.BLOCKED
    return ExitCode.OK


def main() -> ExitCode:
    """Nastaví diagnostiku a ověří prostředí procesu.

    Returns:
        Návratový kód kontroly přístupu.
    """
    configure_logging()
    return run(os.environ)
