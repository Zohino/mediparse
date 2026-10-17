"""Převod výsledku provenance na návratový kód a hlášku v logu."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from mediparse.application.corpus_provenance import (
    ProvenanceRefused,
    ProvenanceWritten,
)
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.entrypoints.provenance_exit import provenance_exit_code

if TYPE_CHECKING:
    import pytest


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
