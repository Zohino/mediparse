"""Provenance syntetického korpusu: vzniká jen k čistě auditovanému korpusu a váže se na jeho audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest

from mediparse.entrypoints.corpus_provenance import run
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.sampler_config import SAMPLER_CONFIG_PATH
from mediparse.infrastructure.synthetic_corpus import PROVENANCE_NAME, RECORD_NAME
from mediparse.infrastructure.verbalization_template import (
    VERBALIZATION_TEMPLATE_PATH,
)

if TYPE_CHECKING:
    from tests.conftest import AuditFiles

REPOSITORY: Final = Path(__file__).parents[1]
CONFIG: Final = REPOSITORY / SAMPLER_CONFIG_PATH
TEMPLATE: Final = REPOSITORY / VERBALIZATION_TEMPLATE_PATH
NOTE: Final = "en/90000001-DS-1.txt"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_provenance_records_generation_and_audit(audit_files: AuditFiles) -> None:
    """Záznam nese argumenty, seed a otisky configu i šablony a otisk záznamu auditu."""
    root = audit_files.audited_corpus({NOTE: "a short synthetic note"})

    assert run(audit_files.provenance_argv(root)) == ExitCode.OK

    record = json.loads((root / PROVENANCE_NAME).read_text(encoding="utf-8"))
    assert record == {
        "generation": {
            "seed": json.loads(CONFIG.read_text(encoding="utf-8"))["seed"],
            "sampler_config_sha256": _sha256(CONFIG),
            "verbalization_template_sha256": _sha256(TEMPLATE),
            "model": "claude-opus-5-5",
            "claude_code_version": "2.1.5",
            "generated_on": "2026-10-10",
            "specification_commit": "c" * 40,
        },
        "audit_sha256": _sha256(root / RECORD_NAME),
    }


def test_provenance_refuses_empty_corpus(audit_files: AuditFiles) -> None:
    """Korpus bez zpráv nemá co popsat."""
    root = audit_files.corpus({})
    audit_files.pin([])

    assert run(audit_files.provenance_argv(root)) == ExitCode.REFUSED
    assert not (root / PROVENANCE_NAME).exists()


def test_provenance_refuses_unaudited_corpus(audit_files: AuditFiles) -> None:
    """Provenance vzniká až po čistém auditu."""
    root = audit_files.audited_corpus({NOTE: "a short synthetic note"})
    (root / RECORD_NAME).unlink()

    assert run(audit_files.provenance_argv(root)) == ExitCode.REFUSED
    assert not (root / PROVENANCE_NAME).exists()


def test_provenance_refuses_corpus_changed_after_audit(
    audit_files: AuditFiles,
) -> None:
    """Korpus, který by neprošel bránou, provenance nedostane."""
    root = audit_files.audited_corpus({NOTE: "a short synthetic note"})
    (root / NOTE).write_text("a regenerated synthetic note", encoding="utf-8")

    assert run(audit_files.provenance_argv(root)) == ExitCode.REFUSED
    assert not (root / PROVENANCE_NAME).exists()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("model", "gpt-4o"),
        ("claude-code-version", "2.1"),
        ("specification-commit", "HEAD"),
    ],
)
def test_provenance_refuses_invalid_generation(
    audit_files: AuditFiles, name: str, value: str
) -> None:
    """Neplatný model, verze nebo commit se do záznamu nedostane."""
    root = audit_files.audited_corpus({NOTE: "a short synthetic note"})

    assert run(audit_files.provenance_argv(root, **{name: value})) == ExitCode.REFUSED
    assert not (root / PROVENANCE_NAME).exists()


def test_provenance_rejects_invalid_date(audit_files: AuditFiles) -> None:
    """Den mimo tvar RRRR-MM-DD odmítne už argparse."""
    root = audit_files.audited_corpus({NOTE: "a short synthetic note"})

    with pytest.raises(SystemExit):
        run(audit_files.provenance_argv(root, date="10. 10. 2026"))
