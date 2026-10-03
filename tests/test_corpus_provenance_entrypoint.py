"""Provenance syntetického korpusu: vzniká jen k čistě auditovanému korpusu a váže se na jeho audit."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from mediparse.domain.corpus_provenance import prompts_sha256
from mediparse.entrypoints.corpus_provenance import run
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.synthetic_corpus import PROVENANCE_NAME, RECORD_NAME
from tests.support import COMMIT, NOTE, REPOSITORY_CONFIG, SHORT_NOTE

if TYPE_CHECKING:
    from pathlib import Path

    from mediparse.domain.note_plan import NotePlan
    from mediparse.domain.synthetic_plan import SamplerConfig
    from tests.conftest import AuditFiles


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_provenance_records_generation_and_audit(
    audit_files: AuditFiles,
    sampler_config: SamplerConfig,
    verbalization_template: str,
    repository_plans: tuple[NotePlan, ...],
) -> None:
    """Záznam nese argumenty, seed, otisky configu a zadání a otisk záznamu auditu."""
    root = audit_files.audited_corpus(SHORT_NOTE)

    assert run(audit_files.provenance_argv(root)) == ExitCode.OK

    record = json.loads((root / PROVENANCE_NAME).read_text(encoding="utf-8"))
    assert record == {
        "generation": {
            "seed": sampler_config.seed,
            "sampler_config_sha256": _sha256(REPOSITORY_CONFIG),
            "prompts_sha256": prompts_sha256(
                verbalization_template, repository_plans, sampler_config
            ),
            "model": "claude-opus-5-5",
            "claude_code_version": "2.1.5",
            "generated_on": "2026-10-10",
            "specification_commit": COMMIT,
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
    root = audit_files.audited_corpus(SHORT_NOTE)
    (root / RECORD_NAME).unlink()

    assert run(audit_files.provenance_argv(root)) == ExitCode.REFUSED
    assert not (root / PROVENANCE_NAME).exists()


def test_provenance_refuses_corpus_changed_after_audit(
    audit_files: AuditFiles,
) -> None:
    """Korpus, který by neprošel bránou, provenance nedostane."""
    root = audit_files.audited_corpus(SHORT_NOTE)
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
    audit_files: AuditFiles,
    capsys: pytest.CaptureFixture[str],
    name: str,
    value: str,
) -> None:
    """Neplatný model, verze nebo commit odmítne už parsování argumentů hláškou domény."""
    root = audit_files.audited_corpus(SHORT_NOTE)

    with pytest.raises(SystemExit) as raised:
        run(audit_files.provenance_argv(root, **{name: value}))

    assert raised.value.code == ExitCode.REFUSED
    assert f"--{name}" in capsys.readouterr().err
    assert not (root / PROVENANCE_NAME).exists()


def test_provenance_rejects_invalid_date(audit_files: AuditFiles) -> None:
    """Den mimo tvar RRRR-MM-DD odmítne už argparse."""
    root = audit_files.audited_corpus(SHORT_NOTE)

    with pytest.raises(SystemExit):
        run(audit_files.provenance_argv(root, date="10. 10. 2026"))
