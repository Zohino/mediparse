"""Vstupní bod vzorkovače plánů: z configu v repu zapíše plán pro každou zprávu."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.domain.note_plan import NotePlan
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.entrypoints.synthetic_plans import run
from tests.support import REPOSITORY_CONFIG

if TYPE_CHECKING:
    from pathlib import Path

    from mediparse.domain.synthetic_plan import SamplerConfig


def test_entrypoint_writes_one_valid_plan_per_note(
    tmp_path: Path, sampler_config: SamplerConfig
) -> None:
    """Každý řádek výstupu je platný plán, k němu řádek labelů; tolik, kolik má korpus zpráv."""
    output = tmp_path / "plans.jsonl"
    labels = tmp_path / "labels.csv"

    code = run([
        "--config",
        str(REPOSITORY_CONFIG),
        "--output",
        str(output),
        "--labels",
        str(labels),
    ])

    plans = [
        NotePlan.model_validate_json(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    notes = sampler_config.patients.notes
    assert code == ExitCode.OK
    assert len(plans) == notes
    assert len(labels.read_text(encoding="utf-8").splitlines()) == notes + 1
