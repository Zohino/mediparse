"""Vstupní bod vzorkovače plánů: z configu v repu zapíše plán pro každou zprávu."""

from pathlib import Path

from mediparse.domain.note_plan import NotePlan
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.entrypoints.synthetic_plans import run
from mediparse.infrastructure.sampler_config import load_sampler_config

REPO_CONFIG = Path(__file__).parents[1] / "config" / "synthetic_plan.json"


def test_entrypoint_writes_one_valid_plan_per_note(tmp_path: Path) -> None:
    """Každý řádek výstupu je platný plán a řádků je tolik, kolik má korpus zpráv."""
    output = tmp_path / "plans.jsonl"

    code = run(["--config", str(REPO_CONFIG), "--output", str(output)])

    lines = output.read_text(encoding="utf-8").splitlines()
    assert code == ExitCode.OK
    assert len(lines) == load_sampler_config(REPO_CONFIG).patients.notes
    assert all(NotePlan.model_validate_json(line) for line in lines)
