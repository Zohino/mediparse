"""Plány v repu: bajtově shodné s novým během vzorkovače a čisté vůči hookům repa."""

from pathlib import Path
from random import Random

from mediparse.application.plan_sampling import PlanSampling
from mediparse.infrastructure.plans_file import PlansFile
from mediparse.infrastructure.sampler_config import load_sampler_config

REPOSITORY = Path(__file__).parents[1]
REPO_CONFIG = REPOSITORY / "config" / "synthetic_plan.json"
REPO_PLANS = REPOSITORY / "resources" / "synthetic" / "plans.jsonl"


def test_repository_plans_match_a_fresh_run(tmp_path: Path) -> None:
    """Plány v repu vznikly z configu v repu.

    Soubor vygeneroval konzolový skript v jiném procesu než test, takže shoda
    zachytí i pořadí závislé na solení hashů.
    """
    fresh = tmp_path / "plans.jsonl"
    sampling = PlanSampling(store=PlansFile(fresh), random_source=Random)

    sampling.run(load_sampler_config(REPO_CONFIG))

    assert REPO_PLANS.read_bytes() == fresh.read_bytes()


def test_repository_plans_survive_hygiene_hooks() -> None:
    """Soubor končí jedním novým řádkem, nemá CR ani koncové mezery; hooky ho nepřepíšou."""
    content = REPO_PLANS.read_bytes()

    assert content.endswith(b"\n")
    assert not content.endswith(b"\n\n")
    assert b"\r" not in content
    assert all(line == line.rstrip() for line in content.splitlines())
