"""Plány v repu: bajtově shodné s novým během vzorkovače a čisté vůči hookům repa."""

import csv
from pathlib import Path
from random import Random

import pytest

from mediparse.application.plan_sampling import PlanSampling
from mediparse.domain.labels import Diagnosis
from mediparse.infrastructure.plans_file import (
    LABELS_PATH,
    PLANS_PATH,
    PlansFile,
    SyntheticPlanFiles,
)
from mediparse.infrastructure.sampler_config import load_sampler_config

REPOSITORY = Path(__file__).parents[1]
REPO_CONFIG = REPOSITORY / "config" / "synthetic_plan.json"
REPO_PLANS = REPOSITORY / PLANS_PATH
REPO_LABELS = REPOSITORY / LABELS_PATH


def test_repository_plans_match_a_fresh_run(tmp_path: Path) -> None:
    """Plány v repu vznikly z configu v repu.

    Soubor vygeneroval konzolový skript v jiném procesu než test, takže shoda
    zachytí i pořadí závislé na solení hashů.
    """
    fresh = SyntheticPlanFiles(tmp_path / "plans.jsonl", tmp_path / "labels.csv")
    sampling = PlanSampling(store=fresh, random_source=Random)

    sampling.run(load_sampler_config(REPO_CONFIG))

    assert REPO_PLANS.read_bytes() == fresh.plans.read_bytes()
    assert REPO_LABELS.read_bytes() == fresh.labels.read_bytes()


@pytest.mark.parametrize("path", [REPO_PLANS, REPO_LABELS], ids=["plans", "labels"])
def test_repository_files_survive_hygiene_hooks(path: Path) -> None:
    """Soubor končí jedním novým řádkem, nemá CR ani koncové mezery; hooky ho nepřepíšou."""
    content = path.read_bytes()

    assert content.endswith(b"\n")
    assert not content.endswith(b"\n\n")
    assert b"\r" not in content
    assert all(line == line.rstrip() for line in content.splitlines())


def test_saved_plans_load_back_unchanged(tmp_path: Path) -> None:
    """Plány zapsané do souboru se načtou zpět beze změny a ve stejném pořadí."""
    plans = PlansFile(REPO_PLANS).load()
    copy = PlansFile(tmp_path / "plans.jsonl")

    copy.save(plans)

    assert copy.load() == plans
    assert len(plans) == load_sampler_config(REPO_CONFIG).patients.notes


def test_repository_labels_follow_plans() -> None:
    """Každý řádek labels.csv nese note_id, pacienta a labely svého plánu."""
    with REPO_LABELS.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert [
        (
            row["note_id"],
            int(row["subject_id"]),
            {d for d in Diagnosis if row[d] == "1"},
        )
        for row in rows
    ] == [
        (plan.note_id, plan.subject_id, set(plan.labels))
        for plan in PlansFile(REPO_PLANS).load()
    ]
