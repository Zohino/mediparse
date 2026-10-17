"""Plány v repu: bajtově totožné s novým během vzorkovače a čisté vůči hookům repa."""

from __future__ import annotations

import csv
from random import Random
from typing import TYPE_CHECKING

import pytest

from mediparse.application.plan_sampling import PlanSampling
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.plans_file import (
    PlansFile,
    SyntheticPlanFiles,
)
from tests.support import REPOSITORY_LABELS, REPOSITORY_PLANS

if TYPE_CHECKING:
    from pathlib import Path

    from mediparse.domain.note_plan import NotePlan
    from mediparse.domain.synthetic_plan import SamplerConfig


def test_repository_plans_match_a_fresh_run(
    sampler_config: SamplerConfig, tmp_path: Path
) -> None:
    """Plány v repu vznikly z configu v repu.

    Soubor vygeneroval konzolový skript v jiném procesu než test, takže porovnání
    zachytí i pořadí závislé na solení hashů.
    """
    fresh = SyntheticPlanFiles(tmp_path / "plans.jsonl", tmp_path / "labels.csv")
    sampling = PlanSampling(store=fresh, random_source=Random)

    sampling.run(sampler_config)

    assert REPOSITORY_PLANS.read_bytes() == fresh.plans.read_bytes()
    assert REPOSITORY_LABELS.read_bytes() == fresh.labels.read_bytes()


@pytest.mark.parametrize(
    "path", [REPOSITORY_PLANS, REPOSITORY_LABELS], ids=["plans", "labels"]
)
def test_repository_files_survive_hygiene_hooks(path: Path) -> None:
    """Soubor končí jedním novým řádkem, nemá CR ani koncové mezery; hooky ho nepřepíšou."""
    content = path.read_bytes()

    assert content.endswith(b"\n")
    assert not content.endswith(b"\n\n")
    assert b"\r" not in content
    assert all(line == line.rstrip() for line in content.splitlines())


def test_saved_plans_load_back_unchanged(
    sampler_config: SamplerConfig,
    repository_plans: tuple[NotePlan, ...],
    tmp_path: Path,
) -> None:
    """Plány zapsané do souboru se načtou zpět beze změny a ve stejném pořadí."""
    copy = PlansFile(tmp_path / "plans.jsonl")

    copy.save(repository_plans)

    assert copy.load() == repository_plans
    assert len(repository_plans) == sampler_config.patients.notes


def test_plan_of_other_patient_than_note_id_fails_to_load(
    repository_plans: tuple[NotePlan, ...], tmp_path: Path
) -> None:
    """Soubor s plánem, jehož subject_id neodpovídá note_id, se nenačte."""
    broken = repository_plans[0].model_copy(
        update={"subject_id": repository_plans[0].subject_id + 1}
    )
    copy = PlansFile(tmp_path / "plans.jsonl")
    copy.save([broken])

    with pytest.raises(InvalidInputError, match="nepatří pacientovi"):
        copy.load()


def test_repository_labels_follow_plans(
    repository_plans: tuple[NotePlan, ...],
) -> None:
    """Každý řádek labels.csv nese note_id, pacienta a labely svého plánu."""
    with REPOSITORY_LABELS.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert [
        (
            row["note_id"],
            int(row["subject_id"]),
            {d for d in Diagnosis if row[d] == "1"},
        )
        for row in rows
    ] == [
        (plan.note_id, plan.subject_id, set(plan.labels)) for plan in repository_plans
    ]


def test_blank_lines_between_and_after_plans_are_skipped(
    repository_plans: tuple[NotePlan, ...], tmp_path: Path
) -> None:
    """Prázdný řádek uprostřed i na konci souboru nenese záznam a plány se načtou celé."""
    path = tmp_path / "plans.jsonl"
    first, second = repository_plans[:2]
    path.write_text(
        f"{first.model_dump_json()}\n\n{second.model_dump_json()}\n\n",
        encoding="utf-8",
    )

    assert PlansFile(path).load() == (first, second)
