"""Zápis manifestu kohorty do JSON a zpětné načtení."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.domain.cohort import (
    CohortCounts,
    CohortManifest,
    FunnelStep,
    FunnelStepName,
)
from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.run_manifest import SourceRevision
from mediparse.infrastructure.cohort_manifest_file import JsonCohortManifestFile

if TYPE_CHECKING:
    from pathlib import Path

COUNT = 40
MANIFEST = CohortManifest(
    cohort_sha256="a" * 64,
    config_sha256="b" * 64,
    sources={"discharge.csv.gz": "c" * 64},
    packages={"python": "3.13.0"},
    source=SourceRevision(commit="d" * 40, dirty=False),
    counts=CohortCounts(
        funnel=tuple(
            FunnelStep(step=step, notes=COUNT, patients=COUNT)
            for step in FunnelStepName
        ),
        positives=dict.fromkeys(Diagnosis, COUNT // 2),
    ),
    invalid_codes=0,
)


def test_round_trip(tmp_path: Path) -> None:
    """Zapsaný JSON se validuje zpět na stejný manifest, odsazení je 2 a končí novým řádkem."""
    path = tmp_path / "results" / "cohort" / "manifest.json"

    JsonCohortManifestFile(path).write(MANIFEST)

    text = path.read_text(encoding="utf-8")
    assert CohortManifest.model_validate_json(text, strict=False) == MANIFEST
    assert text.endswith("}\n")
    assert '\n  "cohort_sha256"' in text
