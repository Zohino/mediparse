"""Sdílené fixtures testů: syntetický korpus, referenční soubory MIMIC-IV-Note a config tabulek."""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest

from mediparse.domain.labels import Diagnosis
from mediparse.domain.mentions import MentionStatus
from mediparse.domain.note_plan import NotePlan, PlannedMention
from mediparse.domain.note_structure import Sex
from mediparse.entrypoints import corpus_audit, corpus_provenance
from mediparse.entrypoints.exit_code import ExitCode
from mediparse.infrastructure.mimic_reference import MimicReference
from mediparse.infrastructure.sampler_config import load_sampler_config
from mediparse.infrastructure.verbalization_template import (
    VERBALIZATION_TEMPLATE_PATH,
    load_verbalization_template,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mediparse.domain.synthetic_plan import SamplerConfig

type Rows = Sequence[tuple[str, str]]

REPOSITORY_CONFIG: Final = Path(__file__).parents[1] / "config" / "synthetic_plan.json"
REPOSITORY_TEMPLATE: Final = Path(__file__).parents[1] / VERBALIZATION_TEMPLATE_PATH

COMMIT: Final = "c" * 40
HEADER: Final = (
    "note_id",
    "subject_id",
    "hadm_id",
    "note_type",
    "note_seq",
    "charttime",
    "storetime",
    "text",
)
RADIOLOGY_ROWS: Final = (("10000033", "an unrelated radiology report"),)
HOSP: Final = {
    "url": "https://physionet.org/files/mimiciv/3.1/hosp/admissions.csv.gz",
    "sha256": "f" * 64,
}


_PLAN: Final = NotePlan(
    note_id="90000001-DS-1",
    subject_id=90000001,
    labels=(Diagnosis.CKD,),
    sex=Sex.FEMALE,
    age_marker=True,
    sections=(
        "allergies",
        "chief_complaint",
        "history_of_present_illness",
        "past_medical_history",
        "social_history",
        "discharge_disposition",
        "discharge_diagnosis",
        "followup_instructions",
    ),
    subheadings=("Facility",),
    narrative_words=28,
    section_words={"history_of_present_illness": 28},
    narrative_deid=1,
    mentions=(
        PlannedMention(
            diagnosis=Diagnosis.CKD,
            status=MentionStatus.AFFIRMED,
            sections=("past_medical_history", "discharge_diagnosis"),
        ),
    ),
)
_NOTE: Final = """\
Name: ___  Unit No: ___
Admission Date: ___  Discharge Date: ___
Date of Birth: ___  Sex: F
Service: MEDICINE  Attending: ___

Allergies:
No Known Allergies / Adverse Drug Reactions

Chief Complaint:
Fatigue

History of Present Illness:
Ms. ___ is a ___ year old woman who presented with two days of fatigue and poor
oral intake. She was given intravenous fluids in the emergency department.

Past Medical History:
Chronic kidney disease, stage 3

Social History:
___

Discharge Disposition:
Home With Service
Facility:
___

Discharge Diagnosis:
Chronic kidney disease, dehydration

Followup Instructions:
___
"""


@dataclass(frozen=True)
class PlannedNote:
    """Plán syntetické zprávy a text, který ho přesně dodržuje."""

    plan: NotePlan
    text: str


@dataclass(frozen=True)
class AuditFiles:
    """Korpus, reference a config tabulek pro audit a bránu v dočasném adresáři testu."""

    root: Path

    @property
    def tables(self) -> Path:
        """Config tabulek MIMIC, do kterého se reference připínají."""
        return self.root / "mimic_tables.json"

    def corpus(self, notes: Mapping[str, str]) -> Path:
        """Zapíše zprávy korpusu do repozitáře.

        Returns:
            Kořen korpusu uvnitř repozitáře.
        """
        repository = self.root / "repo"
        (repository / ".git").mkdir(parents=True, exist_ok=True)
        root = repository / "resources" / "synthetic"
        for name, text in notes.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return root

    def reference(self, name: str, rows: Rows) -> Path:
        """Zapíše referenční soubor ve tvaru tabulky MIMIC-IV-Note.

        Returns:
            Cesta k souboru ``.csv.gz``.
        """
        path = self.root / name
        with gzip.open(path, mode="wt", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(HEADER)
            writer.writerows(
                [f"{subject}-DS-1", subject, "1", "DS", "1", "", "", text]
                for subject, text in rows
            )
        return path

    def pin(self, references: Sequence[Path]) -> Path:
        """Zapíše config, který připíná otisky referencí; tabulka hosp do reference nepatří.

        Returns:
            Cesta ke configu tabulek.
        """
        entries = [
            {
                "url": f"https://physionet.org/files/mimic-iv-note/2.2/note/{path.name}",
                "sha256": MimicReference(path).sha256(),
            }
            for path in references
        ]
        entries.append(HOSP)
        self.tables.write_text(json.dumps({"mimic_tables": entries}), encoding="utf-8")
        return self.tables

    def pinned_references(self, rows: Rows) -> tuple[Path, Path]:
        """Discharge se zadanými řádky a neutrální radiology, obě připnuté v configu.

        Returns:
            Cesty k discharge a radiology.
        """
        references = (
            self.reference("discharge.csv.gz", rows),
            self.reference("radiology.csv.gz", RADIOLOGY_ROWS),
        )
        self.pin(references)
        return references

    def audited_corpus(self, notes: Mapping[str, str]) -> Path:
        """Zapíše korpus a projde ho čistým auditem proti připnuté referenci.

        Returns:
            Kořen korpusu se záznamem auditu.
        """
        root = self.corpus(notes)
        references = self.pinned_references([("10000032", "a reference text")])
        argv = self.audit_argv(root, references, self.root / "report.json")
        assert corpus_audit.run(argv, {}) == ExitCode.OK
        return root

    def released_corpus(self, notes: Mapping[str, str]) -> Path:
        """Zapíše korpus, projde ho čistým auditem a zapíše k němu provenance.

        Returns:
            Kořen korpusu, který smí do repozitáře.
        """
        root = self.audited_corpus(notes)
        assert corpus_provenance.run(self.provenance_argv(root)) == ExitCode.OK
        return root

    def provenance_argv(self, corpus: Path, **overrides: str) -> list[str]:
        """Argumenty provenance korpusu s configem a šablonou z repa.

        Returns:
            Argumenty vstupního bodu provenance; ``overrides`` nahradí údaje o generování.
        """
        generation = {
            "model": "claude-opus-5-5",
            "claude-code-version": "2.1.5",
            "date": "2026-10-10",
            "specification-commit": COMMIT,
        } | overrides
        return [
            *(f"--{name}={value}" for name, value in generation.items()),
            f"--corpus={corpus}",
            f"--config={REPOSITORY_CONFIG}",
            f"--template={REPOSITORY_TEMPLATE}",
            f"--tables={self.tables}",
        ]

    def audit_argv(
        self, corpus: Path, references: Sequence[Path], report: Path
    ) -> list[str]:
        """Argumenty auditu korpusu proti referencím a tomuto configu.

        Returns:
            Argumenty vstupního bodu auditu.
        """
        return [
            "--corpus",
            str(corpus),
            *(f"--reference={path}" for path in references),
            "--tables",
            str(self.tables),
            "--report",
            str(report),
            "--commit",
            COMMIT,
        ]


@pytest.fixture
def audit_files(tmp_path: Path) -> AuditFiles:
    """Soubory pro audit a bránu v dočasném adresáři testu.

    Returns:
        Továrna na korpus, reference a config tabulek.
    """
    return AuditFiles(tmp_path)


@pytest.fixture
def planned_note() -> PlannedNote:
    """Malý plán s jedinou zmínkou CKD a zpráva, která mu odpovídá.

    Returns:
        Plán a text pro kontroly shody.
    """
    return PlannedNote(_PLAN, _NOTE)


@pytest.fixture(scope="session")
def sampler_config() -> SamplerConfig:
    """Config vzorkovače v repu: hlavičky sekcí a klíčová slova diagnóz.

    Returns:
        Config načtený jednou za běh testů.
    """
    return load_sampler_config(REPOSITORY_CONFIG)


@pytest.fixture(scope="session")
def verbalization_template() -> str:
    """Šablona instrukcí verbalizace v repu.

    Returns:
        Text šablony načtený jednou za běh testů.
    """
    return load_verbalization_template(REPOSITORY_TEMPLATE)
