"""Konfigurace vzorkovače plánů syntetických zpráv: parametry, ze kterých se plány losují."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from mediparse.domain.labels import LabelModel
from mediparse.domain.note_structure import StructureModel
from mediparse.domain.synthetic_patients import PatientModel


class SamplerConfig(BaseModel):
    """Všechny parametry vzorkovače; každý má zdroj uvedený v podkladu notes-synthesis."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    labels: LabelModel
    patients: PatientModel
    structure: StructureModel
