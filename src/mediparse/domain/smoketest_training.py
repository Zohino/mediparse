"""Trénink smoketestu: config, odložení po pacientech a held-out metriky jedné diagnózy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.labels import Diagnosis

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.domain.smoketest_input import InputNote


class TrainingConfig(BaseModel):
    """Parametry tréninku: diagnóza, seed odložení, počet foldů a síla regularizace."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    diagnosis: Diagnosis
    seed: int
    folds: Annotated[int, Field(ge=2)]
    regularization: Annotated[float, Field(gt=0.0)]


@dataclass(frozen=True)
class BinaryMetrics:
    """Held-out metriky binárního klasifikátoru pro pozitivní třídu."""

    precision: float
    recall: float
    f1: float
    accuracy: float


@dataclass(frozen=True)
class Holdout:
    """Indexy zpráv v tréninkové a testovací části."""

    train: tuple[int, ...]
    test: tuple[int, ...]


@dataclass(frozen=True)
class TrainingReport:
    """Výsledek běhu: velikosti částí, počet pozitivních v testu a metriky."""

    diagnosis: Diagnosis
    train_notes: int
    test_notes: int
    test_positives: int
    metrics: BinaryMetrics


class SubjectLeakError(ValueError):
    """Jeden pacient je současně v tréninkové a testovací části."""


def ensure_disjoint_subjects(notes: Sequence[InputNote], holdout: Holdout) -> None:
    """Ověří, že žádný pacient není v tréninku i testu.

    Args:
        notes: Zprávy, na které odkazují indexy odložení.
        holdout: Rozdělení indexů zpráv.

    Raises:
        SubjectLeakError: Některý pacient je v obou částech.
    """
    shared = {notes[index].subject_id for index in holdout.train} & {
        notes[index].subject_id for index in holdout.test
    }
    if shared:
        msg = f"Pacienti v tréninku i testu: {', '.join(map(str, sorted(shared)))}."
        raise SubjectLeakError(msg)


def ensure_patients_per_class(
    labels: Sequence[bool], groups: Sequence[int], diagnosis: Diagnosis, folds: int
) -> None:
    """Ověří, že každá třída diagnózy má aspoň tolik pacientů jako foldů.

    Args:
        labels: Label diagnózy každé zprávy.
        groups: Pacient každé zprávy.
        diagnosis: Diagnóza, na které se trénuje.
        folds: Počet foldů odložení.

    Raises:
        InvalidInputError: Některá třída má méně pacientů než foldů.
    """
    positive = {group for label, group in zip(labels, groups, strict=True) if label}
    negative = {group for label, group in zip(labels, groups, strict=True) if not label}
    if min(len(positive), len(negative)) < folds:
        msg = (
            f"Diagnóza {diagnosis}: {len(positive)} pozitivních a {len(negative)} "
            f"negativních pacientů, odložení potřebuje v každé třídě aspoň {folds}."
        )
        raise InvalidInputError(msg)
