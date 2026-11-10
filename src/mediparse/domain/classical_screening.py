"""Mřížka klasického screeningu a výsledky buněk na vývojové části."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from mediparse.domain.model_input import ModelInput
from mediparse.domain.seed import Seed

if TYPE_CHECKING:
    from mediparse.domain.diagnosis import Diagnosis
    from mediparse.domain.evaluation import BinaryMetrics, Classification
    from mediparse.domain.split import SplitConfig


class ClassifierFamily(StrEnum):
    """Rodiny klasického stupně s výchozími hyperparametry."""

    COMPLEMENT_NB = "complement_nb"
    LOGREG = "logreg"
    LINEAR_SVM = "linear_svm"
    NB_SVM = "nb_svm"
    LIGHTGBM_LSA = "lightgbm_lsa"


class VectorizerSpec(BaseModel):
    """Pojmenovaný vektorizér s analyzátorem a rozsahem n-gramů."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    name: Annotated[str, Field(min_length=1)]
    analyzer: Literal["word", "char_wb"]
    ngram_range: tuple[PositiveInt, PositiveInt]

    @model_validator(mode="after")
    def _ngram_range_is_ordered(self) -> Self:
        if self.ngram_range[0] > self.ngram_range[1]:
            msg = "Minimum rozsahu n-gramů nesmí překročit maximum."
            raise ValueError(msg)
        return self


class ClassicalScreeningConfig(BaseModel):
    """Parametry vnější CV, volby prahu a mřížky klasického screeningu."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    seed: Seed
    folds: Annotated[int, Field(ge=2)]
    threshold_folds: Annotated[int, Field(ge=2)]
    lsa_components: PositiveInt
    input: ModelInput
    vectorizers: Annotated[tuple[VectorizerSpec, ...], Field(min_length=1)]
    families: Annotated[tuple[ClassifierFamily, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _vectorizer_names_are_unique(self) -> Self:
        if len({spec.name for spec in self.vectorizers}) != len(self.vectorizers):
            msg = "Jména vektorizérů screeningu musí být jedinečná."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _families_are_unique(self) -> Self:
        if len(set(self.families)) != len(self.families):
            msg = "Rodiny klasifikátorů screeningu musí být jedinečné."
            raise ValueError(msg)
        return self


@dataclass(frozen=True, kw_only=True)
class ScreeningResult:
    """Predikce a metriky jedné buňky mřížky pro diagnózu a vnější fold."""

    fold: int
    vectorizer: str
    family: ClassifierFamily
    diagnosis: Diagnosis
    note_ids: tuple[str, ...]
    labels: tuple[bool, ...]
    classifications: tuple[Classification, ...]
    threshold: float
    converged: bool
    metrics: BinaryMetrics

    def __post_init__(self) -> None:
        """Ověří společnou délku identit, labelů a predikcí.

        Raises:
            ValueError: N-tice buňky mají různé délky.
        """
        if not (len(self.note_ids) == len(self.labels) == len(self.classifications)):
            msg = "Identity, labely a predikce buňky musí mít stejné délky."
            raise ValueError(msg)

    @property
    def test_notes(self) -> int:
        """Počet testovacích zpráv buňky."""
        return len(self.labels)

    @property
    def test_positives(self) -> int:
        """Počet pozitivních testovacích zpráv buňky."""
        return sum(self.labels)


@dataclass(frozen=True)
class ScreeningReport:
    """Velikost vývojové části a původní split pro provenance screeningu."""

    development_notes: int
    development_patients: int
    split_config: SplitConfig
    split_sha256: str
