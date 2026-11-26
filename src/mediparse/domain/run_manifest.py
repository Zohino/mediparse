"""Run manifest: z čeho a čím běh vznikl."""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    field_validator,
    model_validator,
)

from mediparse.domain.clinical_text import TextNormalization
from mediparse.domain.digest import Sha256
from mediparse.domain.environment import Environment, Hardware
from mediparse.domain.identity import RowId
from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.model_input import TextPreprocessing
from mediparse.domain.representation import EncoderSpec, RepresentationName
from mediparse.domain.seed import Seed
from mediparse.domain.stage_tuning import BRANCH_KEY
from mediparse.domain.tuning import ParamValue


class DiagnosticStrip(BaseModel):
    """Záznam ablace D2: které sekce se ze zpráv odstranily a podle jakého lexikonu.

    ``sections`` jsou seřazené klíče sekcí bez opakování, aby byl JSON deterministický.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    sections: Annotated[tuple[str, ...], Field(min_length=1)]
    lexicon_sha256: Sha256

    @field_validator("sections", mode="before")
    @classmethod
    def _sorted_unique(cls, sections: object) -> object:
        if not isinstance(sections, (tuple, list)):
            return sections
        return tuple(sorted(set(sections)))


def _ensure_normalization_matches_preprocessing(
    preprocessing: TextPreprocessing, normalization: TextNormalization | None
) -> None:
    if preprocessing.normalizes != (normalization is not None):
        msg = "Úroveň předzpracování none musí odpovídat nepřítomnému slovníku normalizace."
        raise ValueError(msg)
    if (
        not preprocessing.rewrites_variants
        and normalization is not None
        and normalization.variants
    ):
        msg = "Úroveň předzpracování normalized nesmí obsahovat varianty diagnóz."
        raise ValueError(msg)


class RunManifest(BaseModel):
    """Provenance tréninku; funkce vstupů a prostředí, bez času běhu.

    ``config_sha256`` je hash configu tréninku, ``seed`` seed modelu. Split
    nese ``split_seed``, ``split_folds`` a ``split_sha256``, otisk přiřazení pacientů
    do foldů a vývojové části.
    ``diagnostic_strip`` je ``None``, když se zprávy nestripovaly.
    ``text_preprocessing`` zaznamenává úroveň předzpracování vstupu TF-IDF.
    ``text_normalization`` nese slovníky předané trenérovi, při úrovni ``none`` je ``None``.
    Úroveň ``normalized`` používá jen zkratky, s prázdným slovníkem variant.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    row_id: RowId
    dataset_sha256: Sha256
    predictions_sha256: Sha256
    config_sha256: Sha256
    parameters: dict[str, ParamValue]
    threshold: Annotated[float, Field(allow_inf_nan=False)]
    threshold_folds: Annotated[int, Field(ge=2)]
    converged: bool
    seed: Seed
    split_seed: Seed
    split_folds: Annotated[int, Field(ge=2)]
    split_sha256: Sha256
    diagnostic_strip: DiagnosticStrip | None
    text_preprocessing: TextPreprocessing
    text_normalization: TextNormalization | None
    model: str
    model_revision: str | None
    tokenizer_revision: str | None
    environment: Environment
    hardware: Hardware

    @model_validator(mode="after")
    def _normalization_matches_preprocessing(self) -> Self:
        _ensure_normalization_matches_preprocessing(
            self.text_preprocessing, self.text_normalization
        )
        return self


class EvaluationManifest(BaseModel):
    """Provenance evaluace: ověřené vstupy, otisk metrik a prostředí evaluátoru."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    row_id: RowId
    training_manifest_sha256: Sha256
    predictions_sha256: Sha256
    dataset_sha256: Sha256
    split_sha256: Sha256
    metrics_sha256: Sha256
    environment: Environment


def ensure_training_lineage(
    training: RunManifest,
    *,
    predictions_sha256: str,
    dataset_sha256: str,
    split_sha256: str,
) -> None:
    """Ověří, že vstupy evaluace pocházejí z dodaného tréninku.

    Raises:
        InvalidInputError: Otisky nesedí; zpráva uvádí všechna pole s rozdílem.
    """
    actual = {
        "predictions_sha256": predictions_sha256,
        "dataset_sha256": dataset_sha256,
        "split_sha256": split_sha256,
    }
    mismatched = [
        field for field, digest in actual.items() if getattr(training, field) != digest
    ]
    if mismatched:
        msg = f"Vstupy evaluace neodpovídají tréninku: {', '.join(mismatched)}."
        raise InvalidInputError(msg)


class StageTuningManifest(BaseModel):
    """Provenance CASH stupně, vítěz a otisky obou výstupů.

    ``jobs`` je počet procesů posledního běhu; mění dobu pokusů, ne výsledky.
    ``text_preprocessing`` mapuje jména reprezentací na úrovně vstupu.
    ``text_normalization`` nese pro stejná jména skutečně použité slovníky.
    """

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", allow_inf_nan=False
    )

    encoders: tuple[EncoderSpec, ...] = ()
    dataset_sha256: Sha256
    config_sha256: Sha256
    split_seed: Seed
    split_folds: Annotated[int, Field(ge=2)]
    split_sha256: Sha256
    seed: Seed
    jobs: PositiveInt
    trials: PositiveInt
    random_startup_trials: Annotated[int, Field(ge=0)]
    completed_trials: PositiveInt
    nonconverged_trials: Annotated[int, Field(ge=0)]
    best_trial: Annotated[int, Field(ge=0)]
    best_branch: str
    best_params: dict[str, ParamValue]
    best_macro_f1: Annotated[float, Field(ge=0, le=1)]
    sampler: str
    pruner: str
    seconds: Annotated[float, Field(ge=0)]
    sample_notes: PositiveInt
    sample_patients: PositiveInt
    development_notes: PositiveInt
    development_patients: PositiveInt
    results_sha256: Sha256
    predictions_sha256: Sha256
    diagnostic_strip: DiagnosticStrip | None
    text_preprocessing: Annotated[
        dict[RepresentationName, TextPreprocessing], Field(min_length=1)
    ]
    text_normalization: dict[RepresentationName, TextNormalization | None]
    environment: Environment
    hardware: Hardware

    @model_validator(mode="after")
    def _consistent_stage(self) -> Self:
        if self.completed_trials != self.trials:
            msg = "Dokončené pokusy musí vyčerpat rozpočet stupně."
            raise ValueError(msg)
        if self.text_preprocessing.keys() != self.text_normalization.keys():
            msg = "Mapy předzpracování a normalizace manifestu stupně musí mít stejné reprezentace."
            raise ValueError(msg)
        for name, level in self.text_preprocessing.items():
            _ensure_normalization_matches_preprocessing(
                level, self.text_normalization[name]
            )
        return self


# Jednotlivá pole odpovídají nezávislým vstupům provenance předkola.
def ensure_stage_lineage(  # ruff: ignore[too-many-arguments]
    manifest: StageTuningManifest,
    *,
    config_sha256: str,
    results_sha256: str,
    dataset_sha256: str,
    split_sha256: str,
    diagnostic_strip: DiagnosticStrip | None,
    text_normalization: dict[str, TextNormalization | None],
    best_trial: int,
) -> None:
    """Ověří vstupy a vítěze předkola před kontrolou kandidátů.

    Raises:
        InvalidInputError: Původ nesedí; zpráva uvádí všechna rozdílná pole.
    """
    actual = {
        "config_sha256": config_sha256,
        "results_sha256": results_sha256,
        "dataset_sha256": dataset_sha256,
        "split_sha256": split_sha256,
        "diagnostic_strip": diagnostic_strip,
        "text_normalization": text_normalization,
        "best_trial": best_trial,
    }
    mismatched = [
        field for field, value in actual.items() if getattr(manifest, field) != value
    ]
    if mismatched:
        msg = f"Manifest předkola neodpovídá vstupům doladění: {', '.join(mismatched)}."
        raise InvalidInputError(msg)


class RefinementCandidate(BaseModel):
    """Bod rodiny z předkola a jeho cíl při kontrole na celé vývojové části."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", allow_inf_nan=False
    )

    stage_trial: Annotated[int, Field(ge=0)]
    branch: str
    params: dict[str, ParamValue]
    stage_macro_f1: Annotated[float, Field(ge=0, le=1)]
    check_macro_f1: Annotated[float, Field(ge=0, le=1)]

    @model_validator(mode="after")
    def _matching_branch(self) -> Self:
        if self.params.get(BRANCH_KEY) != self.branch:
            msg = "Větev kandidáta musí odpovídat jeho parametrům."
            raise ValueError(msg)
        return self


class StageRefinementManifest(BaseModel):
    """Provenance doladění a vítěz stupně; cíl ladění není odhad výkonu modelu."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", allow_inf_nan=False
    )

    encoders: tuple[EncoderSpec, ...] = ()
    dataset_sha256: Sha256
    stage_config_sha256: Sha256
    refinement_config_sha256: Sha256
    matrix_sha256: Sha256
    stage_manifest_sha256: Sha256
    stage_results_sha256: Sha256
    split_seed: Seed
    split_folds: Annotated[int, Field(ge=2)]
    split_sha256: Sha256
    seed: Seed
    folds: Annotated[int, Field(ge=2)]
    jobs: PositiveInt
    tuning_trials: PositiveInt
    random_startup_trials: Annotated[int, Field(ge=0)]
    candidates: tuple[RefinementCandidate, RefinementCandidate]
    best_trial: Annotated[int, Field(ge=0)]
    best_params: dict[str, ParamValue]
    best_macro_f1: Annotated[float, Field(ge=0, le=1)]
    completed_trials: PositiveInt
    nonconverged_trials: Annotated[int, Field(ge=0)]
    sampler: str
    pruner: str
    seconds: Annotated[float, Field(ge=0)]
    development_notes: PositiveInt
    development_patients: PositiveInt
    results_sha256: Sha256
    predictions_sha256: Sha256
    diagnostic_strip: DiagnosticStrip | None
    text_preprocessing: Annotated[
        dict[RepresentationName, TextPreprocessing], Field(min_length=1)
    ]
    text_normalization: dict[RepresentationName, TextNormalization | None]
    environment: Environment
    hardware: Hardware

    @property
    def winner_branch(self) -> str:
        """Větev s vyšším cílem kontroly; při stejném cíli dřívější kandidát."""
        return max(self.candidates, key=lambda c: c.check_macro_f1).branch

    @model_validator(mode="after")
    def _consistent_refinement(self) -> Self:
        if self.completed_trials != self.tuning_trials + 1:
            msg = "Dokončené pokusy musí pokrýt kontrolu i rozpočet doladění."
            raise ValueError(msg)
        if self.best_params.get(BRANCH_KEY) != self.winner_branch:
            msg = "Nejlepší parametry musí patřit vítězné větvi kontroly."
            raise ValueError(msg)
        if self.text_preprocessing.keys() != self.text_normalization.keys():
            msg = "Mapy předzpracování a normalizace musí mít stejné reprezentace."
            raise ValueError(msg)
        for name, level in self.text_preprocessing.items():
            _ensure_normalization_matches_preprocessing(
                level, self.text_normalization[name]
            )
        return self
