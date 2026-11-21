"""Reprezentace textu a kompatibilní rodiny klasifikátorů."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self, assert_never

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from mediparse.domain.digest import CommitSha
from mediparse.domain.tuning import (
    CategoricalParam,
    FloatParam,
    IntParam,
    Parameter,
    ParamValue,
    SearchSpace,
)

RepresentationName = Annotated[str, Field(min_length=1, pattern=r"^[^./]+$")]


class ClassifierFamily(StrEnum):
    """Rodiny klasifikátorů screeningu a ladění stupňů."""

    COMPLEMENT_NB = "complement_nb"
    LOGREG = "logreg"
    LINEAR_SVM = "linear_svm"
    NB_SVM = "nb_svm"
    LIGHTGBM_LSA = "lightgbm_lsa"
    FASTTEXT = "fasttext"


class VectorizerSpec(BaseModel):
    """Pojmenovaný vektorizér s analyzátorem a rozsahem n-gramů."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    name: RepresentationName
    analyzer: Literal["word", "char_wb"]
    ngram_range: tuple[PositiveInt, PositiveInt]

    @model_validator(mode="after")
    def _ngram_range_is_ordered(self) -> Self:
        if self.ngram_range[0] > self.ngram_range[1]:
            msg = "Minimum rozsahu n-gramů nesmí překročit maximum."
            raise ValueError(msg)
        return self


class EncoderSpec(BaseModel):
    """Připnutý encoder a explicitní numerický režim reprezentace."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    name: RepresentationName
    model: Annotated[str, Field(min_length=1)]
    revision: CommitSha
    window: PositiveInt
    device: Literal["cpu", "cuda"]
    dtype: Literal["float32", "bfloat16"]


class TokensSpec(BaseModel):
    """Slovní tokeny pro rodinu, která se reprezentaci učí sama."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    name: RepresentationName


def _int_choices(param: Parameter) -> tuple[int, int]:
    if isinstance(param, IntParam):
        return param.low, param.high
    if isinstance(param, CategoricalParam):
        choices = tuple(v for v in param.choices if type(v) is int)
        if len(choices) == len(param.choices):
            return min(choices), max(choices)
    msg = f"Parametr {param.name} vyžaduje celočíselný prostor."
    raise ValueError(msg)


class TfidfRepresentation(BaseModel):
    """Pojmenovaná reprezentace se striktním prostorem TF-IDF."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    name: RepresentationName
    analyzer: Literal["word", "char_wb"]
    space: SearchSpace
    default: dict[str, ParamValue]

    @model_validator(mode="after")
    def _valid_space(self) -> Self:
        params = {p.name: p for p in self.space.params}
        if params.keys() != {
            "ngram_min",
            "ngram_max",
            "min_df",
            "max_df",
            "sublinear_tf",
        }:
            msg = "Prostor TF-IDF musí obsahovat právě pět parametrů reprezentace."
            raise ValueError(msg)
        ngram_min = _int_choices(params["ngram_min"])
        ngram_max = _int_choices(params["ngram_max"])
        _int_choices(params["min_df"])
        if ngram_min[1] > ngram_max[0]:
            msg = "Maximum ngram_min nesmí překročit minimum ngram_max."
            raise ValueError(msg)
        flag = params["sublinear_tf"]
        if not isinstance(params["max_df"], FloatParam) or not (
            isinstance(flag, CategoricalParam)
            and all(type(v) is bool for v in flag.choices)
        ):
            msg = "max_df vyžaduje float a sublinear_tf bool kategorie."
            raise ValueError(msg)
        if not self.space.contains(self.default):
            msg = "Výchozí parametry reprezentace neleží v prostoru."
            raise ValueError(msg)
        return self


def accepts(
    spec: VectorizerSpec | TfidfRepresentation | EncoderSpec | TokensSpec,
    family: ClassifierFamily,
) -> bool:
    """Určí, zda rodina pracuje nad daným druhem reprezentace.

    Returns:
        Kompatibilita reprezentace a klasifikátoru.
    """
    match spec:
        case EncoderSpec():
            return family is ClassifierFamily.LOGREG
        case TokensSpec():
            return family is ClassifierFamily.FASTTEXT
        case VectorizerSpec() | TfidfRepresentation():
            return family is not ClassifierFamily.FASTTEXT
        case _:
            assert_never(spec)
