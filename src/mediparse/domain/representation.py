"""Reprezentace textu a kompatibilní rodiny klasifikátorů."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self, assert_never

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from mediparse.domain.digest import CommitSha
from mediparse.domain.model_input import TextPreprocessing
from mediparse.domain.tuning import (
    CategoricalParam,
    FloatParam,
    IntParam,
    Parameter,
    ParamValue,
    SearchSpace,
)

Aggregation = Literal["max", "mean"]

Device = Literal["cpu", "cuda"]

RepresentationName = Annotated[str, Field(min_length=1, pattern=r"^[^./]+$")]


class ClassifierFamily(StrEnum):
    """Rodiny klasifikátorů pro ladění a trénink."""

    COMPLEMENT_NB = "complement_nb"
    LOGREG = "logreg"
    LINEAR_SVM = "linear_svm"
    NB_SVM = "nb_svm"
    LIGHTGBM_LSA = "lightgbm_lsa"
    LIGHTGBM = "lightgbm"
    FASTTEXT = "fasttext"
    MULTIRESCNN = "multirescnn"


class EncoderSpec(BaseModel):
    """Připnutý encoder a explicitní numerický režim reprezentace."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    name: RepresentationName
    text_preprocessing: TextPreprocessing
    model: Annotated[str, Field(min_length=1)]
    revision: CommitSha
    window: PositiveInt
    device: Device
    dtype: Literal["float32", "bfloat16"]


class TokensSpec(BaseModel):
    """Slovní tokeny pro rodinu, která se reprezentaci učí sama."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    name: RepresentationName
    text_preprocessing: TextPreprocessing


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
    text_preprocessing: TextPreprocessing
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


class Word2VecRepresentation(BaseModel):
    """Word2vec s laděným prostorem a pevným počtem vláken."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    name: RepresentationName
    text_preprocessing: TextPreprocessing
    workers: PositiveInt
    space: SearchSpace
    default: dict[str, ParamValue]

    @model_validator(mode="after")
    def _valid_space(self) -> Self:
        params = {p.name: p for p in self.space.params}
        if params.keys() != {
            "vector_size",
            "window",
            "min_count",
            "sg",
            "epochs",
            "negative",
            "sample",
        }:
            msg = "Prostor word2vec musí obsahovat právě sedm parametrů gensim."
            raise ValueError(msg)
        sample = params["sample"]
        # Od 1 gensim chápe sample jako absolutní práh, nikoli podíl korpusu.
        if not isinstance(sample, FloatParam) or sample.low <= 0 or sample.high >= 1:
            msg = "sample vyžaduje kladný float menší než 1."
            raise ValueError(msg)
        sg_low, sg_high = _int_choices(params["sg"])
        if sg_low < 0 or sg_high > 1:
            msg = "sg vyžaduje volby 0 nebo 1."
            raise ValueError(msg)
        if any(
            _int_choices(params[name])[0] < 1
            for name in ("vector_size", "window", "min_count", "epochs", "negative")
        ):
            msg = "Celočíselné parametry word2vec musí být kladné."
            raise ValueError(msg)
        if not self.space.contains(self.default):
            msg = "Výchozí parametry reprezentace neleží v prostoru."
            raise ValueError(msg)
        return self


type Representation = (
    TfidfRepresentation | Word2VecRepresentation | EncoderSpec | TokensSpec
)


def accepts(spec: Representation, family: ClassifierFamily) -> bool:
    """Určí, zda rodina pracuje nad daným druhem reprezentace.

    Returns:
        Kompatibilita reprezentace a klasifikátoru.
    """
    match spec:
        case EncoderSpec() | Word2VecRepresentation():
            return family in {
                ClassifierFamily.LOGREG,
                ClassifierFamily.LINEAR_SVM,
                ClassifierFamily.LIGHTGBM,
            }
        case TokensSpec():
            return family in {ClassifierFamily.FASTTEXT, ClassifierFamily.MULTIRESCNN}
        case TfidfRepresentation():
            return family in {
                ClassifierFamily.COMPLEMENT_NB,
                ClassifierFamily.LOGREG,
                ClassifierFamily.LINEAR_SVM,
                ClassifierFamily.NB_SVM,
                ClassifierFamily.LIGHTGBM_LSA,
            }
        case _:
            assert_never(spec)
