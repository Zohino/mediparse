"""Kompatibilita rodin a jména společných reprezentací."""

import json

import pytest
from pydantic import ValidationError

from mediparse.domain.representation import (
    ClassifierFamily,
    EncoderSpec,
    TfidfRepresentation,
    TokensSpec,
    VectorizerSpec,
    accepts,
)
from tests.hf_encoder_support import SPEC
from tests.stage_support import stage_config


@pytest.mark.parametrize("family", ClassifierFamily)
@pytest.mark.parametrize(
    ("spec", "allowed"),
    [
        (
            VectorizerSpec(name="word", analyzer="word", ngram_range=(1, 2)),
            {"complement_nb", "logreg", "linear_svm", "nb_svm", "lightgbm_lsa"},
        ),
        (
            stage_config().representations[0],
            {"complement_nb", "logreg", "linear_svm", "nb_svm", "lightgbm_lsa"},
        ),
        (SPEC, {"logreg"}),
        (TokensSpec(name="tok"), {"fasttext"}),
    ],
)
def test_accepts_pairs(
    spec: VectorizerSpec | TfidfRepresentation | EncoderSpec | TokensSpec,
    allowed: set[str],
    family: ClassifierFamily,
) -> None:
    """Čtyři druhy přijímají právě rodiny uvedené v kontraktu."""
    assert accepts(spec, family) == (family.value in allowed)


@pytest.mark.parametrize("name", ["a/b", "a.b"])
@pytest.mark.parametrize(
    "spec",
    [
        VectorizerSpec(name="word", analyzer="word", ngram_range=(1, 2)),
        SPEC,
        TokensSpec(name="tok"),
    ],
)
def test_representation_name_rejects_separators(
    name: str, spec: VectorizerSpec | EncoderSpec | TokensSpec
) -> None:
    """Jméno nesmí obsahovat oddělovač větve ani parametru."""
    with pytest.raises(ValidationError):
        type(spec).model_validate_json(
            json.dumps(spec.model_dump(mode="json") | {"name": name})
        )
