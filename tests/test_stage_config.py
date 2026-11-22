"""Doménové invarianty podmíněného prostoru a vítěze."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from mediparse.domain import stage_tuning as stage_domain
from mediparse.domain.model_input import TextPreprocessing
from mediparse.domain.representation import (
    EncoderSpec,
    TfidfRepresentation,
    TokensSpec,
    Word2VecRepresentation,
)
from mediparse.domain.stage_tuning import (
    StageLsa,
    StageTrialRecord,
    StageTuningConfig,
    best_trial,
)
from tests.hf_encoder_support import SPEC
from tests.stage_support import stage_config, token_stage_config, word2vec_stage_config

if TYPE_CHECKING:
    from collections.abc import Callable


def test_branches_and_namespaces() -> None:
    """Reprezentace tvoří vnější smyčku a rodiny sdílejí jmenný prostor."""
    config = stage_config()
    assert config.startup_trials == 5
    assert [b.name for b in config.branches] == [
        "r1/logreg",
        "r1/complement_nb",
        "r2/logreg",
        "r2/complement_nb",
    ]
    for branch in config.branches:
        params = branch.default_params()
        rep, fam = branch.params_of(params)
        assert isinstance(branch.representation, TfidfRepresentation)
        assert rep == branch.representation.default
        assert fam == branch.family.default
        assert {p.name for p in branch.prefixed_space()} == params.keys() - {"branch"}
        assert params["branch"] == branch.name


def test_winner_tie_uses_earlier_trial() -> None:
    """Hodnoty 0,7; 0,9; 0,9 vyberou pokus 1."""
    records = [
        StageTrialRecord(n, {}, default=False, value=value, converged=True, seconds=1.0)
        for n, value in enumerate([0.7, 0.9, 0.9])
    ]
    assert best_trial(records).number == 1


@pytest.mark.parametrize(
    ("factory", "changes", "message"),
    [
        *[
            pytest.param(
                word2vec_stage_config,
                {("representations", 0, "space", "params", 6, "high"): high},
                "sample vyžaduje kladný float menší než 1.",
                id=f"word2vec-sample-boundary-{high}",
            )
            for high in (1.0, 2.0)
        ],
        pytest.param(
            word2vec_stage_config,
            {("representations", 0, "workers"): 0},
            "Input should be greater than 0",
            id="word2vec-workers-zero",
        ),
        pytest.param(
            word2vec_stage_config,
            {("representations", 0, "alpha"): 0.025},
            "Extra inputs are not permitted",
            id="word2vec-extra-alpha",
        ),
        pytest.param(
            stage_config,
            {("families", 0, "default", "C"): 10000.0},
            "Výchozí parametry rodiny",
            id="family-default-outside",
        ),
        pytest.param(
            stage_config,
            {("representations", 0, "default", "max_df"): 1},
            "Výchozí parametry reprezentace",
            id="max-df-int-default",
        ),
        pytest.param(
            stage_config,
            {
                ("representations", 1): {
                    "name": "tok",
                    "text_preprocessing": "dictionary",
                }
            },
            "Reprezentace tok nepatří do žádné větve",
            id="unused-representation",
        ),
        pytest.param(
            stage_config, {("trials",): 4}, "Rozpočet", id="insufficient-budget"
        ),
        pytest.param(
            stage_config,
            {
                ("representations", 0, "space", "params", 0): {
                    "kind": "int",
                    "name": "ngram_min",
                    "low": 1,
                    "high": 3,
                    "log": False,
                }
            },
            "Maximum ngram_min",
            id="ngram-int-order",
        ),
        pytest.param(
            stage_config,
            {
                ("representations", 0, "space", "params", 2): {
                    "kind": "float",
                    "name": "min_df",
                    "low": 0.1,
                    "high": 1.0,
                    "log": False,
                },
                ("representations", 0, "default", "min_df"): 1.0,
            },
            "min_df vyžaduje celočíselný",
            id="min-df-float-space",
        ),
        pytest.param(
            stage_config,
            {("families", 0, "family"): "fasttext"},
            "Rodina fasttext",
            id="incompatible-family",
        ),
        pytest.param(
            stage_config,
            {("representations", 1, "name"): "r1"},
            "Jména reprezentací",
            id="duplicate-representation",
        ),
        pytest.param(
            stage_config,
            {("seed",): 4294967286, ("trials",): 60},
            "Seed spolu",
            id="seed-overflow",
        ),
        pytest.param(
            stage_config,
            {("representations", 0, "space", "params", 0, "name"): "unknown"},
            "právě pět parametrů",
            id="invalid-parameter-names",
        ),
        pytest.param(
            stage_config,
            {
                ("representations", 0, "space", "params", 3): {
                    "kind": "categorical",
                    "name": "max_df",
                    "choices": [1.0],
                }
            },
            "max_df vyžaduje float",
            id="max-df-category",
        ),
        pytest.param(
            stage_config,
            {
                ("representations", 0, "space", "params", 4): {
                    "kind": "categorical",
                    "name": "sublinear_tf",
                    "choices": [1],
                }
            },
            "sublinear_tf bool",
            id="sublinear-int",
        ),
        pytest.param(
            stage_config,
            {
                ("families", 1): {
                    "family": "logreg",
                    "space": {
                        "params": [
                            {"kind": "categorical", "name": "C", "choices": [1.0]}
                        ]
                    },
                    "default": {"C": 1.0},
                }
            },
            "Rodiny musí být jedinečné",
            id="duplicate-family",
        ),
        pytest.param(
            stage_config,
            {
                ("representations", 0, "space", "params", 0, "choices"): [1, 3],
                ("representations", 0, "space", "params", 1, "choices"): [2, 4],
            },
            "Maximum ngram_min",
            id="ngram-category-order",
        ),
        (
            word2vec_stage_config,
            {
                ("representations", 0, "space", "params"): [
                    p
                    for p in word2vec_stage_config()
                    .representations[0]
                    .model_dump(mode="json")["space"]["params"]
                    if p["name"] != "epochs"
                ]
            },
            "Prostor word2vec musí obsahovat právě sedm parametrů gensim",
        ),
        (
            word2vec_stage_config,
            {
                ("representations", 0, "space", "params"): [
                    *word2vec_stage_config()
                    .representations[0]
                    .model_dump(mode="json")["space"]["params"],
                    {"name": "alpha", "kind": "categorical", "choices": [1]},
                ]
            },
            "Prostor word2vec musí obsahovat právě sedm parametrů gensim",
        ),
        (
            word2vec_stage_config,
            {
                ("representations", 0, "space", "params", 1): {
                    "name": "window",
                    "kind": "float",
                    "low": 1.0,
                    "high": 2.0,
                    "log": False,
                }
            },
            "vyžaduje celočíselný prostor",
        ),
        (
            word2vec_stage_config,
            {("representations", 0, "space", "params", 0, "choices"): [0]},
            "Celočíselné parametry word2vec musí být kladné",
        ),
        (
            word2vec_stage_config,
            {("representations", 0, "space", "params", 1, "choices"): [0]},
            "Celočíselné parametry word2vec musí být kladné",
        ),
        (
            word2vec_stage_config,
            {("representations", 0, "space", "params", 2, "choices"): [0]},
            "Celočíselné parametry word2vec musí být kladné",
        ),
        (
            word2vec_stage_config,
            {("representations", 0, "space", "params", 4, "choices"): [0]},
            "Celočíselné parametry word2vec musí být kladné",
        ),
        (
            word2vec_stage_config,
            {("representations", 0, "space", "params", 5, "choices"): [0]},
            "Celočíselné parametry word2vec musí být kladné",
        ),
        (
            word2vec_stage_config,
            {("representations", 0, "space", "params", 3, "choices"): [0, 2]},
            "sg vyžaduje volby 0 nebo 1",
        ),
        (
            word2vec_stage_config,
            {("representations", 0, "space", "params", 3, "choices"): [-1, 0]},
            "sg vyžaduje volby 0 nebo 1",
        ),
        (
            word2vec_stage_config,
            {
                ("representations", 0, "space", "params", 6): {
                    "name": "sample",
                    "kind": "categorical",
                    "choices": [0.001],
                }
            },
            "sample vyžaduje kladný float",
        ),
        (
            word2vec_stage_config,
            {
                ("representations", 0, "space", "params", 6): {
                    "name": "sample",
                    "kind": "float",
                    "low": 0.0,
                    "high": 0.01,
                    "log": False,
                }
            },
            "sample vyžaduje kladný float",
        ),
        (
            word2vec_stage_config,
            {("representations", 0, "default", "window"): 3},
            "Výchozí parametry reprezentace",
        ),
    ],
)
def test_invalid_config(
    factory: Callable[..., StageTuningConfig],
    changes: dict[tuple[str | int, ...], object],
    message: str,
) -> None:
    """Každou neplatnou úpravu odmítne její vlastní validační pravidlo."""
    data = factory().model_dump(mode="json")
    for path, value in changes.items():
        target: Any = data
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    with pytest.raises(ValidationError, match=message):
        StageTuningConfig.model_validate_json(json.dumps(data))


def test_branch_identity_is_resolved_in_domain() -> None:
    """Doména vrátí větev i její části z identity uložené v pokusu."""
    config = stage_config()
    for branch in config.branches:
        params = branch.default_params()
        assert params[stage_domain.BRANCH_KEY] == branch.name
        assert config.branch(params) == branch
        assert stage_domain.branch_parts(branch.name) == (
            branch.representation.name,
            branch.family.family,
        )


def test_lsa_required_exactly_with_lightgbm() -> None:
    """LSA je povinné právě pro rodinu lightgbm_lsa."""
    config = stage_config()
    data = config.model_dump(mode="json")
    data["families"][0]["family"] = "lightgbm_lsa"
    with pytest.raises(
        ValidationError, match="LSA musí být zadané právě s rodinou lightgbm_lsa"
    ):
        StageTuningConfig.model_validate_json(json.dumps(data))
    with pytest.raises(
        ValidationError, match="LSA musí být zadané právě s rodinou lightgbm_lsa"
    ):
        stage_config(lsa={"components": 2, "max_features": 1000})
    assert config.lsa is None


def test_lsa_cap_exceeds_components() -> None:
    """Strop slovníku musí převyšovat počet komponent."""
    with pytest.raises(
        ValidationError, match="Strop slovníku LSA musí převyšovat počet komponent"
    ):
        StageLsa(components=2, max_features=2)


def test_branches_follow_accepted_pairs() -> None:
    """Filtrovaný součin respektuje pořadí i rozpočet výchozích bodů."""
    base = stage_config().model_dump(mode="json")
    token = token_stage_config().model_dump(mode="json")
    config = stage_config(
        representations=[
            base["representations"][0],
            SPEC.model_dump(mode="json") | {"name": "enc"},
            {"name": "tok", "text_preprocessing": "dictionary"},
        ],
        families=[base["families"][0], token["families"][0]],
    )
    assert [b.name for b in config.branches] == [
        "r1/logreg",
        "enc/logreg",
        "tok/fasttext",
    ]
    assert config.startup_trials == 3 + config.random_startup_trials
    assert config.encoders == (SPEC.model_copy(update={"name": "enc"}),)


def test_branch_without_representation_space() -> None:
    """Tokeny ponechávají v pokusu pouze prostor rodiny."""
    branch = token_stage_config().branches[0]
    assert {p.name for p in branch.prefixed_space()} == {
        "fasttext.lr",
        "fasttext.epoch",
        "fasttext.word_ngrams",
        "fasttext.dim",
    }
    params = {
        "branch": "tok/fasttext",
        "fasttext.lr": 0.5,
        "fasttext.epoch": 1,
        "fasttext.word_ngrams": 2,
        "fasttext.dim": 2,
    }
    assert branch.default_params() == params
    assert branch.params_of(params) == ({}, branch.family.default)


def test_word2vec_branch_namespaces() -> None:
    """Word2vec má vlastní prostor a unie rozliší jeho typ."""
    config = word2vec_stage_config()
    assert type(config.representations[0]) is Word2VecRepresentation
    branch = config.branches[0]
    assert branch.name == "w2v/logreg"
    assert {p.name for p in branch.prefixed_space()} == {
        "w2v.vector_size",
        "w2v.window",
        "w2v.min_count",
        "w2v.sg",
        "w2v.epochs",
        "w2v.negative",
        "w2v.sample",
        "logreg.C",
    }
    assert branch.params_of(branch.default_params()) == (
        {
            "vector_size": 7,
            "window": 2,
            "min_count": 2,
            "sg": 0,
            "epochs": 5,
            "negative": 5,
            "sample": 0.001,
        },
        {"C": 1.0},
    )


@pytest.mark.parametrize(
    "spec",
    [
        stage_config().representations[0],
        word2vec_stage_config().representations[0],
        TokensSpec(name="tok", text_preprocessing=TextPreprocessing.DICTIONARY),
        SPEC,
    ],
)
def test_representation_requires_preprocessing(
    spec: TfidfRepresentation | Word2VecRepresentation | TokensSpec | EncoderSpec,
) -> None:
    """Každá reprezentace vyžaduje explicitní vědeckou volbu vstupu."""
    data = spec.model_dump(mode="json")
    del data["text_preprocessing"]
    with pytest.raises(ValidationError, match=r"text_preprocessing\s+Field required"):
        type(spec).model_validate_json(json.dumps(data))


def test_preprocessing_rejects_obsolete_fields() -> None:
    """Stupeň nepřijímá původní společné nastavení vstupu."""
    data = stage_config().model_dump(mode="json") | {"input": {}}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        StageTuningConfig.model_validate_json(json.dumps(data))
