"""Rozšíření mezí bez změny původních configů předkola."""

from __future__ import annotations

import json

import pytest

from mediparse.domain.stage_refinement import RefinementConfig, branch_config, widen
from mediparse.domain.stage_tuning import StageTuningConfig
from mediparse.domain.tuning import CategoricalParam, FloatParam
from mediparse.infrastructure.stage_config import (
    load_refinement_config,
    load_stage_config,
)
from tests.stage_support import stage_config
from tests.support import REPOSITORY


def test_widen_contains_original_candidate() -> None:
    """C = 0,00104 zůstává přípustné při dolní mezi 1e-5."""
    stage = load_stage_config(REPOSITORY / "config/stage_classical.json")
    refinement = load_refinement_config(REPOSITORY / "config/refinement.json")
    widened = widen(stage, refinement)
    branch = next(b for b in widened.branches if b.name == "word/nb_svm")
    param = next(p for p in branch.prefixed_space() if p.name == "nb_svm.C")
    assert isinstance(param, FloatParam)
    assert param.low == pytest.approx(1e-5)
    assert branch.family.space.contains(branch.family.default | {"C": 0.00104})
    original = next(p for p in stage.families[3].space.params if p.name == "C")
    assert isinstance(original, FloatParam)
    assert original.low == pytest.approx(1e-3)


@pytest.mark.parametrize(
    "widened",
    [
        {"nb_svm.C": {"low": 1e-2}},
        {"nb_svm.C": {"high": 50}},
        {"nb_svm.C": {"low": 1e-3}},
        {"nb_svm.typo": {"low": 1e-5}},
        {"word.sublinear_tf": {"choices": [True]}},
        {"word.sublinear_tf": {"choices": [False, True]}},
        {"word.ngram_min": {"choices": [True]}},
    ],
)
def test_widen_refuses_invalid_changes(widened: dict[str, object]) -> None:
    """Zúžení, překlep i nezměněná mez se odmítnou."""
    stage = load_stage_config(REPOSITORY / "config/stage_classical.json")
    refinement = RefinementConfig.model_validate_json(
        json.dumps({"random_startup_trials": 5, "widened": widened})
    )
    with pytest.raises(ValueError, match=r"rozšířit|Neznámé"):
        widen(stage, refinement)


def test_repository_refinement_config() -> None:
    """Schválené literály patří právě jednomu stupni a všechny stupně se rozšíří."""
    config = load_refinement_config(REPOSITORY / "config/refinement.json")
    assert config.random_startup_trials == 5
    assert config.widened["nb_svm.C"].low == pytest.approx(1e-5)
    assert config.widened["word.ngram_max"].high == 4
    assert config.widened["word2vec.epochs"].low == 1
    assert config.widened["fasttext.lr"].high == pytest.approx(10.0)
    assert config.widened["fasttext.word_ngrams"].high == 3
    stages = [
        load_stage_config(REPOSITORY / f"config/stage_{name}.json")
        for name in ("classical", "embeddings", "networks")
    ]
    for key in config.widened:
        assert (
            sum(
                key in {p.name for b in stage.branches for p in b.prefixed_space()}
                for stage in stages
            )
            == 1
        )
    for stage in stages:
        widen(stage, config)


@pytest.mark.parametrize(
    ("stage_name", "family", "network", "lsa"),
    [
        ("networks", "multirescnn", True, False),
        ("networks", "fasttext", False, False),
        ("classical", "lightgbm_lsa", False, True),
        ("classical", "logreg", False, False),
        ("classical", "linear_svm", False, False),
        ("classical", "nb_svm", False, False),
        ("classical", "complement_nb", False, False),
    ],
)
def test_branch_config_keeps_only_required_technology(
    stage_name: str, family: str, *, network: bool, lsa: bool
) -> None:
    """Síť patří jen MultiResCNN, LSA jen příslušné větvi; studie má šest startů."""
    stage = widen(
        load_stage_config(REPOSITORY / f"config/stage_{stage_name}.json"),
        load_refinement_config(REPOSITORY / "config/refinement.json"),
    )
    for branch in stage.branches:
        if branch.name.split("/")[1] != family:
            continue
        selected = branch_config(stage, branch.name, trials=20, random_startup_trials=5)
        assert (selected.network is not None) is network
        assert (selected.lsa is not None) is lsa
        assert selected.startup_trials == 6
        assert [b.name for b in selected.branches] == [branch.name]


def test_widen_accepts_proper_categorical_superset() -> None:
    """Kategorie ngram_max [2] se rozšíří na [2, 3]."""
    refinement = RefinementConfig.model_validate_json(
        '{"random_startup_trials": 1, "widened": {"r1.ngram_max": {"choices": [2, 3]}}}'
    )
    widened = widen(stage_config(), refinement)
    param = next(
        p for p in widened.branches[0].prefixed_space() if p.name == "r1.ngram_max"
    )
    assert isinstance(param, CategoricalParam)
    assert param.choices == (2, 3)


def test_branch_config_keeps_transformer_and_runs_sequentially() -> None:
    """Vybraná větev zachová transformer a šest startů sekvenčního doladění."""
    stage = load_stage_config(REPOSITORY / "config/stage_transformers.json")
    selected = branch_config(
        stage, "xlmr/transformer", trials=20, random_startup_trials=5
    )
    assert selected.transformer == stage.transformer
    assert selected.concurrent_trials == 1
    assert selected.sampler_startup_trials == 6
    assert selected.network is None
    assert selected.lsa is None


def test_branch_config_clears_transformer_for_mixed_stage() -> None:
    """Lineární větev smíšeného stupně nepřenáší techniku transformeru."""
    stage = load_stage_config(REPOSITORY / "config/stage_transformers.json")
    data = stage.model_dump(mode="json")
    data["concurrent_trials"] = 1
    data["families"].append(stage_config().families[0].model_dump(mode="json"))
    mixed = StageTuningConfig.model_validate_json(json.dumps(data))
    selected = branch_config(mixed, "xlmr/logreg", trials=20, random_startup_trials=5)
    assert selected.transformer is None
