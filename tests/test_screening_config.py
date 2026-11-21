"""Mřížka screeningu: hodnoty protokolu a odmítnutí neplatného configu."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.representation import (
    ClassifierFamily,
    EncoderSpec,
    TfidfRepresentation,
    TokensSpec,
    VectorizerSpec,
    accepts,
)
from mediparse.domain.screening import ScreeningConfig
from mediparse.infrastructure.screening_config import load_screening_config
from tests.hf_encoder_support import SPEC
from tests.stage_support import stage_config
from tests.support import REPOSITORY

if TYPE_CHECKING:
    from pathlib import Path

WORD = {"name": "word-1-2", "analyzer": "word", "ngram_range": [1, 2]}
CHAR = {"name": "char_wb-3-5", "analyzer": "char_wb", "ngram_range": [3, 5]}


def _config() -> dict[str, object]:
    return {
        "seed": 0,
        "folds": 5,
        "threshold_folds": 3,
        "lsa_components": 100,
        "input": {
            "strip_diagnostic_sections": False,
            "text_preprocessing": "dictionary",
        },
        "vectorizers": [WORD, CHAR],
        "families": ["complement_nb", "logreg", "linear_svm", "nb_svm", "lightgbm_lsa"],
    }


def test_repository_config_matches_protocol() -> None:
    """Soubor deklaruje přesně mřížku a volby vstupu schválené v briefu."""
    path = REPOSITORY / "config/screening_classical.json"

    assert load_screening_config(path).model_dump(mode="json") == _config()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("vectorizers", [WORD, WORD]),
        ("families", ["logreg", "logreg"]),
        ("vectorizers", [WORD | {"ngram_range": [3, 2]}]),
        ("folds", 1),
        ("threshold_folds", 1),
        ("lsa_components", 0),
        ("lsa_components", True),
        ("unknown", 0),
        ("vectorizers", []),
        ("families", []),
        ("vectorizers", [WORD | {"name": ""}]),
        ("vectorizers", [WORD | {"analyzer": "char"}]),
        ("vectorizers", [WORD | {"ngram_range": [0, 2]}]),
        ("vectorizers", [WORD | {"ngram_range": ["1", "2"]}]),
        ("vectorizers", [WORD | {"unknown": 0}]),
        ("families", ["unknown"]),
        ("folds", "5"),
        ("seed", -1),
        ("seed", 2**32),
        ("input", {"strip_diagnostic_sections": False, "text_preprocessing": "bad"}),
    ],
    ids=[
        "duplicate-vectorizer",
        "duplicate-family",
        "reversed-ngram-range",
        "one-fold",
        "one-threshold-fold",
        "zero-lsa-components",
        "non-strict-lsa-components",
        "unknown-key",
        "no-vectorizers",
        "no-families",
        "empty-name",
        "unknown-analyzer",
        "zero-ngram",
        "non-strict-ngram",
        "unknown-vectorizer-key",
        "unknown-family",
        "non-strict-folds",
        "negative-seed",
        "seed-out-of-range",
        "unknown-preprocessing",
    ],
)
def test_invalid_config_is_rejected(field: str, value: object) -> None:
    """Mřížka odmítne duplicity, neplatné rozsahy, prázdné položky i chybný typ."""
    body = _config() | {field: value}

    with pytest.raises(ValidationError):
        ScreeningConfig.model_validate_json(json.dumps(body))


def test_missing_file_is_invalid_input(tmp_path: Path) -> None:
    """Chybějící config se přeloží na doménovou chybu."""
    with pytest.raises(InvalidInputError, match="neexistuje"):
        load_screening_config(tmp_path / "missing.json")


def test_missing_lsa_components_is_rejected() -> None:
    """Počet komponent je povinná vědecká volba i pro mřížku bez LightGBM."""
    body = _config()
    body["families"] = ["logreg"]
    del body["lsa_components"]

    with pytest.raises(ValidationError, match="lsa_components"):
        ScreeningConfig.model_validate_json(json.dumps(body))


def test_invalid_file_is_invalid_input(tmp_path: Path) -> None:
    """Chyba validace souboru se přeloží na doménovou chybu."""
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(_config() | {"folds": 1}), encoding="utf-8")

    with pytest.raises(InvalidInputError, match="neodpovídá schématu"):
        load_screening_config(path)


def test_shallow_config_matches_protocol() -> None:
    """Mělký stupeň používá původní text a připnutý mmBERT s LR."""
    config = load_screening_config(REPOSITORY / "config/screening_shallow.json")
    assert len(config.encoders) == 1
    assert config.encoders[0].model_dump() == {
        "name": "mmbert-base-mean",
        "model": "jhu-clsp/mmBERT-base",
        "revision": "c5955035435e2bf121cde7f3c8863ef52ff35d82",
        "window": 8192,
        "device": "cuda",
        "dtype": "bfloat16",
    }
    assert config.input.model_dump(mode="json") == {
        "strip_diagnostic_sections": False,
        "text_preprocessing": "none",
    }
    assert config.model_dump(mode="json")["families"] == ["logreg"]
    assert (
        load_screening_config(REPOSITORY / "config/screening_classical.json").encoders
        == ()
    )


@pytest.mark.parametrize(
    "changes", [{"revision": "a" * 39}, {"device": "mps"}, {"extra": 1}]
)
def test_encoder_spec_rejects_invalid_values(changes: dict[str, object]) -> None:
    """Encoder vyžaduje plnou revizi, známé zařízení a uzavřené schéma."""
    spec = SPEC.model_dump(mode="json")
    with pytest.raises(ValidationError):
        EncoderSpec.model_validate(spec | changes)


def test_duplicate_representation_name_is_rejected() -> None:
    """Encoder a TF-IDF sdílejí jeden prostor jedinečných jmen."""
    encoder = SPEC.model_dump(mode="json") | {"name": WORD["name"]}
    with pytest.raises(
        ValidationError, match="Jména vektorizérů screeningu musí být jedinečná"
    ):
        ScreeningConfig.model_validate_json(
            json.dumps(
                _config() | {"families": ["logreg"], "vectorizers": [encoder, WORD]}
            )
        )


@pytest.mark.parametrize("family", ClassifierFamily)
@pytest.mark.parametrize(
    ("spec", "kind"),
    [
        (WORD, VectorizerSpec),
        (SPEC.model_dump(mode="json"), EncoderSpec),
        ({"name": "word-tokens"}, TokensSpec),
    ],
)
def test_family_representation_compatibility(
    spec: dict[str, object],
    kind: type,
    family: ClassifierFamily,
) -> None:
    """Celý součin odpovídá doslovné tabulce sedmi povolených dvojic."""
    allowed = {
        (VectorizerSpec, "complement_nb"),
        (VectorizerSpec, "logreg"),
        (VectorizerSpec, "linear_svm"),
        (VectorizerSpec, "nb_svm"),
        (VectorizerSpec, "lightgbm_lsa"),
        (EncoderSpec, "logreg"),
        (TokensSpec, "fasttext"),
    }
    body = json.dumps(_config() | {"vectorizers": [spec], "families": [family]})
    if (kind, family) in allowed:
        assert isinstance(
            ScreeningConfig.model_validate_json(body).vectorizers[0], kind
        )
    else:
        message = f"Rodina {family} nepracuje nad reprezentací {spec['name']}."
        with pytest.raises(ValidationError, match=re.escape(message)):
            ScreeningConfig.model_validate_json(body)


def test_first_incompatible_pair_is_reported() -> None:
    """První chyba respektuje pořadí reprezentací a pak rodin."""
    message = "Rodina fasttext nepracuje nad reprezentací word-1-2."
    with pytest.raises(ValidationError, match=re.escape(message)):
        ScreeningConfig.model_validate_json(
            json.dumps(
                _config()
                | {
                    "vectorizers": [WORD, {"name": "word-tokens"}],
                    "families": ["logreg", "fasttext"],
                }
            )
        )


@pytest.mark.parametrize("value", [{"name": ""}, {"name": "x", "analyzer": "word"}])
def test_tokens_spec_rejects_invalid_values(value: dict[str, object]) -> None:
    """Tokeny vyžadují neprázdné jméno a odmítají parametry TF-IDF."""
    with pytest.raises(ValidationError):
        TokensSpec.model_validate(value)


def test_fasttext_config_matches_protocol() -> None:
    """FastText drží slovník a parametry klasiky, embeddingy se učí sám."""
    config = load_screening_config(REPOSITORY / "config/screening_fasttext.json")
    assert config.vectorizers == (TokensSpec(name="word-tokens"),)
    assert config.model_dump(mode="json")["families"] == ["fasttext"]
    assert config.input.model_dump(mode="json") == {
        "strip_diagnostic_sections": False,
        "text_preprocessing": "dictionary",
    }


@pytest.mark.parametrize("name", ["classical", "shallow", "fasttext"])
def test_stage_configs_share_protocol(name: str) -> None:
    """Všechny stupně sdílejí seed a parametry vnější i vnitřní CV a LSA."""
    config = load_screening_config(REPOSITORY / f"config/screening_{name}.json")
    assert (
        config.seed,
        config.folds,
        config.threshold_folds,
        config.lsa_components,
    ) == (0, 5, 3, 100)


@pytest.mark.parametrize("family", ClassifierFamily)
@pytest.mark.parametrize(
    ("spec", "allowed"),
    [
        (
            VectorizerSpec.model_validate_json(json.dumps(WORD)),
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
        VectorizerSpec.model_validate_json(json.dumps(WORD)),
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
