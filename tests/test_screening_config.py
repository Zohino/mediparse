"""Mřížka klasického screeningu: hodnoty protokolu a odmítnutí neplatného configu."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.screening import EncoderSpec, ScreeningConfig
from mediparse.infrastructure.screening_config import load_screening_config
from tests.hf_encoder_support import SPEC

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
    path = Path(__file__).parents[1] / "config/screening_classical.json"

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
    config = load_screening_config(Path("config/screening_shallow.json"))
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
        config.seed,
        config.folds,
        config.threshold_folds,
        config.lsa_components,
    ) == (0, 5, 3, 100)
    assert load_screening_config(Path("config/screening_classical.json")).encoders == ()


@pytest.mark.parametrize(
    "changes", [{"revision": "a" * 39}, {"device": "mps"}, {"extra": 1}]
)
def test_encoder_spec_rejects_invalid_values(changes: dict[str, object]) -> None:
    """Encoder vyžaduje plnou revizi, známé zařízení a uzavřené schéma."""
    spec = SPEC.model_dump(mode="json")
    with pytest.raises(ValidationError):
        EncoderSpec.model_validate(spec | changes)


@pytest.mark.parametrize(
    ("families", "duplicate", "message"),
    [
        (["complement_nb"], False, "Encoder podporuje jen rodinu logreg."),
        (["logreg", "complement_nb"], False, "Encoder podporuje jen rodinu logreg."),
        (["logreg"], True, "Jména vektorizérů screeningu musí být jedinečná."),
    ],
)
def test_encoder_grid_rejects_incompatible_family_or_duplicate_name(
    families: list[str], message: str, *, duplicate: bool
) -> None:
    """Hustý encoder smí jen k LR a jména sdílí s TF-IDF jeden prostor."""
    encoder = SPEC.model_dump(mode="json") | {"name": WORD["name"]}
    with pytest.raises(ValidationError, match=re.escape(message)):
        ScreeningConfig.model_validate_json(
            json.dumps(
                _config()
                | {
                    "families": families,
                    "vectorizers": [encoder, WORD] if duplicate else [encoder],
                }
            )
        )
