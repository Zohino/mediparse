"""Adaptéry převzetí parametrů: čtení exportu a bezpečný zápis do configu vzorkovače."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Any

import pytest

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.labels import Diagnosis
from mediparse.domain.synthetic_parameters import SyntheticParameters
from mediparse.infrastructure.sampler_config import (
    SamplerConfigFile,
    load_sampler_config,
)
from mediparse.infrastructure.synthetic_parameters_file import SyntheticParametersFile
from tests.support import REPOSITORY_CONFIG, export_payload

if TYPE_CHECKING:
    from pathlib import Path

    from mediparse.domain.synthetic_plan import SamplerConfig


NEW_NOTES = 300


def _parameters(payload: dict[str, Any]) -> SyntheticParameters:
    return SyntheticParameters.model_validate_json(json.dumps(payload))


def _config_copy(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic_plan.json"
    shutil.copyfile(REPOSITORY_CONFIG, path)
    return path


def _own_values(config: SamplerConfig) -> SyntheticParameters:
    keys = tuple(config.labels.prevalence)
    return _parameters({
        "selection": list(keys),
        "prevalence": dict(config.labels.prevalence),
        "conditional": {row: dict(config.labels.conditional[row]) for row in keys},
        "mentions": {
            key: {
                "given_positive": config.mentions.diagnoses[key].given_positive,
                "given_negative": config.mentions.diagnoses[key].given_negative,
                "keywords": list(config.mentions.diagnoses[key].keywords),
            }
            for key in keys
        },
    })


def test_export_file_is_parsed(tmp_path: Path) -> None:
    """Soubor exportu se načte do parametrů."""
    path = tmp_path / "export.json"
    path.write_text(json.dumps(export_payload(("a", "b"))), encoding="utf-8")

    assert SyntheticParametersFile(path).load().selection == ("a", "b")


def test_missing_export_is_invalid_input(tmp_path: Path) -> None:
    """Chybějící export je chyba vstupu, ne systémová výjimka."""
    with pytest.raises(InvalidInputError, match="neexistuje"):
        SyntheticParametersFile(tmp_path / "missing.json").load()


@pytest.mark.parametrize("content", ["null", "{}"])
def test_export_with_wrong_content_is_invalid_input(
    tmp_path: Path, content: str
) -> None:
    """Soubor s null nebo prázdným objektem neodpovídá schématu."""
    path = tmp_path / "export.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(InvalidInputError, match="neodpovídá schématu"):
        SyntheticParametersFile(path).load()


def test_apply_replaces_three_blocks_and_notes_and_keeps_the_rest(
    tmp_path: Path, sampler_config: SamplerConfig
) -> None:
    """Config po zápisu nese nové bloky a počet zpráv, ostatní klíče zůstanou."""
    path = _config_copy(tmp_path)
    parameters = _parameters(export_payload(tuple(Diagnosis), prevalence=0.25))

    SamplerConfigFile(path).apply(parameters, NEW_NOTES)

    config = load_sampler_config(path)
    assert config.patients.notes == NEW_NOTES
    assert config.labels.prevalence[Diagnosis.CKD] == pytest.approx(0.25)
    assert config.mentions.diagnoses[Diagnosis.AKI].given_positive == pytest.approx(0.9)
    assert config.mentions.diagnoses[Diagnosis.AKI].keywords == ("aki phrase", "AKI")
    assert config.seed == sampler_config.seed
    assert config.structure == sampler_config.structure
    assert config.labels.ipf_max_passes == sampler_config.labels.ipf_max_passes
    assert config.mentions.negation == sampler_config.mentions.negation
    assert config.patients.max_notes == sampler_config.patients.max_notes


def test_selection_outside_domain_is_refused_and_config_untouched(
    tmp_path: Path,
) -> None:
    """Pětice, kterou doména nezná, config nepřepíše ani o bajt."""
    path = _config_copy(tmp_path)
    before = path.read_bytes()
    parameters = _parameters(export_payload(("a", "b", "c", "d", "e")))

    with pytest.raises(InvalidInputError, match="schématu"):
        SamplerConfigFile(path).apply(parameters, 200)

    assert path.read_bytes() == before


def test_ipf_not_converging_is_refused_and_config_untouched(tmp_path: Path) -> None:
    """Matice s odporujícími marginály (A=B, B=C, A a C nezávislé) skončí chybou vstupu a config zůstane."""
    path = _config_copy(tmp_path)
    before = path.read_bytes()
    payload = export_payload(tuple(Diagnosis), prevalence=0.2)
    conditional = payload["conditional"]
    for first, second in (("diabetes", "ckd"), ("ckd", "heart_failure")):
        conditional[first][second] = 1.0
        conditional[second][first] = 1.0

    with pytest.raises(InvalidInputError):
        SamplerConfigFile(path).apply(_parameters(payload), 200)

    assert path.read_bytes() == before


@pytest.mark.parametrize("content", ["{not json", "null", "{}"])
def test_malformed_config_is_invalid_input_and_untouched(
    tmp_path: Path, content: str
) -> None:
    """Config, který není JSON nebo nemá tři bloky, je chyba vstupu, ne traceback."""
    path = tmp_path / "synthetic_plan.json"
    path.write_text(content, encoding="utf-8")
    parameters = _parameters(export_payload(tuple(Diagnosis)))

    with pytest.raises(InvalidInputError, match="schématu"):
        SamplerConfigFile(path).apply(parameters, 200)

    assert path.read_text(encoding="utf-8") == content


def test_missing_config_is_invalid_input(tmp_path: Path) -> None:
    """Chybějící config je chyba vstupu."""
    parameters = _parameters(export_payload(tuple(Diagnosis)))

    with pytest.raises(InvalidInputError, match="neexistuje"):
        SamplerConfigFile(tmp_path / "missing.json").apply(parameters, 200)


def test_roundtrip_with_own_values_changes_only_notes(
    tmp_path: Path, sampler_config: SamplerConfig
) -> None:
    """Export s hodnotami configu změní jen počet zpráv, bajty ostatních řádků zůstanou."""
    path = _config_copy(tmp_path)
    before = path.read_text(encoding="utf-8")
    current = sampler_config.patients.notes

    SamplerConfigFile(path).apply(_own_values(sampler_config), current + 100)

    expected = before.replace(f'"notes": {current}', f'"notes": {current + 100}')
    assert expected != before
    assert path.read_text(encoding="utf-8") == expected


def test_roundtrip_with_same_notes_keeps_bytes(
    tmp_path: Path, sampler_config: SamplerConfig
) -> None:
    """Zápis stejných hodnot dá stejné bajty, takže diff ukáže jen skutečné změny."""
    path = _config_copy(tmp_path)
    before = path.read_bytes()

    SamplerConfigFile(path).apply(
        _own_values(sampler_config), sampler_config.patients.notes
    )

    assert path.read_bytes() == before
