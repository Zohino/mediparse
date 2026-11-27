"""Pacientské podíly a striktní podmínky pilotu."""

from math import ceil
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from mediparse.domain.split import patient_fraction
from mediparse.domain.transformer_pilot import TransformerPilotConfig
from mediparse.infrastructure.stage_config import load_pilot_config, load_stage_config
from tests.screening_support import note

CONFIG = Path("config/transformer_pilot.json")


@given(
    visits=st.lists(st.integers(1, 3), min_size=1, max_size=30),
    fractions=st.tuples(st.floats(0.001, 1), st.floats(0.001, 1)),
)
def test_patient_fraction_nested_whole_patients(
    visits: list[int], fractions: tuple[float, float]
) -> None:
    """Podíly jsou vnořené, zaokrouhlené nahoru a zachovávají celé pacienty."""
    notes = tuple(
        note(90000001 + i, v, positive=True)
        for i, count in enumerate(visits)
        for v in range(count)
    )
    small, large = [patient_fraction(notes, f, 7) for f in sorted(fractions)]
    assert {n.subject_id for n in small} <= {n.subject_id for n in large}
    for fraction, subset in zip(sorted(fractions), (small, large), strict=True):
        patients = {n.subject_id for n in subset}
        assert len(patients) == ceil(fraction * len(visits))
        assert subset == tuple(n for n in notes if n.subject_id in patients)
    assert patient_fraction(notes, 1.0, 7) == notes


@pytest.mark.parametrize("fraction", [0.5, 0.3])
def test_patient_fraction_example(fraction: float) -> None:
    """Polovina čtyř i třicet procent čtyř znamená dva pacienty."""
    notes = tuple(note(i, 0, positive=True) for i in range(1, 5))
    assert len(patient_fraction(notes, fraction, 0)) == 2


@pytest.mark.parametrize(
    "change", ["duplicate", "zero", "large", "dictionary", "warmup", "empty"]
)
def test_config_rejects(change: str) -> None:
    """Odmítne neplatné podíly, preprocessing a nejednoznačné běhy."""
    config = load_pilot_config(CONFIG).model_dump(mode="json")
    match change:
        case "duplicate":
            config["runs"].append(config["runs"][0])
        case "zero" | "large":
            config["runs"][0]["fraction"] = 0 if change == "zero" else 1.5
        case "dictionary":
            config["runs"][0]["encoder"]["text_preprocessing"] = "dictionary"
        case "warmup":
            config["training"]["warmup_ratio"] = 1.0
        case "empty":
            config["runs"] = []
    import json  # ruff: ignore[import-outside-top-level]

    with pytest.raises(ValidationError):
        TransformerPilotConfig.model_validate_json(json.dumps(config))


def test_pilot_config_matches_stage_sample() -> None:
    """Pilot sdílí seed, vzorek a foldy všech stupňů i stop split sítí."""
    pilot = load_pilot_config(CONFIG)
    assert len(pilot.runs) == 8
    for path in Path("config").glob("stage_*.json"):
        stage = load_stage_config(path)
        assert (pilot.seed, pilot.subsample_parts, pilot.folds) == (
            stage.seed,
            stage.subsample_parts,
            stage.folds,
        )
    stage = load_stage_config(Path("config/stage_networks.json"))
    assert stage.network is not None
    assert pilot.stopping_parts == stage.network.stopping_parts


@pytest.mark.parametrize(("seed", "expected"), [(0, {3, 4}), (1, {2, 4})])
def test_patient_fraction_hash_order(seed: int, expected: set[int]) -> None:
    """SHA-256 má pro seed 0 pořadí 4,3,2,1 a pro seed 1 pořadí 4,2,3,1."""
    notes = tuple(note(i, 0, positive=True) for i in range(1, 5))
    assert {n.subject_id for n in patient_fraction(notes, 0.5, seed)} == expected


def test_patient_fraction_seed_changes_selection() -> None:
    """Dva seedy nad 30 pacienty vybírají různé poloviny, nikoli nejnižší ID."""
    notes = tuple(note(i, 0, positive=True) for i in range(1, 31))
    first, second = (
        {n.subject_id for n in patient_fraction(notes, 0.5, seed)} for seed in (0, 1)
    )
    assert first != second
    assert first != set(range(1, 16)) or second != set(range(1, 16))
