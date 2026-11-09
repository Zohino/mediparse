"""Pevná syntetická provenance pro testy manifestů a jejich kontraktu s R."""

from mediparse.domain.environment import Environment, Hardware
from mediparse.domain.model_input import TextPreprocessing
from mediparse.domain.run_manifest import RunManifest
from tests.support import REVISION


def training_manifest(
    *,
    row_id: str = "row",
    predictions_sha256: str = "a" * 64,
    dataset_sha256: str = "b" * 64,
    split_sha256: str = "c" * 64,
) -> RunManifest:
    """Sestaví úplný manifest z pevných hodnot a dodaných otisků.

    Returns:
        Syntetický manifest bez závislosti na stroji a instalovaných verzích.
    """
    return RunManifest(
        row_id=row_id,
        predictions_sha256=predictions_sha256,
        dataset_sha256=dataset_sha256,
        split_sha256=split_sha256,
        config_sha256="d" * 64,
        seed=0,
        split_seed=0,
        split_folds=2,
        diagnostic_strip=None,
        text_preprocessing=TextPreprocessing.NONE,
        text_normalization=None,
        model="synthetic classifier",
        model_revision=None,
        tokenizer_revision=None,
        environment=Environment(packages={"python": "3.13.0"}, source=REVISION),
        hardware=Hardware(
            cpu_model="Synthetic CPU",
            logical_cores=8,
            available_cores=2,
            memory_bytes=1024,
            gpus=(),
        ),
    )
