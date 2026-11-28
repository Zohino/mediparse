"""Sdílené skládání adaptérů a pracovních argumentů nástrojů stupně."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.inputs import InvalidInputError
from mediparse.infrastructure.diagnosis_candidates_file import (
    text_normalization,
)
from mediparse.infrastructure.hf_encoder import ENCODER_PACKAGES, HfMeanEncoder
from mediparse.infrastructure.note_embeddings import NoteEmbeddings
from mediparse.infrastructure.package_versions import installed_versions
from mediparse.infrastructure.sklearn_stage import SklearnStageScorer
from mediparse.infrastructure.torch_device import (
    NETWORK_PACKAGES,
    DevicePool,
    require_device,
)

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Mapping

    from mediparse.domain.clinical_text import TextNormalization
    from mediparse.domain.representation import EncoderSpec
    from mediparse.domain.stage_tuning import StageTuningConfig


PACKAGES: Final = (
    "python",
    "numpy",
    "scipy",
    "scikit-learn",
    "lightgbm",
    "fasttext",
    "gensim",
    "joblib",
    "threadpoolctl",
    "optuna",
    "pyarrow",
    "pydantic",
)


def check_work_paths(*, scratch: Path, embeddings: Path, checkpoint: Path) -> None:
    """Ověří oddělení dočasných souborů od checkpointu.

    Raises:
        InvalidInputError: Scratch nebo cache leží uvnitř checkpointu.
    """
    # Pracovní adaptéry zakládají adresáře dřív než checkpoint zapíše svůj klíč.
    for path in (scratch, embeddings):
        if path.resolve().is_relative_to(checkpoint.resolve()):
            msg = "Scratch a cache embeddingů musí ležet mimo adresář checkpointu."
            raise InvalidInputError(msg)


def stage_normalizations(
    candidates: Path, config: StageTuningConfig
) -> dict[str, TextNormalization | None]:
    """Načte slovníky reprezentací společné pro scorer i manifest.

    Returns:
        Normalizace podle jména reprezentace.
    """
    return {
        name: text_normalization(candidates, level)
        for name, level in config.text_preprocessing.items()
    }


def stage_packages(config: StageTuningConfig) -> dict[str, str]:
    """Ověří zařízení a zjistí verze knihoven stupně.

    Returns:
        Balíčky pro manifest a klíč checkpointu.
    """
    for spec in config.encoders:
        require_device(spec.device)
    if config.network is not None:
        require_device(config.network.device)
    versions = installed_versions(ENCODER_PACKAGES) if config.encoders else {}
    packages = installed_versions(PACKAGES) | versions
    if config.network is not None:
        packages |= installed_versions(NETWORK_PACKAGES)
    return packages


# Cesty, encoder a počet procesů jsou nezávislé provozní vstupy composition root.
def stage_scorer(  # ruff: ignore[too-many-arguments]
    config: StageTuningConfig,
    normalizations: Mapping[str, TextNormalization | None],
    *,
    embeddings: Path,
    scratch: Path,
    versions: Mapping[str, str],
    encoders: Callable[[EncoderSpec], HfMeanEncoder],
    jobs: int,
) -> SklearnStageScorer:
    """Složí scorer se společnou cache embeddingů.

    Returns:
        Adaptér pro kontrolu i ladění bodů stupně.
    """
    return SklearnStageScorer(
        config,
        normalizations,
        config.seed,
        scratch,
        NoteEmbeddings(
            embeddings,
            {name: versions[name] for name in ENCODER_PACKAGES}
            if config.encoders
            else {},
            lambda spec: encoders(spec).embed,
        ),
        jobs=jobs,
        devices=DevicePool.of(config.encoders[0].device, config.concurrent_trials)
        if config.transformer is not None
        else None,
    )


def add_stage_work_arguments(parser: argparse.ArgumentParser) -> None:
    """Přidá společné cesty cache, scratch a počet nezávislých procesů."""
    parser.add_argument(
        "--embeddings",
        type=Path,
        required=True,
        help="Cache embeddingů; doporučeno resources/mimic/screening/embeddings, mimo results/.",
    )
    parser.add_argument(
        "--scratch",
        type=Path,
        required=True,
        help="Dočasné tokeny zpráv a memmapy matic paralelních fitů; doporučeno resources/mimic/screening/scratch, mimo checkpoint a results/.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        choices=range(1, len(Diagnosis) + 1),
        default=3,
        help=(
            "Procesy nezávislých fitů pokusu: diagnózy u klasiky a fastTextu, "
            "foldy u MultiResCNN (nejvýš jeden fold na GPU, fold i na GPU i mod počet GPU). "
            "Výsledky na počtu nezávisí. Stupeň 4 --jobs nepoužívá; souběh řídí concurrent_trials v configu."
        ),
    )
