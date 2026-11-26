"""Kontrola explicitního výpočetního zařízení před drahými pokusy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from mediparse.domain.representation import Device


NETWORK_PACKAGES: Final = ("torch", "transformers")


def require_device(device: Device) -> None:
    """Odmítne chybějící torch nebo nedostupnou CUDA.

    Raises:
        InvalidInputError: Prostředí nepodporuje požadované zařízení.
    """
    try:
        # Výchozí prostředí záměrně torch neobsahuje.
        import torch  # ruff: ignore[import-outside-top-level]
    except ImportError:
        msg = "Chybí torch; spusť přes pixi run --frozen -e gpu mediparse-stage-tuning."
        raise InvalidInputError(msg) from None
    if device == "cuda" and not torch.cuda.is_available():
        msg = "CUDA není dostupná; změň device v configu (změní se i klíč cache)."
        raise InvalidInputError(msg)


def cuda_device_count() -> int:
    """Zjistí počet GPU viditelných rodičovskému procesu.

    Returns:
        Počet CUDA zařízení.
    """
    import torch  # ruff: ignore[import-outside-top-level]

    return torch.cuda.device_count()


def fold_device(device: Device, fold: int, count: int) -> str:
    """Přiřadí fold zařízení cyklicky podle jeho pořadí.

    Returns:
        CPU nebo explicitní CUDA zařízení.

    Raises:
        InvalidInputError: Pro CUDA není k dispozici žádná GPU.
    """
    if device == "cpu":
        return device
    if count == 0:
        msg = "Souběžné foldy vyžadují alespoň jednu GPU."
        raise InvalidInputError(msg)
    return f"cuda:{fold % count}"
