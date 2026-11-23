"""Kontrola explicitního výpočetního zařízení před drahými pokusy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mediparse.domain.inputs import InvalidInputError

if TYPE_CHECKING:
    from mediparse.domain.representation import Device


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
