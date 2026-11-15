"""Preregistrované pravidlo non-inferiority a cenového srovnání modelů."""

import math
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.seed import Seed


class Verdict(StrEnum):
    """Tři výsledky posouzení non-inferiority klasického modelu."""

    NON_INFERIOR = "non_inferior"
    INFERIOR = "inferior"
    INCONCLUSIVE = "inconclusive"


class DecisionRule(BaseModel):
    """Pravidlo pro rozdíl macro-F1 klasika minus transformer.

    Meze pocházejí z oboustranného CI s hladinou ``confidence`` z párového
    cluster bootstrapu přes pacienty na stejných zprávách obou modelů.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    margin: Annotated[float, Field(gt=0, lt=1)]
    min_latency_ratio: Annotated[float, Field(ge=1)]
    min_training_cost_ratio: Annotated[float, Field(ge=1)]
    confidence: Annotated[float, Field(gt=0, lt=1)]
    bootstrap_resamples: Annotated[int, Field(ge=1000)]
    min_evaluation_notes: PositiveInt
    seed: Seed

    def verdict(self, ci_low: float, ci_high: float, evaluation_notes: int) -> Verdict:
        """Určí non-inferioritu z mezí CI s ostrou hranicí minus delta.

        Args:
            ci_low: Dolní mez CI rozdílu macro-F1 klasika minus transformer.
            ci_high: Horní mez téhož CI.
            evaluation_notes: Počet společně evaluovaných zpráv.

        Returns:
            Prokázaná, vyvrácená nebo nerozhodnutá non-inferiorita.

        Raises:
            InvalidInputError: Neplatné meze CI nebo příliš málo zpráv.
        """
        if not -1 <= ci_low <= ci_high <= 1:
            msg = "Neplatné meze CI: musí být v rozsahu [-1, 1] a dolní mez nesmí překročit horní."
            raise InvalidInputError(msg)
        if evaluation_notes < self.min_evaluation_notes:
            msg = f"Evaluace vyžaduje alespoň {self.min_evaluation_notes} zpráv."
            raise InvalidInputError(msg)
        if ci_low > -self.margin:
            return Verdict.NON_INFERIOR
        if ci_high < -self.margin:
            return Verdict.INFERIOR
        return Verdict.INCONCLUSIVE

    def is_cheaper(self, latency_ratio: float, training_cost_ratio: float) -> bool:
        """Ověří dosažení alespoň jednoho prahu úspory klasického modelu.

        Args:
            latency_ratio: Poměr latence p95 na zprávu transformer ku klasice.
            training_cost_ratio: Poměr nákladů tréninku v kWh transformer ku klasice.

        Returns:
            Zda alespoň jeden poměr dosahuje preregistrovaného prahu.

        Raises:
            InvalidInputError: Některý poměr není konečný nebo je záporný.
        """
        if not all(
            math.isfinite(r) and r >= 0 for r in (latency_ratio, training_cost_ratio)
        ):
            msg = "Neplatný poměr: musí být konečný a nezáporný."
            raise InvalidInputError(msg)
        return (
            latency_ratio >= self.min_latency_ratio
            or training_cost_ratio >= self.min_training_cost_ratio
        )
