"""Překlad doménových zpráv pilotu na trénink transformeru."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.transformer_pilot import PilotEpoch
from mediparse.domain.tuning import ValidationScores
from mediparse.infrastructure.hf_classifier import HfWindowClassifier
from mediparse.infrastructure.sklearn_split import label_matrix

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from mediparse.domain.input_note import InputNote
    from mediparse.domain.seed import Seed
    from mediparse.domain.transformer_pilot import PilotRun, PilotTraining


@dataclass(frozen=True)
class HfPilotTrainer:
    """Adaptér tréninku se seedem společným pro všechny varianty."""

    seed: Seed

    def train(
        self,
        run: PilotRun,
        training: PilotTraining,
        fit: Sequence[InputNote],
        stop: Sequence[InputNote],
        validation: Sequence[str],
    ) -> Iterator[PilotEpoch]:
        """Naučí jednu variantu a předá měření v doménových typech.

        Yields:
            Skóre po diagnózách ve vstupním pořadí validace.
        """
        classifier = HfWindowClassifier(
            encoder=run.encoder,
            aggregation=run.aggregation,
            freeze_embeddings=run.freeze_embeddings,
            random_state=self.seed,
            device=run.encoder.device,
            **training.model_dump(),
        )
        for epoch in classifier.train_epochs(
            [n.text for n in fit],
            label_matrix(fit),
            [n.text for n in stop],
            label_matrix(stop),
            validation,
        ):
            yield PilotEpoch(
                epoch.epoch,
                epoch.seconds,
                epoch.validation_seconds,
                epoch.peak_memory_bytes,
                epoch.stop_loss,
                ValidationScores(
                    {
                        d: tuple(epoch.validation_scores[:, j])
                        for j, d in enumerate(Diagnosis)
                    },
                    converged=math.isfinite(epoch.stop_loss),
                    n_features=classifier.model_.config.hidden_size,
                ),
            )
