"""Evaluace: predikce zpráv, jejich skóre a metriky na diagnózu a řádek matice."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mediparse.domain.inputs import InvalidInputError
from mediparse.domain.note import ensure_note_of_subject

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mediparse.domain.diagnosis import Diagnosis
    from mediparse.domain.identity import RowId, SubjectId
    from mediparse.domain.input_note import InputNote
    from mediparse.domain.split import Split


@dataclass(frozen=True)
class Classification:
    """Výstup klasifikátoru pro jeden text: label a rozhodovací skóre."""

    predicted: bool
    score: float


@dataclass(frozen=True)
class Prediction:
    """Predikce jedné diagnózy pro jednu testovací zprávu řádku matice."""

    row_id: RowId
    fold: int
    note_id: str
    subject_id: SubjectId
    diagnosis: Diagnosis
    truth: bool
    predicted: bool
    score: float

    def __post_init__(self) -> None:
        """Ověří, že note_id patří pacientovi predikce, aby párování po pacientech sedělo.

        Raises:
            InvalidNoteIdError: Note_id je mimo skladbu MIMIC-IV-Note.
            NoteSubjectMismatchError: Note_id patří jinému pacientovi než ``subject_id``.
        """
        ensure_note_of_subject(self.note_id, self.subject_id)


@dataclass(frozen=True)
class BinaryMetrics:
    """Held-out metriky binárního klasifikátoru pro pozitivní třídu."""

    precision: float
    recall: float
    f1: float
    accuracy: float
    roc_auc: float
    pr_auc: float


@dataclass(frozen=True)
class DiagnosisMetrics:
    """Metriky jedné diagnózy na jednom řádku matice s velikostí testu."""

    row_id: RowId
    diagnosis: Diagnosis
    test_notes: int
    test_positives: int
    metrics: BinaryMetrics


def ensure_both_classes(diagnosis: Diagnosis, truth: Sequence[bool]) -> None:
    """Ověří, že pravda diagnózy obsahuje pozitivní i negativní zprávu.

    Args:
        diagnosis: Diagnóza, pro kterou se počítají metriky.
        truth: Pravda testovacích zpráv.

    Raises:
        InvalidInputError: Pravda má jedinou třídu, AUC by nebyla definovaná.
    """
    positives = sum(truth)
    if 0 < positives < len(truth):
        return
    msg = (
        f"Diagnóza {diagnosis}: {positives} pozitivních z {len(truth)} zpráv, "
        "metriky potřebují obě třídy."
    )
    raise InvalidInputError(msg)


def ensure_unique_keys(predictions: Sequence[Prediction]) -> None:
    """Ověří, že klíč (row_id, note_id, diagnóza) se v predikcích neopakuje.

    Fold v klíči není: evaluace foldy slévá a zpráva ve dvou foldech by se započetla
    dvakrát.

    Args:
        predictions: Predikce ke zpracování.

    Raises:
        InvalidInputError: Některý klíč se opakuje.
    """
    seen: set[tuple[str, str, Diagnosis]] = set()
    for item in predictions:
        key = (item.row_id, item.note_id, item.diagnosis)
        if key in seen:
            # note_id nese subject_id pacienta MIMIC; hláška ani log ho nesmí nést.
            msg = (
                f"Opakovaná predikce zprávy: řádek {item.row_id}, "
                f"diagnóza {item.diagnosis}."
            )
            raise InvalidInputError(msg)
        seen.add(key)


def ensure_complete_folds(
    predictions: Sequence[Prediction], notes: Sequence[InputNote], split: Split
) -> None:
    """Ověří, že predikce každého foldu pokrývají přesně testovací zprávy foldu.

    Kontrolují se jen foldy přítomné v predikcích; že jsou přítomné všechny, není
    věc této funkce. Duplicitu klíče nevidí (množina ji sloučí), proto má běžet po
    ``ensure_unique_keys``.

    Args:
        predictions: Predikce ke zpracování.
        notes: Zprávy korpusu, jejichž pacienti jsou všichni ve splitu.
        split: Split, který určuje test každého foldu.

    Raises:
        InvalidInputError: Skupina (řádek, diagnóza, fold) nemá zprávu testu foldu
            nebo má cizí; fold je mimo rozsah splitu nebo zpráva pacienta mimo split.
    """
    predicted: defaultdict[tuple[str, Diagnosis, int], set[str]] = defaultdict(set)
    for item in predictions:
        predicted[item.row_id, item.diagnosis, item.fold].add(item.note_id)
    expected: dict[int, frozenset[str]] = {}
    for (row_id, diagnosis, fold), names in sorted(predicted.items()):
        if fold not in expected:
            test = split.holdout(notes, fold).test
            expected[fold] = frozenset(note.note_id for note in test)
        missing, extra = len(expected[fold] - names), len(names - expected[fold])
        if missing or extra:
            # note_id nese subject_id pacienta MIMIC; hláška ani log ho nesmí nést.
            msg = (
                f"Predikce neodpovídají testu foldu: řádek {row_id}, "
                f"diagnóza {diagnosis}, fold {fold}, chybí {missing}, navíc {extra}."
            )
            raise InvalidInputError(msg)
