"""Sdílený syntetický korpus screeningu se všemi pěti diagnózami."""

from random import Random

from mediparse.domain.diagnosis import Diagnosis
from mediparse.domain.input_note import InputNote
from mediparse.domain.language import Language
from mediparse.domain.split import SharedSplit, Split, SplitConfig

DEVELOPMENT = frozenset(range(90000001, 90000021))
SHARED = SharedSplit(
    SplitConfig(seed=23, folds=2),
    Split((frozenset({1}), frozenset({2})), development=DEVELOPMENT),
)


def note(subject: int, visit: int, *, positive: bool) -> InputNote:
    """Sestaví syntetickou zprávu.

    Returns:
        Zpráva s doplňkovými labely dvou tříd pacientů.
    """
    labels = frozenset(Diagnosis) - {Diagnosis.CKD} if positive else {Diagnosis.CKD}
    return InputNote(
        f"{subject}-DS-{visit}",
        subject,
        Language.EN,
        f"synthetic {'positive' if positive else 'negative'} {subject} visit {visit}",
        frozenset(labels),
    )


def notes(*, visits: int = 2) -> tuple[InputNote, ...]:
    """Sestaví korpus screeningu.

    Args:
        visits: Počet zpráv každého vývojového pacienta.

    Returns:
        Promíchané zprávy vývojových i evaluačních pacientů.
    """
    notes = [
        note(subject, visit, positive=subject % 2 == 1)
        for subject in sorted(DEVELOPMENT)
        for visit in range(visits)
    ]
    notes.extend((note(1, 0, positive=True), note(2, 0, positive=False)))
    Random(19).shuffle(notes)
    return tuple(notes)
