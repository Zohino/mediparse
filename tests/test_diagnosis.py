"""Diagnóza: pětice a jediná akutní diagnóza."""

from mediparse.domain.diagnosis import ACUTE_DIAGNOSIS, Diagnosis


def test_five_follow_notebook_8_with_one_acute_diagnosis() -> None:
    """Pětice sešitu 8 v pořadí kandidátů; jedinou akutní diagnózou je pneumonie."""
    assert [d.value for d in Diagnosis] == [
        "diabetes",
        "copd",
        "ckd",
        "pneumonia",
        "atrial_fibrillation",
    ]
    assert [d for d in Diagnosis if not d.chronic] == [ACUTE_DIAGNOSIS]
    assert ACUTE_DIAGNOSIS is Diagnosis.PNEUMONIA
