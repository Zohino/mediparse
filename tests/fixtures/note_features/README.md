# Zkonstruované zprávy testů R

Tyto tabulky obsahují výhradně zkonstruované texty testů `test-segments.R`,
`test-bhc.R`, `test-preprocessing.R`, `test-ngrams.R` a `test-spelling.R`, včetně výplně z `corpus_con`.
Vstupy byly zaznamenány při běhu původních testů před odstraněním mostu
(415 + 55 úspěšných kontrol). Žádný text nepochází z MIMIC.

`notes.parquet` mapuje text na lokální klíč fixtury. Ostatní čtyři tabulky mají
stejný kontrakt jako fixtury v `tests/fixtures/contracts/`; zapsal je
`ParquetNoteFeatures` nad těmito vstupy přes `ComputeNoteFeatures`.
Pytest kontroluje shodu se zapisovačem. Pomocník R jen přemapuje klíče podle
textů aktuálního testu; parser ani analyzér v R neběží. Neznámý vstup odmítne.

Při přidání zkonstruovaného případu doplňte `notes.parquet` a znovu zapište
všechny čtyři artefakty stejným zapisovačem. Očekávání agregací v R se nemění
podle výstupu tohoto výpočtu.

`note_variants.parquet` nese všechny varianty z pole `variants_en` v `config/diagnosis_candidates.json`
včetně nul, a měří tak varianty názvů diagnóz ve stejných zprávách. Obě sady fixtur se obnoví z kořene repozitáře příkazem:

```bash
pixi run --frozen python - <<'PYTHON'
from pathlib import Path
import pyarrow.parquet as pq
from mediparse.infrastructure.note_features_table import ParquetNoteFeatures
from tests.note_features_support import CONTRACT_NOTES, compute

with ParquetNoteFeatures(Path("tests/fixtures/contracts")).writer() as sink:
    compute(CONTRACT_NOTES, sink)
directory = Path("tests/fixtures/note_features")
notes = pq.read_table(directory / "notes.parquet").to_pylist()
with ParquetNoteFeatures(directory).writer() as sink:
    compute([(row["note_id"], row["text"]) for row in notes], sink)
PYTHON
```

Očekávání testů jsou nezávislá na generování fixtur a tímto příkazem se nemění.
