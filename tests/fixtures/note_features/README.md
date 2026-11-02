# Zkonstruované zprávy testů R

Tyto tabulky obsahují výhradně zkonstruované texty testů `test-segments.R`,
`test-bhc.R`, `test-preprocessing.R` a `test-ngrams.R`, včetně výplně z `corpus_con`.
Vstupy byly zaznamenány při běhu původních testů před odstraněním mostu
(415 + 55 úspěšných kontrol). Žádný text nepochází z MIMIC.

`notes.parquet` mapuje text na lokální klíč fixtury. Ostatní tři tabulky mají
stejný kontrakt jako fixtury v `tests/fixtures/contracts/`; zapsal je
`ParquetNoteFeatures` nad těmito vstupy přes `ComputeNoteFeatures`.
Pytest kontroluje shodu se zapisovačem. Pomocník R jen přemapuje klíče podle
textů aktuálního testu; parser ani analyzér v R neběží. Neznámý vstup odmítne.

Při přidání zkonstruovaného případu doplňte `notes.parquet` a znovu zapište
všechny tři artefakty stejným zapisovačem. Očekávání agregací v R se nemění
podle výstupu tohoto výpočtu.
