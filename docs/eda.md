# EDA v Quartu

Průzkum dat žije v Quarto projektu `eda/`: jeden dokument `eda/<n>-*.qmd` na sešit. R vede dokument (statistiky, tabulky, inline `` `r …` ``), Python běží v blocích přes reticulate ze stejného prostředí pixi `eda`. Obě strany dotazují parquet přes DuckDB. Dokumenty slouží hodnotiteli a zároveň jako podklad rozhodnutí pro pipeline.

## Forma dokumentů

Dokumenty mají akademickou formu: abstrakt, číslované kapitoly, obsah, číslované tabulky s titulkem (`#| label: tbl-…`, odkaz `@tbl-…`), citace `[@klíč]` z `eda/references.bib` a seznam literatury. Sdílená metadata (autor, bibliografie, obsah, číslování, A4) jsou v `eda/_quarto.yml`. Citace zpracovává citeproc (styl autor–rok), protože Typst v Quartu 1.9 nadepisuje vlastní bibliografii „Bibliografie“; kapitola „Literatura“ je proto v dokumentu ručně. Počty v tabulkách jdou přes `dua_table`, rozměry tabulek z veřejného manifestu validace se vkládají jako text.

## Rendery

- `pixi run eda <n>` vyrenderuje `eda/<n>-*.qmd` nad skutečnými tabulkami. Pouští ho jen uživatel ve vlastním terminálu. Skript odmítne běh pod Claude Code (`CLAUDECODE`) a při necommitnutých změnách, aby provenance neukazovala na jiný kód.
- `pixi run eda-synthetic [n]` vyrenderuje jeden nebo všechny dokumenty nad syntetickými tabulkami v dočasném adresáři a zkontroluje kanárka. Na `eda/_freeze/` v repu nesahá. Na konci vypíše cestu k PDF.
- `pixi run eda-test` pustí testy R pomocníků (testthat).

Tabulky parquet se čtou z `MEDIPARSE_PARQUET` (výchozí `resources/mimic/parquet`). Dočasné soubory DuckDB jdou do `duckdb-tmp` vedle toho adresáře, limit paměti určuje `MEDIPARSE_DUCKDB_MEMORY` (výchozí `8GB`).

Oba rendery běží nad kopií `eda/` v dočasném adresáři, ne na místě. Quarto v cestě se skrytou složkou (třeba `.worktrees/`) tiše ignoruje `_quarto.yml` a vyrenderuje výchozí HTML bez freeze, takže `quarto render eda` přímo ve worktree pod `.worktrees/` nic neudělá. Render na místě navíc zakládá v `eda/` netrackované soubory (`.gitignore`, mezisoubory knitr), kvůli kterým by provenance hlásila dirty. Reálný render proto pracuje vedle adresáře parquetu (pod `resources/mimic/`), zkopíruje zpět jen `eda/_freeze/<doc>/` a `results/eda/<doc>.pdf` a po sobě uklidí. Skripty po renderu ověří, že PDF i freeze vznikly.

Pořadí reálného renderu: nejdřív commit `.qmd`, pak `pixi run eda <n>`, pak commit `eda/_freeze/` a `results/eda/<n>-*.pdf`. Jinak by provenance hlásila dirty strom. Commituje se PDF a zmražené výsledky, HTML ne.

## Zmražené výsledky

Projekt má `freeze: auto`. `quarto render eda` (celý projekt) použije zmražené výsledky a data nepotřebuje. Render jednoho souboru kód vždy spustí. CI proto renderuje celý projekt bez dat: změněný `.qmd` se zastaralým freeze sáhne po datech a CI spadne, takže zastaralý freeze se nedostane na main.

## Ochrana dat MIMIC

- Kategorie s n < 10 se potlačují. `dua_table` v `eda/R/dua.R` vyžaduje agregovanou tabulku se sloupcem `n`, řádky pod prahem sloučí do jednoho řádku „ostatní (n < 10)“ a součet ukáže přes `dua_count`. Neagregovaná tabulka skončí chybou. Popisky smí být jen text nebo faktor; další číselný sloupec vedle `n` se odmítne, protože by nešel potlačit.
- `eda_connect()` odmítne adresář bez `synthetic.json`, když je nastavená proměnná `CLAUDECODE` nebo `CI`: skutečná data se tam číst nesmí.
- Dokument dostává připojení z `eda_connect()`: dočasné soubory DuckDB leží pod `resources/mimic/` a paměť je omezená.
- `eda_check_python()` ověří, že reticulate použil Python z prostředí pixi. Prostředí obsahuje `micromamba` a nastavuje `RETICULATE_PYTHON` a `RETICULATE_CONDA`, jinak reticulate hledá binárku conda nebo tiše sáhne po cizím Pythonu.
- Do dokumentů patří jen agregáty. Řádková data a texty z MIMIC se nevypisují.

## Kanárek

`eda-synthetic` vygeneruje z veřejného manifestu validace (`results/mimic/manifest.json`) tabulky ve tvaru MIMIC, kde je každá buňka jedinečná řetězec `eda-sentinel-<tabulka>-<sloupec>-<řádek>`. Každá kategorie má proto n = 1 a únik popisku kategorie je selhání potlačení. Po renderu skript převede PDF na text (`pdftotext`) a hledá `eda-sentinel-` v kopii `eda/` (včetně freeze), ve výstupu a v textu PDF. Nalezený soubor znamená návratový kód 1 s jeho cestou. Hledá se prefix s pomlčkou, protože provenance prefix uvádí v popisu zdroje dat.

Mezera: text uvnitř obrázků grafů kanárek nevidí. Vyřeší se s prvním grafem v S20c.

## Provenance

Každý dokument končí přílohou s commitem, příznakem dirty, otiskem `pixi.lock`, zdrojem dat (otisky CSV z `config/mimic_tables.json`, u syntetiky prefix) a verzemi Pythonu, DuckDB a R. Skládá ji `mediparse.entrypoints.eda_provenance`. Při syntetickém renderu se commit předává proměnnými `MEDIPARSE_COMMIT` a `MEDIPARSE_DIRTY`, protože kopie v dočasném adresáři není repozitář git.
