# EDA v Quartu

Průzkum dat žije v Quarto projektu `eda/`: jeden dokument `eda/<n>-*.qmd` na sešit. R vede dokument (statistiky, tabulky, inline `` `r …` ``), Python běží v blocích přes reticulate ze stejného prostředí pixi `eda`. Obě strany dotazují parquet přes DuckDB. Dokumenty slouží hodnotiteli a zároveň jako podklad rozhodnutí pro pipeline.

Příloha s provenance a seznam literatury jsou ve sdíleném `eda/_partials/appendix.qmd`, který sešity vkládají přes `{{< include _partials/appendix.qmd >}}`. Podadresář s podtržítkem nepočítá CI ani `eda_render` a `quarto render eda` ho přeskočí. Změna partialu mění všechny sešity, proto vyžaduje nový reálný render každého z nich.

## Forma dokumentů

Dokumenty mají akademickou formu: abstrakt, číslované kapitoly, obsah, číslované tabulky s titulkem (`#| label: tbl-…`, odkaz `@tbl-…`), citace `[@klíč]` z `eda/references.bib` a seznam literatury. Sdílená metadata (autor, bibliografie, obsah, číslování, A4) jsou v `eda/_quarto.yml`. Citace zpracovává citeproc (styl autor–rok), protože Typst v Quartu 1.9 nadepisuje vlastní bibliografii „Bibliografie“; kapitola „Literatura“ je proto v dokumentu ručně. Počty v tabulkách jdou přes `dua_table`, rozměry tabulek z veřejného manifestu validace se vkládají jako text.

## Rendery

- `pixi run eda <n>` vyrenderuje `eda/<n>-*.qmd` nad skutečnými tabulkami. Pouští ho jen uživatel ve vlastním terminálu. Skript odmítne běh pod Claude Code (`CLAUDECODE`) a při necommitnutých změnách, aby provenance neukazovala na jiný kód.
- `pixi run eda-synthetic [n]` vyrenderuje jeden nebo všechny dokumenty nad syntetickými tabulkami v dočasném adresáři a zkontroluje kanárka. Na `eda/_freeze/` v repu nesahá. Na konci vypíše cestu k PDF.
- `pixi run eda-test` pustí testy R pomocníků (testthat).

Tabulky parquet se čtou z `MEDIPARSE_PARQUET` (výchozí `resources/mimic/parquet`). Dočasné soubory DuckDB jdou do `duckdb-tmp` vedle toho adresáře, limit paměti určuje `MEDIPARSE_DUCKDB_MEMORY` (výchozí `8GB`).

Oba rendery běží nad kopií `eda/` v dočasném adresáři, ne na místě. Quarto v cestě se skrytou složkou (třeba `.worktrees/`) tiše ignoruje `_quarto.yml` a vyrenderuje výchozí HTML bez freeze, takže `quarto render eda` přímo ve worktree pod `.worktrees/` nic neudělá. Render na místě navíc zakládá v `eda/` netrackované soubory (`.gitignore`, mezisoubory knitr), kvůli kterým by provenance hlásila dirty. Reálný render proto pracuje vedle adresáře parquetu (pod `resources/mimic/`), zkopíruje zpět jen `eda/_freeze/<doc>/`, `results/eda/<doc>.pdf` a `results/eda/<doc>/` (starý adresář exportů v repu nejdřív smaže, aby zrušený export nezůstal) a po sobě uklidí. Skripty po renderu ověří, že PDF i freeze vznikly.

Pořadí reálného renderu: nejdřív commit `.qmd`, pak `pixi run eda <n>`, pak commit `eda/_freeze/`, `results/eda/<n>-*.pdf` a `results/eda/<n>-*/` (exporty). Jinak by provenance hlásila dirty strom. Commituje se PDF a zmražené výsledky, HTML ne.

## Zmražené výsledky

Projekt má `freeze: auto`. `quarto render eda` (celý projekt) použije zmražené výsledky a data nepotřebuje. Render jednoho souboru kód vždy spustí. CI proto renderuje celý projekt bez dat: změněný `.qmd` se zastaralým freeze sáhne po datech a CI spadne, takže zastaralý freeze se nedostane na main.

## Ochrana dat MIMIC

- Kategorie s n < 10 se potlačují. `dua_table` v `eda/R/dua.R` vyžaduje agregovanou tabulku se sloupcem `n`, řádky pod prahem sloučí do jednoho řádku „ostatní (n < 10)“ a součet ukáže přes `dua_count`. Neagregovaná tabulka skončí chybou. Popisky smí být jen text nebo faktor; další číselný sloupec vedle `n` se odmítne, protože by nešel potlačit.
- `dua_table` se skládá z `dua_suppress` (potlačení s číselným `n`, z něhož se kreslí grafy) a `dua_count` (formátování). `dua_percent(n, total)` vrací podíl s desetinnou čárkou jako text a „–“, když je čitatel pod prahem nebo jmenovatel nulový. `dua_share(df)` je potlačená tabulka se sloupcem podílu z celku před potlačením.
- Dotazy sešitu 2 (pacienti, pokrytí, éra ICD) jsou v `eda/R/patients.R` a testuje je testthat nad parquetem vyrobeným v testu, protože syntetika neobsahuje NULL ani ICD-9 a ICD-10.
- `eda_connect()` odmítne adresář bez `synthetic.json`, když je nastavená proměnná `CLAUDECODE` nebo `CI`: skutečná data se tam číst nesmí.
- Dokument dostává připojení z `eda_connect()`: dočasné soubory DuckDB leží pod `resources/mimic/` a paměť je omezená.
- `eda_check_python()` ověří, že reticulate použil Python z prostředí pixi. Prostředí obsahuje `micromamba` a nastavuje `RETICULATE_PYTHON` a `RETICULATE_CONDA`, jinak reticulate hledá binárku conda nebo tiše sáhne po cizím Pythonu.
- Do dokumentů patří jen agregáty. Řádková data a texty z MIMIC se nevypisují.

## Kanárek

`eda-synthetic` vygeneruje z veřejného manifestu validace (`results/mimic/manifest.json`) tabulky ve tvaru MIMIC, kde je každá buňka jedinečná řetězec `eda-sentinel-<tabulka>-<sloupec>-<řádek>`. Každá kategorie má proto n = 1 a únik popisku kategorie je selhání potlačení. Po renderu skript převede PDF na text (`pdftotext`) a hledá `eda-sentinel-` v kopii `eda/` (včetně freeze), ve výstupu a v textu PDF. Nalezený soubor znamená návratový kód 1 s jeho cestou. Hledá se prefix s pomlčkou, protože provenance prefix uvádí v popisu zdroje dat.

Kanárek čte i exporty: `.tex` fragmenty hledá `grep` nad `results/`, text grafů z exportovaných PDF (`pdftotext` nad všemi PDF pod `results/eda` rekurzivně). Typst sešitu vykresluje graf jako PNG, ale exportované PDF vzniká ze stejného objektu, takže kanárek vidí osy i popisky kategorií. Mezera „text v obrázcích“ tím zaniká.

## Exporty pro text práce

Každý graf a každá tabulka sešitu se ukládá i pro LaTeX do `results/eda/<sešit>/`. Text práce je odkazuje přes `../results/eda/` (`\graphicspath`, `\input`) a nekopíruje je. Soubor se jmenuje podle labelu chunku (`tbl-korpus.tex`, `fig-….pdf`); label musí odpovídat `^[a-z0-9-]+$`. `\caption`, `\label` a float píše text práce.

- `export_table(df, col.names, align, code = character())` (`eda/R/export.R`) zapíše fragment `tabular` s `booktabs` bez floatu a vrátí `knitr::kable` ze stejného data.frame pro Typst, chunk tedy volá jen tuto funkci. Šířky sloupců bere z `#| tbl-colwidths` chunku a sloupec se šířkou sází jako `p{\dimexpr w\linewidth-2\tabcolsep\relax}`; bez nich zůstane prosté `l/c/r`. Sloupce v `code` se v Typstu obalí backticky, v LaTeXu `\texttt{}`. Tabulky se exportují všechny, včetně provenance.
- `export_figure(plot, height_cm)` uloží `ggplot` jako vektorové PDF (`cairo_pdf`, šířka sazby 16,5 cm, výška v cm) a vrátí graf, aby ho chunk vykreslil do sešitu. Základem je `theme_thesis()`: Libertinus Serif, 10 pt, graf se v práci vkládá bez škálování.
- Písmo Libertinus Serif musí být v systému (Fedora `libertinus-fonts`, Debian `fonts-libertinus`); conda-forge ho nemá a Ubuntu ho samostatně nebalí. Jinde ho `scripts/install-libertinus.sh` stáhne z vydání Libertinus 7.051 na GitHubu, ověří otisk SHA-256 a rozbalí řez Serif do `~/.local/share/fonts`, kde ho najde fontconfig z prostředí pixi. Tak ho instaluje i CI. `export_figure` přes `pdffonts` ověří, že ho PDF vložilo, protože cairo jinak tiše dosadí jiné.
- Exporty jdou ze stejného objektu jako sešit, po potlačení n < 10. Render z freeze exporty nepřepisuje ani nemaže: commitnuté `results/eda/<sešit>/` jsou výsledek posledního reálného renderu. Změna `.qmd` proto vyžaduje nový `pixi run eda <n>` a commit freeze i exportů, jinak CI krok „render z freeze“ spadne.

## Provenance

Každý dokument končí přílohou s commitem, příznakem dirty, otiskem `pixi.lock`, zdrojem dat (otisky CSV z `config/mimic_tables.json`, u syntetiky prefix) a verzemi Pythonu, DuckDB a R. Skládá ji `mediparse.entrypoints.eda_provenance`. Při syntetickém renderu se commit předává proměnnými `MEDIPARSE_COMMIT` a `MEDIPARSE_DIRTY`, protože kopie v dočasném adresáři není repozitář git.
