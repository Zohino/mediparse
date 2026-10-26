# EDA v Quartu

Průzkum dat žije v Quarto projektu `eda/`: jeden dokument `eda/<n>-*.qmd` na sešit. Sešit 1 běží na enginu jupyter v Pythonu, sešity 2–9 zatím na knitru (R s Pythonem přes reticulate). Všechny používají prostředí pixi `eda` a dotazují parquet přes DuckDB. Dokumenty slouží hodnotiteli a zároveň jako podklad rozhodnutí pro pipeline.

Příloha s provenance a seznam literatury jsou pro sešity na knitru ve sdíleném `eda/_partials/appendix.qmd`, který sešity vkládají přes `{{< include _partials/appendix.qmd >}}`. Podadresář s podtržítkem nepočítá CI ani `eda_render` a `quarto render eda` ho přeskočí. Změna partialu mění všechny sešity, které jej vkládají, proto vyžaduje nový reálný render každého z nich. Pro sešity na jupyteru je připravený samostatný `eda/_partials/appendix-python.qmd` se stejným nadpisem a literaturou; vloží se přes `{{< include _partials/appendix-python.qmd >}}`.

## Forma dokumentů

Dokumenty mají akademickou formu: abstrakt, číslované kapitoly, obsah, číslované tabulky s titulkem (v R `#| label: tbl-…`, v Pythonu argument `label` exportéru; odkaz `@tbl-…`), citace `[@klíč]` z `eda/references.bib` a seznam literatury. Sdílená metadata (autor, bibliografie, obsah, číslování, A4) jsou v `eda/_quarto.yml`. Citace zpracovává citeproc (styl autor–rok), protože Typst v Quartu 1.9 nadepisuje vlastní bibliografii „Bibliografie“; kapitola „Literatura“ je proto v dokumentu ručně. V sešitech na knitru formátují počty a podíly v tabulkách `number_text`, `percent_text` a `share_table` z `eda/R/dua.R`, rozměry tabulek z veřejného manifestu validace se vkládají jako text.

## Engine po sešitech

`eda/_quarto.yml` ponechává `engine: knitr` pro nepřepsané sešity a určuje
`jupyter: python3` pro všechny sešity na jupyteru. Přepsaný sešit nastaví ve front
matter `engine: jupyter`, jako `eda/1-dataset.qmd`. Kernel `python3` se bere
z prostředí pixi; bez explicitní volby by Quarto mohlo vybrat starý uživatelský
kernel s neexistujícím interpretem. K renderu stačí balíky `ipykernel` a `nbclient`,
celý balík `jupyter` s notebookem a labem není potřeba.

Sešity a R pomocníci vstupují do `mediparse` jen přes `mediparse.entrypoints`.
`tests/test_architecture.py` kontroluje importy v `.qmd` i `.R`, zakazuje přímý
import `duckdb` a `pyarrow` v `.qmd` a ověřuje obalení Python inline výrazů do
`Markdown(...)`. Import-linter tyto formáty nečte. V kontejneru se kontrola EDA
přeskočí, protože adresář `eda/` v obrazu není; hlídá ji job Test v CI.

## Rendery

- `pixi run eda <n>` vyrenderuje `eda/<n>-*.qmd` nad skutečnými tabulkami. Pouští ho jen uživatel ve vlastním terminálu. Skript odmítne běh pod Claude Code (`CLAUDECODE`) a při necommitnutých změnách, aby provenance neukazovala na jiný kód.
- `pixi run eda-synthetic [n]` vyrenderuje jeden nebo všechny dokumenty nad syntetickými tabulkami v dočasném adresáři a zkontroluje kanárka. Na `eda/_freeze/` v repu nesahá. Na konci vypíše cestu k PDF.
- `pixi run eda-test` pustí testy R pomocníků (testthat).

V sandboxu Codexu přesměruj datový adresář Quarta jen pro konkrétní render:

```bash
XDG_DATA_HOME=/tmp/mediparse-codex-cache/xdg-data pixi run --frozen eda-synthetic 1
```

Quarto zapisuje log kernelu do `${XDG_DATA_HOME:-~/.local/share}/quarto/logs`;
v sandboxu není výchozí umístění zapisovatelné. Proměnná proto nepatří do skriptů
ani aktivace pixi: v CI by mohla schovat písmo Libertinus instalované do
`~/.local/share/fonts`.

Tabulky parquet se čtou z `MEDIPARSE_PARQUET` (výchozí `resources/mimic/parquet`). Dočasné soubory DuckDB jdou do `duckdb-tmp` vedle toho adresáře, limit paměti určuje `MEDIPARSE_DUCKDB_MEMORY` (výchozí `8GB`).

Oba rendery běží nad kopií `eda/` v dočasném adresáři, ne na místě. Quarto v cestě se skrytou složkou (třeba `.worktrees/`) tiše ignoruje `_quarto.yml` a vyrenderuje výchozí HTML bez freeze, takže `quarto render eda` přímo ve worktree pod `.worktrees/` nic neudělá. Render na místě navíc zakládá v `eda/` netrackované soubory (`.gitignore`, mezisoubory knitr), kvůli kterým by provenance hlásila dirty. Reálný render proto pracuje vedle adresáře parquetu (pod `resources/mimic/`), zkopíruje zpět jen `eda/_freeze/<doc>/`, `results/eda/<doc>.pdf` a `results/eda/<doc>/` (starý adresář exportů v repu nejdřív smaže, aby zrušený export nezůstal) a po sobě uklidí. Skripty po renderu ověří, že PDF i freeze vznikly.

Pořadí reálného renderu: nejdřív commit `.qmd`, pak `pixi run eda <n>`, pak commit `eda/_freeze/`, `results/eda/<n>-*.pdf` a `results/eda/<n>-*/` (exporty). Jinak by provenance hlásila dirty strom. Commituje se PDF a zmražené výsledky, HTML ne.

## Zmražené výsledky

Projekt má `freeze: auto`. `quarto render eda` (celý projekt) použije zmražené výsledky a data nepotřebuje. Render jednoho souboru kód vždy spustí. CI proto renderuje celý projekt bez dat: změněný `.qmd` se zastaralým freeze sáhne po datech a CI spadne, takže zastaralý freeze se nedostane na main.

## Ochrana dat MIMIC

- DUA PhysioNetu o malých počtech nic neříká; práh n < 10 je pravidlo projektu a platí jen pro buňky skupin pacientů vymezených kvaziidentifikátorem (věk, pohlaví, rasa, období, diagnóza, výkon, lék, typ přijetí, pojištění, úmrtí) a pro kategorie, jejichž popisek je hodnota z dat. Takové kategorie s n < 10 se potlačují. Metriky textu a korpusu (počty záznamů, pacientů, přijetí a zpráv, zprávy na pacienta a přijetí, pokrytí, éra ICD, délky, značky, masky, kvalita, maska věku jako počet zpráv) nepopisují skupiny pacientů a zveřejňují se přesně. V sešitech 1–5 se potlačuje jen rozpad éry ICD podle období ukotvení pacienta (sešit 2, sešit 5 z něj přebírá jen závěr o přechodu mezi verzemi); sešit 6 potlačuje všechny kategoriální tabulky a graf demografie uvnitř každého atributu; sešit 7 potlačuje binární prevalence diagnóz funkcí `dua_binary`; sešit 8 potlačuje celé řádky kandidátů a příznaků, jejichž zmínky v textu vymezují skupinu pacientů diagnózou nebo příznakem.
- `dua_binary(n, total)` v `eda/R/dua.R` vrátí `NA` pro buňku, kde `n < 10` nebo `total − n < 10`, protože doplněk by šel dopočítat z přesně zveřejněného jmenovatele. Používá ho sešit 7 pro každou skupinu pacientů vymezenou diagnózou (Charlsonova kategorie, kandidát, skupina kódů v žebříčku, souběh); potlačená buňka se v tabulce ukáže jako „potlačeno“ a v grafu se nekreslí. Kanárek ověří potlačení jen u nemapovaných popisků z dat (období ukotvení v sešitu 2, doloženo mutací `DUA_THRESHOLD <- 1L`). V sešitech 6 a 7 potlačení dokazují testy testthat: popisky sešitu 6 R mapuje na české a skupina kódů jsou první tři znaky po `upper`, takže se prefix `eda-sentinel-` ztratí (mutace nad sešity 6 a 7 kanárek nevidí).
- `dua_suppress(df, by)` v `eda/R/dua.R` vyžaduje agregovanou tabulku se sloupcem `n`, řádky pod prahem sloučí do jednoho řádku „ostatní (n < 10)“ a vrací číselné `n`, z něhož se kreslí grafy. Skryté buňky doplňuje `dua_hidden` nejmenší zachovanou buňkou, dokud jejich součet není aspoň 10, aby nešly dopočítat z celku; s `by` potlačuje uvnitř každé skupiny zvlášť. Neagregovaná tabulka skončí chybou. Popisky smí být jen text nebo faktor; další číselný sloupec vedle `n` se odmítne, protože by nešel potlačit.
- Dotazy sešitu 2 (pacienti, pokrytí, éra ICD) jsou v `eda/R/patients.R` a testuje je testthat nad parquetem vyrobeným v testu, protože syntetika neobsahuje NULL.
- Metriky textu (sešit 3) jsou v `eda/R/text.R`. `note_metrics(con, table)` projde text zpráv jednou a vytvoří dočasnou tabulku s délkou, počtem slov, značkami `___`, zbytkovými maskami, příznaky věku a otiskem textu; agregační funkce čtou jen ji. Značka je přesně tři podtržítka (stejná definice jako `_MARKER` v `domain/translation_consistency.py`), delší úseky se počítají zvlášť. Z textu se zveřejňují jen počty a míry, nikdy nalezené řetězce ani kontext; percentily, minimum a maximum i počty zpráv jsou bez potlačení. Funkce testuje testthat nad parquetem s textem vytvořeným v testu (`helper-mimic.R`), protože syntetika nemá masky ani věk (značky `___` nese jen preambule zpráv, viz Kanárek).
- Tokenizace (sešit 4) je v `eda/R/tokens.R`. Tokenizují se XLM-R (`FacebookAI/xlm-roberta-base`), mDeBERTa (`microsoft/mdeberta-v3-base`) v pevných revizích a překladový model; jeho model, revize, okno a `SPLIT_RATIO` se čtou z `translation/prepare.py` přes `reticulate::import_from_path`, v `eda/R` se nepíšou znovu. `note_tokens(con, table, tokenizers)` čte zprávy z `eda_connect()` po dávkách 2 000 podle `note_id` a zapíše počty tokenů do dočasné tabulky `note_tokens_<tabulka>`, agregace čtou jen ji. Počet částí překladu z celého korpusu je spodní mez, přesné dělení (`note_parts.split` s `prepare._fits`) běží na deterministickém vzorku 1 000 zpráv na tabulku. Česká délka je odhad: poměr CS/EN z paralelní syntetiky v `resources/synthetic/{en,cs}`, ne měření na překladu MIMIC. Prostředí `eda` má kvůli tomu `transformers`, `tokenizers`, `sentencepiece`, `protobuf` a `jinja2` a `TRANSFORMERS_VERBOSITY=error`, aby hlášky transformers nešly do PDF. Tokenizér `google/translategemma-27b-it` je gated: lokálně stačí přihlášený token v `~/.cache/huggingface`, v CI ho krok `eda-synthetic` bere ze secretu repa `HF_TOKEN`. Funkce se testují offline nad falešným tokenizérem (R funkce se stejným voláním).
- Volba korpusu a rozdělení dat (sešit 5) přebírá ukazatele sešitů 2 a 3 voláním týchž funkcí, nové dotazy jsou v `eda/R/selection.R`. `corpus_funnel(con)` vybírá vstupní korpus z propouštěcích zpráv ve čtyřech navazujících krocích (všechny, s kódem ICD, aspoň `SHORT_LIMIT` znaků, bez přesných duplicit) a ke každému uvádí zprávy, pacienty a pacienty s více zprávami; délku a otisk textu bere z `note_metrics` a z duplicit ponechá zprávu s nejmenším `note_id`. `corpus_comparison` skládá srovnání `discharge` a `radiology` z výstupů `patient_summary`, `hadm_coverage`, `note_quality` a percentilů délky. Délky v tokenech se znovu nepočítají, uvádí je sešit 4.
- Demografie vstupního korpusu (sešit 6) je v `eda/R/demographics.R`. Populaci vymezuje `input_corpus_sql(con)` ze `selection.R`, tedy tentýž výběr jako `corpus_funnel` (`eligible_cte`). `demographic_notes(con)` vytvoří jednou dočasnou tabulku zpráv vstupního korpusu s atributy z `patients` a `admissions` (`LEFT JOIN`, chybějící řádek je „neuvedeno“); popisky kategorií jsou pevné slovníky v SQL, takže se surová hodnota z dat do výstupu nedostane. Pohlaví se počítá na pacienta, ostatní atributy na zprávu. Věk při přijetí je `anchor_age` + rok přijetí − `anchor_year` přes `TRY_CAST`; `anchor_age = 91` je maskovaný věk a jde do pásma 90+, stejně jako každý věk od 90, a zveřejňuje se jen medián a kvartily. Kategorie pod prahem se slučují v rámci každého atributu (`dua_suppress(counts, by = "atribut")`); verdikty (`majority_gender`, `largest_race`) a počet maskovaných (`masked_text` s pásmem 90+ z publikované tabulky, aby nešel dopočítat doplněk) čtou potlačenou tabulku. Graf kreslí jen buňky, které nebyly sloučeny ani podle pohlaví, ani podle věkového pásma (`drawable_cells`).
- Krajina diagnóz vstupního korpusu (sešit 7) je v `eda/R/diagnoses.R` a čte jen `diagnoses_icd` a `d_icd_diagnoses`. `diagnosis_codes(con)` vytvoří jednou dočasnou tabulku kódů zpráv vstupního korpusu (`upper(trim(icd_code))`, `TRY_CAST(icd_version AS INTEGER)`); kód se shoduje s předponou jen ve své verzi. Pool deseti kandidátů (předpony ICD-9 a ICD-10, pole `charlson`) je v `config/diagnosis_candidates.json` a čte se přes `read_candidates` (`jsonlite`), aby prevalence nemohla utéct od definice, kterou použije výběr štítků. Předpony pětice vybrané do štítků (diabetes, CHOPN, CKD, pneumonie, fibrilace síní) načítá z téhož souboru `load_diagnosis_codes` a test `tests/test_icd_gems.py` je ověřuje proti GEMs v `resources/icd/`. Pole `acronyms_en` pětice a první položku jejího `keywords_en` čte i preprocessor TF-IDF (`load_text_normalization`) jako slovník zkratka → kanonický tvar; změna pořadí klíčových slov nebo zkratek proto mění vstup modelu. Charlsonovy kategorie počítá balík `comorbidity` s mapami Quan 2005 zvlášť pro obě verze a bez hierarchie. Žebříčky tříznakových skupin pojmenovává nejčastější plný kód skupiny z číselníku a zobrazují jen skupiny, které prošly `dua_binary`. Souběh se skrývá v obou směrech, prevalence v obou érách spolu, rozdíl Charlsonovy kategorie a kandidáta pod deseti skryje obě čísla (`reconcile_charlson`) a název skupiny pochází jen z kódu s aspoň deseti zprávami; kódy ICD-9 začínající E se seskupují na čtyři znaky. Pásmo prevalence 5–60 % (`PREVALENCE_BAND`) je stanoveno předem. Rozhodnutí na konci sešitu (`preselection`, `uncovered_charlson`, `redundant_pairs`, `era_gaps`, `symptom_share`) jsou čisté funkce nad potlačenými tabulkami s větví `nelze` pro prázdný korpus.
- Zmínky diagnóz a příznaků v textu (sešit 8) jsou v `eda/R/mentions.R`; klíčová slova a zkratky kandidátů jsou v `config/diagnosis_candidates.json`, negační výrazy a seed lexikon příznaků v `config/text_mentions.json`. Zmínka je vzor RE2 (fráze bez ohledu na velikost písmen a s mezerou jako libovolným bílým znakem, zkratky s rozlišením), tvrzená zmínka vzniká vyříznutím negovaných úseků (výraz a do 40 znaků bez tečky, středníku a konce řádku) a novým hledáním. `mention_flags(con, candidates, mentions)` projde text jednou do dočasné tabulky `mention_flags_input`, sloupce se jmenují podle pořadí termínu (`k001_m`, `k001_t`, `p001_m`), ne podle fráze. Řádek kandidáta se skryje celý, když kterákoli ze šesti atomických buněk rozkladu ICD± × {bez zmínky, jen negovaná, tvrzená} klesne pod 10 (`suppress_agreement`; buňky 2×2 jsou jejich součty, takže stačí těchto šest), a také když ho sešit 7 nezveřejnil (`align_agreement`); příznak se skrývá podle `dua_binary` a rozdílu zmínek a tvrzení (`suppress_symptoms`). Syntetický text nese zmínky v první sekci lexikonu (viz Kanárek), takže nad ním řádek souladu diabetu má všech šest buněk, rubrika diabetes vybere a všech 20 příznaků má hodnoty. Kanárek ale potlačení tu neověří, protože popisky kandidátů a příznaků jsou konstanty configu, ne hodnoty z dat; potlačení a rubriku (`rubric`, `select_five`: jedna diagnóza na kapitolu ICD-10) dokazují testy nad fixture.
- Hlavičky a lexikon sekcí (sešit 9) jsou v `eda/R/sections.R`; lexikon je `config/section_headers.json` (sekce šablony MIMIC s celofrázovými synonymy a seznam administrativních polí `preamble`), čte ho R (audit hlaviček) a parser sekcí `domain/section_headers.py` (segmentace, viz další odstavec); sekce s příznakem `diagnostic` vystřihne strip D2 (`strip_diagnostic_sections`). Hlavička je řádek `HEADER_PATTERN` (RE2, odsazení povoleno, aspoň 3 znaky, bez číslic), shoda se synonymem je rovnost řetězců normalizovaných jedinou SQL funkcí `normalized_sql` (malá písmena, bílé znaky do jedné mezery, bez koncových mezer a teček). `note_headers(con)` projde text jednou do dočasné tabulky `note_headers_input`. Žebříček hlaviček publikuje jen hlavičky nad `publishable_floor` (1 % korpusu, nejméně 10 zpráv), protože řetězec hlavičky je popisek z dat; synonyma lexikonu se publikují všechna i s nulou. Syntetický text má preambuli a všechny sekce lexikonu, takže žebříček má hodnoty a kanárek hlídá, že `publishable_floor` nepustí pseudohlavičku jediné zprávy (doloženo mutací `publishable_floor` na 1); rozsah potlačení dokazují testy nad fixture.
- Segmentace zpráv a kostra šablony (sešit 9) jsou v `eda/R/segments.R`. Zprávy rozděluje parser sekcí z balíčku, ne dotaz: `segment_corpus(con, module, lexicon_path)` pošle texty vstupního korpusu po dávkách (`batched_table` v `eda/R/db.R`, týž pomocník jako `note_tokens`) do `mediparse.entrypoints.eda_sections.note_sections` přes reticulate a do dočasné tabulky `note_sections_input` zapíše jen pozici, klíč sekce a délky (`span_chars`, `body_chars`), ne text. Texty a identifikátory se do Pythonu předávají jako `as.list(unname(...))`, protože reticulate pojmenovaný seznam změní na slovník a duplicitní jména sloučí. Agregace (kostra podle mediánu pozice, konzistence pořadí, matice precedence, sekce na zprávu, délky těl, podíl textu stripu D2, zprávy bez nadpisů) jsou SQL nad touto tabulkou; diagnostické sekce stripu čte `strip_share` z lexikonu (příznak `diagnostic`). Funkce včetně grafů testuje testthat nad ručně vytvořenou tabulkou, protože syntetika sekce má, ale mnohé větve (odsazená hlavička, hlavička verzálkami, opakování další sekce) zůstávají nad ní nulové.
- Předzpracování pro TF-IDF (sešit 9) je v `eda/R/preprocessing.R`. Tokeny nepočítá dotaz, ale analyzér `tfidf_vectorizer(None | normalizace)` z `infrastructure/sklearn_classifier.py`, týž vectorizer, který používá trénink, volaný přes reticulate v `mediparse.entrypoints.eda_preprocessing`. `note_preprocessing` posílá zprávy vstupního korpusu po dávkách a do tabulky `note_preprocessing_input` zapíše jen počty tokenů bez a s předzpracováním a výskyty zkratek pětice na zprávu (`find_diagnosis_acronyms` nad normalizovaným textem); `note_vocabulary` zapíše do `preprocessing_vocabulary_input` distinct tokeny dávky po variantách, slovník je `count(DISTINCT token)` (analyzér před prořezáním `min_df`/`max_df`). Tabulka zkratek nezveřejňuje počet zpráv se zkratkou, jen výskyty (1–9 jako „< 10“) a podíl zpráv bez kódu diagnózy, protože počet zpráv by s podílem a sešitem 8 prozradil malou skupinu. Výskyty počet zpráv jen shora omezují, proto se podíl uvádí jen u zkratek s aspoň 1 000 výskyty (`SHARE_MIN_OCCURRENCES`) a navíc se skryje při buňce křížení zkratka × kód pod 10. Funkce testuje testthat přes `corpus_con`. Syntetické zprávy nesou zkratky pětice v první sekci (viz Kanárek), takže tabulka zkratek ukáže u DM, DM2 a COPD podíl a u AF, CKD a ESRD výskyty bez podílu (nad syntetikou nemají buňku „s kódem“); větve „< 10“ a „0 výskytů“ zůstávají nad syntetikou nevyzkoušené.
- Česká typografie je společná všem sešitům. `plural_form` a `count_phrase` v `eda/R/czech.R` skloňují po číslovkách (1 zmínka, 2 zmínky, 5 zmínek, 1,5 zmínky) a spojují číslo se slovem nedělitelnou mezerou; věty sešitu 8 skládané z počtů jdou přes ně. Lua filtr `eda/_filters/nbsp.lua` zapsaný v `eda/_quarto.yml` nahradí mezeru i zalomení řádku nedělitelnou mezerou po jednopísmenné předložce či spojce (k, s, v, z, o, u, a, i) a po čísle. Filtr běží až v pandocu, fragmenty `.tex` nemění (v práci je ošetří luavlna) a freeze je výsledek výpočtu, takže `quarto render eda` z commitnutých freeze změní PDF sešitů 1–7; jejich přerender a commit dělá uživatel.
- Pravidla přístupu k datům mají jedinou kopii v Pythonu. Doména (`domain/data_access.py`) drží blokující proměnné `CI` a `CLAUDECODE` s důvody a funkce `blocked_reason` a `check_data_access`; sdílí je audit korpusu, takže obě místa hlásí tentýž důvod. Infrastruktura (`infrastructure/eda_duckdb.py`, `EdaDatabase`) je jediná cesta ke spojení: `connect()` ověří, že adresář existuje a obsahuje parquet, pak stráž, a teprve potom otevře spojení a založí pohled nad každým parquetem. Bez úspěšné stráže spojení nevznikne.
- Stráž odmítne adresář bez platné značky `synthetic.json`, když je neprázdná proměnná `CLAUDECODE` nebo `CI`: skutečná data se tam číst nesmí. Značku čte `read_marker`, takže prázdný nebo poškozený soubor je chyba, ne průchod.
- Dokument dostává připojení z `eda_connect()`: dočasné soubory DuckDB leží vedle adresáře parquetu (`resources/mimic/duckdb-tmp`) a paměť je omezená na `MEDIPARSE_DUCKDB_MEMORY` (výchozí `8GB`, prázdná hodnota znamená výchozí). `eda_connect()` a `eda_parquet_dir()` v `eda/R/db.R` jen volají `entrypoints/eda_database.py` (`connection_settings`, `parquet_dir`) a podle výsledku otevřou vlastní spojení `r-duckdb`; prostředí mu předávají jako `as.list(Sys.getenv())`, protože `os.environ` pod reticulate je snímek ze startu. Python sešity použijí `database(os.environ).connect()`.
- `eda_check_python()` ověří, že reticulate použil Python z prostředí pixi. Prostředí obsahuje `micromamba` a nastavuje `RETICULATE_PYTHON` a `RETICULATE_CONDA`, jinak reticulate hledá binárku conda nebo tiše sáhne po cizím Pythonu.
- Do dokumentů patří jen agregáty. Řádková data a texty z MIMIC se nevypisují.

## Kanárek

`eda-synthetic` vygeneruje z veřejného manifestu validace (`results/mimic/manifest.json`) a z configů `config/diagnosis_candidates.json`, `config/section_headers.json` a `config/text_mentions.json` tabulky ve tvaru MIMIC. Mimo typové sloupce je každá buňka v tabulce jedinečný řetězec `eda-sentinel-<vlastník>-<sloupec>-<řádek>`. Vlastníkem je tabulka sama. U `subject_id` a `hadm_id` je to tabulka, která entitu zakládá (`patients`, `admissions`), a u `note_id` z MIMIC-IV-Ext-BHC tabulka `discharge`. Řádek r každé tabulky tak patří pacientovi a hospitalizaci r a vstupní korpus má tolik zpráv, kolik je řádků. Klíč není kategorie, takže každá kategorie mimo typové sloupce má dál n = 1 a únik popisku kategorie nebo textu z buňky je selhání. Text propouštěcí zprávy bloku (`discharge.text`, `domain/synthetic_discharge.py`) začíná na prvním řádku textem buňky s tokenem do `MIN_CHARS` znaků (tentýž tvar jako u vzácných řádků a radiologie, takže délku splní každá zpráva), pak má pole preambule `Pole: ___` a všechny sekce lexikonu v jeho pořadí; synonymum hlavičky je `synonyms[blok % počet synonym]`. Tělo každé sekce nese token zprávy ze samých písmen (`edasentinel` a číslo řádku zapsané písmeny) dvakrát: jako pseudohlavičku `Token:` mimo lexikon a v řádku se synonymem malými písmeny. Pseudohlavička je svědek žebříčku hlaviček (popisek z dat s n = 1, který `publishable_floor` musí skrýt). Za ní stojí v těle první sekce lexikonu řádky zmínek (`domain/synthetic_mentions.py`, každá na vlastním řádku): v tvrzených blocích klíčová slova pětice z `config/diagnosis_candidates.json`, příznaky z `config/text_mentions.json` a každá zkratka pětice opakovaná tolikrát, aby součet výskytů dosáhl prahu `SHARE_MIN_OCCURRENCES` sešitu 9 (50 krát, výskytů je 1 020), v negovaných blocích totéž s prvním negačním výrazem před každým výrazem a zkratka jednou (opakování by vyskočilo z okna 40 znaků a stalo se tvrzenou zmínkou). První sekce není BHC ani sekce kolem něj, takže řezy targetu a strip D2 zůstávají beze změny. Čtyři varianty bloků (`blok % 4`) se liší vynechanou sekcí hned za BHC, opakovanou hlavičkou diagnostické sekce v těle poslední sekce a řezem targetu BHC (kategorie sešitu 9 v pořadí variant: shoda, celá následující sekce, náš konec později, náš konec dřív). Target MIMIC-IV-Ext-BHC se skládá z částí téhož textu (ne z parseru, který sešit 9 proti targetu ověřuje) a má sloučené bílé znaky. Vzácné řádky jsou zprávy bez nadpisů (buňka a token do `MIN_CHARS` znaků filtru délky kohorty) a jejich target zůstává buňkou. Tokenizace zahodí pomlčky a normalizace čísel číslice, token ale projde beze změny. Token je v každé zprávě jiný (DF = 1), takže ho n-gramy ani žebříčky s prahem potlačení nepustí. Slova textu bez prefixu jsou slova synonym lexikonu a polí preambule, slova buňky, slova zmínek z configů a značka `___`, sdílená aspoň `MIN_CELL` zprávami (zmínky 20), takže pro ně platí pravidlo níže také.

Typové sloupce, které sešity 2 a 5-7 přetypovávají nebo porovnávají (`anchor_age`, `anchor_year`, `anchor_year_group`, `admittime`, `gender`, `insurance`, `marital_status`, `race`, `language`, `icd_version`, `icd_code` a číselník `d_icd_diagnoses`), nesou hodnoty slovníku MIMIC jako řetězce, protože prefix by `TRY_CAST` a `CASE` zahodily. Výchozích 62 řádků tvoří šest bloků po `MIN_CELL` řádcích (blok je `(řádek // MIN_CELL) % 6`) a dva vzácné řádky. Bloky 0, 1 a 4 mají ICD-9 a období `2008 - 2010`, bloky 2, 3 a 5 ICD-10 a `2017 - 2019`; diabetes mají bloky 0, 2 a 4, CHOPN bloky 1, 3 a 5; kódem je první předpona diagnózy z configu. Zmínky v textu mají bloky 0 a 1 tvrzené, 2 a 3 jen negované a 4 a 5 žádné, takže každá diagnóza má každý ze tří stavů jednou s kódem a jednou bez něj a řádek souladu ICD × zmínka v sešitu 8 má všech šest buněk aspoň 10. Každá hodnota bez prefixu má aspoň `MIN_CELL` řádků, každá verze ICD 30 zpráv a prevalence projde `dua_binary`. Vzácné řádky nesou prefix jen ve sloupcích-svědcích (`patients.anchor_year_group`, `diagnoses_icd.icd_version` a `icd_code`), jinde hodnoty bloku 0.

Pravidlo pro hodnotu bez prefixu (platí pro aspoň `BLOCKS * MIN_CELL` řádků, tedy 60): je to buď veřejná konstanta číselníku `d_icd_diagnoses` (kód, verze, název), nebo hodnota v tabulce pacientů, hospitalizací či diagnóz, kterou sdílí aspoň `MIN_CELL` řádků. Věk je shodný uvnitř bloku (při přijetí 20 až 70 podle bloku) a rok ukotvení a čas přijetí jsou konstanty všech řádků. Kategorie pod prahem nesou prefix, takže kdyby je potlačení nesloučilo, kanárek je najde (ověřeno mutací `DUA_THRESHOLD <- 1L`). Slepá místa: české popisky mapované v R (pohlaví, pásmo, pojištění) nejsou data a jejich potlačení hlídá jen testthat; skupiny kódů jsou první tři znaky po `upper`, takže se prefix ztratí, a vzácné řádky proto nemají platnou verzi a do žebříčku se nedostanou. Nad syntetikou se kategorie Charlsona `cpd` a CHOPN skryjí v `reconcile_charlson` (rozdíl 30 − 30 < 10), diabetes zůstane viditelný.

Po renderu skript převede PDF na text (`pdftotext`) a hledá `eda-sentinel-` v kopii `eda/` (včetně freeze), ve výstupu a v textu PDF. Navíc hledá `edasentinel` bez ohledu na velikost písmen, a to v textu PDF, kde se měkký spojovník, kterým Typst dělí dlouhá slova, maže spolu s konci řádků za ním (i před prázdným řádkem) a ostatní konce řádků jsou mezera. Sousední popisky `eda` a `sentinel` tak nedají falešný token. Nalezený soubor znamená návratový kód 1 s jeho cestou. Hledá se prefix s pomlčkou, protože provenance prefix uvádí v popisu zdroje dat.

Kanárek čte i exporty: `.tex` fragmenty hledá `grep` nad `results/`, text grafů z exportovaných PDF (`pdftotext` nad všemi PDF pod `results/eda` rekurzivně). Typst sešitu vykresluje graf jako PNG, ale exportované PDF vzniká ze stejného objektu, takže kanárek vidí osy i popisky kategorií. Mezera „text v obrázcích“ tím zaniká.

## Exporty pro text práce

Každý graf a každá tabulka sešitu se ukládá i pro LaTeX do `results/eda/<sešit>/`. Text práce je odkazuje přes `../results/eda/` (`\graphicspath`, `\input`) a nekopíruje je. V R se soubor jmenuje podle labelu chunku (`tbl-korpus.tex`, `fig-….pdf`); label musí odpovídat `^[a-z0-9-]+$`. V Pythonu název tabulky určuje argument `label` exportéru (viz níže). `\caption`, `\label` a float píše text práce.

- `export_table(df, col.names, align, code = character())` (`eda/R/export.R`) zapíše fragment `tabular` s `booktabs` bez floatu a vrátí `knitr::kable` ze stejného data.frame pro Typst, chunk tedy volá jen tuto funkci. Šířky sloupců bere z `#| tbl-colwidths` chunku a sloupec se šířkou sází jako `p{\dimexpr w\linewidth-2\tabcolsep\relax}`; bez nich zůstane prosté `l/c/r`. Sloupce v `code` se v Typstu obalí backticky, v LaTeXu `\texttt{}`. Tabulky se exportují všechny, včetně provenance.
- `export_json(x, label, dir)` (`eda/R/export.R`) zapíše strojově čitelný export `<label>.json` vedle tabulek (`check_label`, `jsonlite::write_json` s `auto_unbox`, `digits = NA` a `na = "null"`); pole, která mají zůstat polem i s jedním nebo žádným prvkem, se obalují `I()`. Sešit 8 jím v chunku `parametry-syntetiky` zapisuje `results/eda/8-mentions/parametry-syntetiky.json`: `selection` (pětice v pořadí `config/diagnosis_candidates.json`), `prevalence`, matici `conditional` a `mentions.<k>.given_positive|given_negative|keywords` po potlačení, podíly na 4 desetinná místa, potlačená hodnota je `null`, výběr „nelze“ dá prázdnou `selection` a prázdné objekty (`synthetic_parameters` v `eda/R/mentions.R`). Export čte příkaz `mediparse-synthetic-parameters` (pixi task `synthetic-parameters`), který ho převezme do `config/synthetic_plan.json`.
- `export_figure(plot, height_cm)` uloží `ggplot` jako vektorové PDF (`cairo_pdf`, šířka sazby 16,5 cm, výška v cm) a vrátí graf, aby ho chunk vykreslil do sešitu. Základem je `theme_thesis()`: Libertinus Serif, 10 pt, graf se v práci vkládá bez škálování.
- Písmo Libertinus Serif musí být v systému (Fedora `libertinus-fonts`, Debian `fonts-libertinus`); conda-forge ho nemá a Ubuntu ho samostatně nebalí. Jinde ho `scripts/install-libertinus.sh` stáhne z vydání Libertinus 7.051 na GitHubu, ověří otisk SHA-256 a rozbalí řez Serif do `~/.local/share/fonts`, kde ho najde fontconfig z prostředí pixi. Tak ho instaluje i CI. `export_figure` přes `pdffonts` ověří, že ho PDF vložilo, protože cairo jinak tiše dosadí jiné.
- Exporty jdou ze stejného objektu jako sešit, po potlačení tam, kde ho pravidlo vyžaduje. Render z freeze exporty nepřepisuje ani nemaže: commitnuté `results/eda/<sešit>/` jsou výsledek posledního reálného renderu. Změna `.qmd` proto vyžaduje nový `pixi run eda <n>` a commit freeze i exportů, jinak CI krok „render z freeze“ spadne.

### Tabulky v Pythonu (engine jupyter)

Setup buňka sešitu 1 načte přehled přes entrypoint a vytvoří exportér.
`include: false` skryje setup i jeho výstupy ve freeze:

````qmd
```{python}
#| include: false
import os

from IPython.display import Markdown

from mediparse.entrypoints import eda_corpus, eda_export

corpus = eda_corpus.corpus_overview(os.environ)
export = eda_export.export(os.environ)
```
````

Přehled nese hotovou doménovou `ThesisTable`, takže tabulková buňka nerozkládá
rozvržení na jednotlivé argumenty:

```python
export.write(corpus.table, label="tbl-korpus", caption="Přehled korpusu.")
```

Pro tabulku sestavenou z textových řádků přímo v sešitu zůstává `export.table`.
Tabulková buňka vrací posledním výrazem výsledek jediného volání:

```python
export.table(
    [("diagnoses_icd", "1\u00a0234")],
    header=("Tabulka", "Záznamů"),
    align="lr",
    code=("Tabulka",),
    label="tbl-korpus",
    caption="Přehled korpusu.",
    widths=(70, 30),
)
```

Buňka nemá `#| label`, `#| tbl-cap` ani `#| tbl-colwidths`. Label, popisek a šířky
se zadávají jen v tomto volání. Exportér zapíše `<label>.tex` a vrátí objekt
s protokolem `_repr_markdown_()`. Jupyter jej zobrazí jako pipe tabulku zakončenou
řádkem `: Přehled korpusu. {#tbl-korpus tbl-colwidths="[70,30]"}`. Quarto z něj
převezme id pro křížové odkazy, popisek i šířky; další id buňky by bylo duplicitní.
Bez `widths` se atribut vynechá a LaTeX použije prosté sloupce `l/c/r`.

Label musí odpovídat `^tbl-[a-z0-9-]+$`, popisek musí být neprázdný na jednom řádku.
Hlavička a všechny buňky smějí obsahovat jen `str` bez konců řádku. Čísla nejprve
naformátuje volající; exportér nepřevádí `None` ani číselné typy na text.
`code` jmenuje sloupce podle hlavičky, v LaTeXu je obalí `\texttt{}` a v markdownu
backticky. Popisek sloučených kategorií `ostatní (n < 10)` zůstává prostým textem.

Adresář se odvodí z prostředí Quarta jako
`<rodič QUARTO_DOCUMENT_PATH>/results/eda/<kmen QUARTO_DOCUMENT_FILE>`.
Chybějící nebo prázdná proměnná vyvolá chybu. Adresář vznikne až po validaci celé
tabulky, labelu a popisku; opakovaný export stejného labelu přepíše soubor.

Inline text se v jupyteru vrací jako `` `{python} Markdown(s)` `` s importem
`from IPython.display import Markdown` v sešitu. Před otevíracím backtickem musí
být mezera, jinak Quarto inline nerozpozná. Například:

```qmd
Databáze obsahuje `{python} Markdown(corpus.tables)` tabulek.
```

`s` musí být vždy neprázdný text. Prázdný `Markdown("")` Quarto vypíše jako
zdrojový kód inline výrazu. Volitelná věta o největší tabulce je proto spojená
s neprázdnou větou o manifestu v `corpus.records`. Samotný `str` se vypíše jako
repr s uvozovkami a escape sekvencemi, například `\xa0` místo nedělitelné mezery.
`Markdown(s)` navíc čte soubor, pokud je `s` existující cesta, a u adresáře vyvolá
`IsADirectoryError`; předávej mu věty, nikoli cesty.

Golden soubory v `tests/fixtures/thesis_table/` jsou společná spustitelná
specifikace Python a R exportu včetně provenance. Vznikly původním R exportem;
pytest i testthat ověřují stejné bajty. R implementace i její golden test zůstanou
do konce migrace z R. Parita zahrnuje padding markdownu, NBSP a prázdné tabulky bez `\midrule`.

Parita je omezena na vstupy používané v sešitech: ochrana `{}` před `[` v kable
používá `\s` podle locale (TRE), první hlavičku ze samých mezer kable měří jako
prázdnou a šířku markdownu měří bez URL odkazů (`remove_urls`) přes
`nchar(type = "width")`, zatímco Python používá `len`. Žádná z těchto odlišností
v sešitech nenastává; obecnější napodobení by přidalo pravidla bez využití.

## Provenance

Každý dokument končí přílohou s commitem, příznakem dirty, otiskem `pixi.lock`, zdrojem dat (otisky CSV z `config/mimic_tables.json`, u syntetiky prefix) a verzemi Pythonu, DuckDB a v sešitech na knitru také R. Skládá ji `mediparse.entrypoints.eda_provenance`. Při syntetickém renderu se commit předává proměnnými `MEDIPARSE_COMMIT` a `MEDIPARSE_DIRTY`, protože kopie v dočasném adresáři není repozitář git.

Přílohu na jupyteru sestaví `mediparse.entrypoints.eda_provenance.appendix(os.environ)` a exportuje ji jako `tbl-provenance` s popiskem „Provenance renderu.“ a šířkami 28/72. Klíč `r` obsahuje jen R příloha, protože v sešitu na jupyteru R neběží; mapa českých popisků jej zná kvůli paritě s R.
