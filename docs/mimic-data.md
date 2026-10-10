# Data MIMIC-IV

Reálná data práce pocházejí ze dvou projektů PhysioNetu: z MIMIC-IV-Note 2.2,
který obsahuje text zpráv, a z modulu `hosp` MIMIC-IV 3.1. Z modulu `hosp`
pocházejí diagnózy, procedury a kontext hospitalizace. Oba projekty jsou pod
DUA. Do repa proto smí jen agregáty a veřejné schéma, nikdy hodnoty z řádků.
Seznam tabulek, URL a SHA-256 drží `config/mimic_tables.json` jako jediný zdroj.

## Postup

1. **Stažení.** `snakemake download_mimic` stáhne tabulky do `resources/mimic/`
   a každý soubor ověří proti otisku z oficiálních `SHA256SUMS.txt`.
2. **Kontrola otisků.** `pixi run verify-mimic` otisky přepočítá i později, například
   po přesunu dat nebo na jiném stroji.
3. **Validace.** `snakemake validate_mimic` zapíše
   [`results/mimic/manifest.json`](../results/mimic/manifest.json) a flag
   `resources/mimic/validated.flag`. Kroky nad daty MIMIC mají flag na vstupu
   a bez prošlé validace nepoběží.
4. **Převod na parquet.** `snakemake parquet_mimic` převede každou validovanou
   tabulku na `resources/mimic/parquet/<tabulka>.parquet`.

Validace selže, pokud tabulka nejde rozbalit, nemá hlavičku ani záznam nebo
končí neuzavřeným polem v uvozovkách. Takhle vypadá useknutý gzip nebo rozbitý
CSV. Počet záznamů se nepočítá z fyzických řádků, protože text zpráv má
zalomení uvnitř polí v uvozovkách. Záznam končí tam, kde je od začátku souboru
sudý počet uvozovek.
Počty v manifestu se shodují s počty, které uvádí dokumentace MIMIC.

Převod čte CSV po blocích, takže paměť drží jen jeden blok i u `discharge`.
Všechny sloupce zůstávají řetězce a prázdná hodnota je NULL, takže parquet je
bezeztrátová kopie CSV: kód `0389` si nechá úvodní nulu. Typy sloupců přiřadí
až dotazy nad parquetem. Počet záznamů a jména sloupců z převodu se porovnají
s inventářem validace. Pokud se liší, krok skončí chybou a výstup se smaže.
Počty tak ověřují dvě nezávislé metody: parita uvozovek ve validaci a parser
pyarrow v převodu. Krok nad reálnými daty se spouští ručně; testy běží nad malými
syntetickými `.csv.gz`.

## Tabulky

Stav podle manifestu. Velikost je velikost komprimovaného `.csv.gz`.

| Tabulka | Projekt a modul | Záznamů | MB | Sloupců |
|---|---|---:|---:|---:|
| `discharge` | MIMIC-IV-Note 2.2 `note` | 331 793 | 1 139,2 | 8 |
| `radiology` | MIMIC-IV-Note 2.2 `note` | 2 321 355 | 781,8 | 8 |
| `admissions` | MIMIC-IV 3.1 `hosp` | 546 028 | 19,9 | 16 |
| `d_icd_diagnoses` | MIMIC-IV 3.1 `hosp` | 112 107 | 0,9 | 3 |
| `d_icd_procedures` | MIMIC-IV 3.1 `hosp` | 86 423 | 0,6 | 3 |
| `diagnoses_icd` | MIMIC-IV 3.1 `hosp` | 6 364 488 | 33,6 | 5 |
| `omr` | MIMIC-IV 3.1 `hosp` | 7 753 027 | 44,1 | 5 |
| `patients` | MIMIC-IV 3.1 `hosp` | 364 627 | 2,8 | 6 |
| `pharmacy` | MIMIC-IV 3.1 `hosp` | 17 847 567 | 525,7 | 27 |
| `prescriptions` | MIMIC-IV 3.1 `hosp` | 20 292 611 | 606,3 | 21 |
| `procedures_icd` | MIMIC-IV 3.1 `hosp` | 859 655 | 7,8 | 6 |
| `services` | MIMIC-IV 3.1 `hosp` | 593 071 | 8,6 | 5 |

Celkem je to 3,17 GB komprimovaně. Jména sloupců jsou v manifestu.

## Co z toho plyne

- **Zprávy jsou drtivá většina objemu.** `discharge` a `radiology` tvoří 60 %
  komprimovaných dat. Převod na parquet (S19) se proto vyplatí hlavně u nich.
- **Propouštěcí zpráva patří k hospitalizaci.** `discharge` má 331 793 záznamů
  a `admissions` 546 028. Kolik hospitalizací zprávu má, ukáže teprve spojení
  přes `hadm_id`, ne počty souborů.
- **Labelů je víc než zpráv.** `diagnoses_icd` má v průměru 11,7 kódu na
  hospitalizaci (6 364 488 / 546 028). Jedna zpráva tak nese víc diagnóz
  a klasifikace je multilabel.
- **Kódy ICD mají dvě verze.** `d_icd_diagnoses` a `diagnoses_icd` nesou sloupec
  `icd_version`, proto se ICD-9 a ICD-10 musí sjednotit (S22).

## Co validace neověřuje

Validace ověřuje jen soubory: úplnost, otisk, tvar CSV, počty a schéma.
Hodnoty ve sloupcích, chybějící hodnoty ani vazby mezi tabulkami nekontroluje.
Převod je jen kopie a hodnoty také nekontroluje. To patří ke krokům nad
parquetem (S20, S20a).
