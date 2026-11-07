# Data MIMIC-IV

Reálná data práce pocházejí ze tří projektů PhysioNetu: z MIMIC-IV-Note 2.2,
který obsahuje text zpráv, z modulu `hosp` MIMIC-IV 3.1 a z MIMIC-IV-Ext-BHC 1.2.0.
Z modulu `hosp` pocházejí diagnózy, procedury a kontext hospitalizace.
MIMIC-IV-Ext-BHC obsahuje sekci Brief Hospital Course propouštěcích zpráv, kterou
jeho autoři vytáhli vlastním regexem; práce ji používá jen jako referenci
pro validaci parseru sekcí. Všechny tři projekty jsou pod DUA a každý má vlastní
podpis smlouvy na PhysioNetu. Do repa proto smí jen agregáty a veřejné schéma,
nikdy hodnoty z řádků. Seznam tabulek, URL a SHA-256 drží
`config/mimic_tables.json` jako jediný zdroj očekávaných otisků CSV.

## Postup

1. **Stažení.** `snakemake download_mimic` stáhne tabulky do `resources/mimic/`
   a každý soubor ověří proti otisku z oficiálních `SHA256SUMS.txt`.
2. **Kontrola otisků.** `pixi run verify-mimic` otisky přepočítá i později, například
   po přesunu dat nebo na jiném stroji. Kontrola je potřebná i po přesunu se
   zachovaným mtime, kdy Snakemake nemusí inventář znovu spustit.
3. **Validace.** `snakemake validate_mimic` přepočte SHA-256 každého CSV a porovná
   ho s configem, i když soubor pipeline nestahovala. Ověřený otisk zapíše
   do inventáře tabulky jako `sha256` a společně s počty a sloupci do
   [`results/mimic/manifest.json`](../results/mimic/manifest.json) a flag
   `resources/mimic/validated.flag` označí úspěšnou validaci. Převod má flag na
   vstupu a bez prošlé validace nepoběží.
4. **Převod na parquet.** `snakemake parquet_mimic` převede každou validovanou
   tabulku na `resources/mimic/parquet/<tabulka>.parquet`. Vedle ní zapíše
   `<tabulka>.source.json`: `source` je jméno CSV, `source_sha256` jeho ověřený
   otisk z inventáře a `sha256` otisk zapsaného parquetu. Oba soubory jsou
   výstupem pravidla převodu.
5. **Kohorta a EDA.** Před použitím tabulky znovu spočítají otisk parquetu a
   porovnají ho s vazbou. Chybějící nebo neodpovídající vazba krok odmítne
   s pokynem převést tabulku znovu. Manifest kohorty uvádí pod jmény
   `discharge` a `diagnoses_icd` celé vazby skutečně čtených tabulek.
   Provenance EDA ověří všechny `*.parquet` v adresáři a uvede pro každou
   tabulku `csv_sha256 <tabulka>` a `parquet_sha256 <tabulka>`.

Validace selže, pokud otisk CSV nesouhlasí s configem, tabulka nejde rozbalit, nemá hlavičku ani záznam nebo
končí neuzavřeným polem v uvozovkách. Takhle vypadá useknutý gzip nebo rozbitý
CSV. Počet záznamů se nepočítá z fyzických řádků, protože text zpráv má
zalomení uvnitř polí v uvozovkách. Záznam končí tam, kde je od začátku souboru
sudý počet uvozovek.
Počty v manifestu odpovídají počtům, které uvádí dokumentace MIMIC.

Tabulky MIMIC-IV a MIMIC-IV-Note jsou `.csv.gz`, MIMIC-IV-Ext-BHC je nekomprimované
`.csv`. Validace i převod rozhodují podle přípony, takže poškozený gzip neprojde
jako prosté CSV. Jméno tabulky je jméno souboru bez přípony.

Převod čte CSV po blocích, takže paměť drží jen jeden blok i u `discharge`.
Všechny sloupce zůstávají řetězce a prázdná hodnota je NULL, takže parquet je
bezeztrátová kopie CSV: kód `0389` si nechá úvodní nulu. Typy sloupců přiřadí
až dotazy nad parquetem. Počet záznamů a jména sloupců z převodu se porovnají
s inventářem validace. Pokud se liší, krok skončí chybou a výstup se smaže.
Počty tak ověřují dvě nezávislé metody: parita uvozovek ve validaci a parser
pyarrow v převodu. Krok nad reálnými daty se spouští ručně; testy běží nad malými
syntetickými `.csv.gz`.

Řetěz původu je tedy config → ověřené CSV v inventáři → vazba CSV a parquetu
→ znovu ověřený parquet u konzumenta. Syntetický smoketest zapisuje stejné
vazby se zdrojem `https://mediparse.invalid/synthetic/<tabulka>`; protože
nemá zdrojové CSV, oba otisky ve vazbě patří syntetickému parquetu.
EDA rozpoznává svůj syntetický adresář nadále pomocí `synthetic.json`.

Nový tvar veřejných manifestů a provenance EDA se nad reálnými daty vytvoří
při společném uživatelském přepočtu po S39c3.

## Tabulky

Stav podle manifestu. Velikost je velikost staženého souboru, u `.csv.gz` komprimovaná, u BHC nekomprimovaná.

| Tabulka | Projekt a modul | Záznamů | MB | Sloupců |
|---|---|---:|---:|---:|
| `discharge` | MIMIC-IV-Note 2.2 `note` | 331 793 | 1 139,2 | 8 |
| `radiology` | MIMIC-IV-Note 2.2 `note` | 2 321 355 | 781,8 | 8 |
| `mimic-iv-bhc` | MIMIC-IV-Ext-BHC 1.2.0 | 270 033 | 2 726,5 | 5 |
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

Celkem je to 5,90 GB staženo, z toho 3,17 GB komprimovaných tabulek MIMIC-IV a MIMIC-IV-Note. Jména sloupců jsou v manifestu.

## Co z toho plyne

- **Zprávy jsou drtivá většina objemu tabulek MIMIC-IV a MIMIC-IV-Note.** `discharge` a `radiology` tvoří 60 %
  komprimovaných tabulek MIMIC-IV a MIMIC-IV-Note. Převod na parquet (S19) se proto vyplatí hlavně u nich.
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
