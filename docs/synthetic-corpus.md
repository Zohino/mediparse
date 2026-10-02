# Specifikace syntetického korpusu

Syntetický korpus napájí smoketest a demo pipeline: 200 anglických propouštěcích
zpráv a jejich české protějšky. Zprávy nejsou z MIMIC, ale strukturou odpovídají
MIMIC-IV-Note, aby je zpracovával tentýž kód jako reálná data.

Tento dokument je závazná specifikace. Verzí specifikace je commit tohoto souboru
a `provenance.json` korpusu na něj odkazuje. Číselné parametry plánu drží
`config/synthetic_plan.json` jako jediný zdroj; specifikace na ně odkazuje klíči
a vlastní hodnoty uvádí jen tam, kde v configu nejsou.

## Postup

1. **Plán.** Deterministický vzorkovač s pevným seedem vylosuje pro každou zprávu
   plán (`resources/synthetic/plans.jsonl`).
2. **Verbalizace.** Model převede plán do textu podle šablony instrukcí v repu.
3. **Kontrola shody.** Automatické kontroly ověří text proti plánu. Zpráva, která
   neprojde, se napíše znovu ze stejného plánu.
4. **Audit.** Autor lokálně ověří, že korpus nesdílí s MIMIC-IV-Note žádný úsek
   textu ani pacienta. Do repa smí jen auditovaný korpus.

## Požadavky

| | Požadavek |
|---|---|
| R1 | Licence: korpus jde zveřejnit pod licencí repozitáře (MIT). |
| R2 | Bez textu MIMIC: nulový překryv 13-gramů slov s referencí (viz [Audit](#audit-a-brána)). |
| R3 | Strukturní věrnost: preambule, sekce, podnadpisy, de-identifikační značky, délky a vztah labelu a textu odpovídají agregátům MIMIC-IV-Note; výjimkou je vědomě zkrácený narativ. |
| R4 | Labely: každá zpráva nese labely pěti diagnóz; vztah labelu a textu má řízený šum. |
| R5 | Skupinová struktura: část pacientů má víc zpráv, aby rozdělení dat podle pacienta mělo co oddělovat. |
| R6 | Párovost: každá česká zpráva má anglický protějšek se stejným `note_id`, pacientem a labely. |
| R7 | Provenance: generování je zaznamenané tak, aby šlo zopakovat se shodnými vlastnostmi korpusu, ne se shodným textem. |
| R8 | Doba běhu: smoketest nad korpusem proběhne nejvýše za několik hodin. |

## Záznam zprávy

| Pole | Typ | Význam |
|---|---|---|
| `note_id` | řetězec | `subject_id-DS-k`, kde `k` je pořadí zprávy pacienta (skladba MIMIC-IV-Note) |
| `subject_id` | celé číslo | syntetický pacient, od `patients.first_subject_id` (90 000 001) výše |
| `text` | řetězec | text zprávy |
| labely | logická hodnota | `diabetes`, `ckd`, `heart_failure`, `atrial_fibrillation`, `aki` |

## Plán

Plán zprávy (`NotePlan`) určuje `note_id`, `subject_id`, labely, pohlaví,
přítomnost věkové značky, přítomné sekce a podnadpisy, délku narativu a její
rozdělení do sekcí, počet narativních značek a zmínky diagnóz se statusem
a sekcemi. Parametry jsou v configu:

| Oblast | Klíč configu |
|---|---|
| seed | `seed` |
| pacienti a počet zpráv | `patients` |
| prevalence a společný výskyt labelů | `labels` |
| sekce, podnadpisy, délka narativu, pohlaví, věková značka | `structure` |
| zmínky, klíčová slova, negace | `mentions` |

Pravidla vzorkování, která config sám nevyjadřuje:

- Chronické diagnózy (`diabetes`, `ckd`, `heart_failure`, `atrial_fibrillation`)
  platí pro všechny zprávy pacienta; `aki` se u dalších zpráv losuje podmíněně.
- Výskyt každé sekce se losuje nezávisle; pořadí sekcí je pořadí v configu.
- Z podnadpisů v jedné skupině configu (vylučující se varianty) se losuje nejvýše
  jeden.
- Zmínka u pozitivního labelu má status `affirmed`; u negativního se losuje
  z `mentions.negative_statuses` (`negated`, `family_history`, `uncertain`,
  `affirmed_uncoded`). Zmínka `affirmed_uncoded` nikdy nestojí v Discharge
  Diagnosis, zmínka `family_history` stojí ve Family History, je-li v plánu.

## Pravidla verbalizace

Verbalizuje `claude-opus-5-5` v Claude Code. Vstupem je zadání, které kód vyrobí
ze šablony instrukcí v repu a z plánu zprávy; model dostane přesně tento text.
Model rozhoduje jen o povrchové podobě textu, tedy o formulacích a klinických
detailech konzistentních s plánem. O struktuře ani labelech nerozhoduje.

**Preambule.** Osm polí, po dvou na řádku: Name, Unit No, Admission Date,
Discharge Date, Date of Birth, Sex, Service, Attending. Hodnotou je `___`, kromě
Service (volí verbalizace v souladu s plánem) a Sex (`F` nebo `M` podle pohlaví
z plánu). Pořadí a párování polí se liší od šablony MIMIC. Počet nad MIMIC-IV-Note
(331 793 zpráv) potvrdil, že tato pole jsou v reálných zprávách téměř vždy
a stojí na řádku za jiným polem; Sex nese `M`/`F` v 99,99 % zpráv.

**Sekce a podnadpisy.** Kanonické hlavičky sekcí (`structure.sections[].header`)
právě v pořadí a výběru podle plánu, žádné jiné kanonické hlavičky. Hlavička stojí
na začátku řádku a končí dvojtečkou (`Brief Hospital Course:`), jako v MIMIC;
tělo sekce sahá do další hlavičky. Podnadpisy jen ty z plánu, uvnitř sekce, ke které
patří, ve stejném tvaru (`Facility:` na začátku řádku). Preambule stojí před první
hlavičkou.

**De-identifikační značky.** Jediná forma značky je `___`.

- Strukturní značky podle plánu: šest hodnot preambule, těla sekcí Social History
  a Followup Instructions, hodnota podnadpisu Facility a věková značka, je-li
  v plánu. Věková značka je `___` hned před `year old` nebo `y/o`
  (`___ year old`, `___-year-old`, `___ y/o`); bez věkové značky v plánu text věk
  neuvádí vůbec.
- Narativní značky: přesně `narrative_deid` z plánu, na místech, kde by text nesl
  chráněný údaj (jména, data, místa, instituce, telefonní čísla).
- Text nikdy neobsahuje číselný věk ani čitelné jméno, datum či místo.

**Délka.** Narativní sekce mají délku podle `section_words` z plánu (slova);
preambule, hlavičky a krátké sekce mají reálnou délku.

**Zmínky diagnóz.** Zmínka je výskyt klíčového slova diagnózy z
`mentions.diagnoses.<diagnóza>.keywords`, bez ohledu na velikost písmen,
ohraničený hranicemi slov (definice EDA).

- Diagnóza bez zmínky v plánu: v textu není žádné její klíčové slovo, ani zkratka
  (například `AF`, `DM`, `HF`). Pozitivní label bez zmínky smí model popsat
  nepřímo, například vývojem kreatininu nebo léčbou.
- Diagnóza se zmínkou: klíčové slovo stojí v sekcích z plánu.
- Slovník se překrývá: například „renal insufficiency“ je klíčové slovo `ckd`,
  takže se nesmí použít tam, kde plán zmínku `ckd` nemá.
- Status `negated` a `family_history`: negační výraz z `mentions.negation.cues`
  stojí nejvýše `mentions.negation.window_chars` (40) znaků před klíčovým slovem
  ve stejné větě, aby zmínku zachytilo i pravidlo negace EDA.
- Status `uncertain`: zmínka je formulovaná jako nejistá nebo hypotetická.

**Volné úseky.** Hodnoty šablonových polí, obsah krátkých sekcí a formát řádků
léků formuluje model volně, aby korpus nereprodukoval šablony MIMIC (R2).

## Kontroly shody

Kontroly jedné zprávy (výstupem jsou `note_id` zpráv k přegenerování):

1. Hlavičky sekcí odpovídají plánu včetně pořadí a jiné kanonické hlavičky se
   nevyskytují.
2. Strukturní značky stojí na všech místech, která určuje plán, a věková značka je
   v textu právě tehdy, když ji plán má; text neobsahuje číselný věk ani jinou
   formu značky než `___` (`[**`, `XXX`, podtržítka jiné délky). Pole Sex
   v preambuli odpovídá pohlaví z plánu a Service má hodnotu.
3. Pro diagnózu bez zmínky se v textu nevyskytuje žádné její klíčové slovo; pro
   diagnózu se zmínkou alespoň jedno v plánovaných sekcích.
4. Sekce Discharge Diagnosis obsahuje klíčové slovo diagnózy právě tehdy, když to
   plán určuje.

Kontroly korpusu (tolerance NÁVRH):

| Vlastnost | Cíl | Tolerance |
|---|---|---|
| medián délky narativu | `structure.narrative.median_words` | ±10 % |
| σ logaritmu délky narativu | `structure.narrative.sigma` | ±0,1 |
| hustota narativních značek | `structure.narrative.deid_per_word` | ±15 % |
| prevalence každého labelu | `labels.prevalence` | `labels.prevalence_tolerance` |

Kontroly jedné zprávy se týkají zpráv, které v korpusu jsou: zpráva bez plánu je
porušení, plán bez zprávy ne (korpus vzniká po polovinách); úplnost hlídají
kontroly korpusu. Kontroly ověřují jen explicitní vlastnosti textu, ne klinickou
věrohodnost; tu posuzuje autor při review. Kontroly běží jako příkaz pro smyčku přegenerování
a jako test v CI nad korpusem v repu.

## Audit a brána

- **Kritérium.** Nulový překryv 13-gramů slov po normalizaci
  `nfkc-lower-alnum-deid-v1` (NFKC, malá písmena, tokeny jsou souvislé úseky písmen
  a číslic, `___` je jeden token) a žádný syntetický `subject_id` mezi pacienty
  reference.
- **Reference.** Tabulky projektu `mimic-iv-note` v `config/mimic_tables.json`
  (discharge a radiology, MIMIC-IV-Note 2.2) s oficiálními otisky SHA-256. Audit
  odmítne soubory, které s nimi nesedí, dřív, než je začne číst.
- **Protokol.**
  1. Audit spouští autor lokálně, mimo CI a mimo relaci Claude Code
     (`just audit-corpus`).
  2. Výstupem jsou jen počty a `note_id` dotčených zpráv; pozice shod jdou do
     reportu mimo repozitář, shodný text se nevypisuje nikdy.
  3. Zpráva se shodou se napíše znovu ze svého plánu. Generátor se dozví jen
     `note_id`, ne proč ani kde byla shoda.
  4. Čistý audit zapíše `audit.json`: otisk korpusu, reference se jmény, otisky
     a počty řádků, metodu, commit nástroje a datum.
- **Brána.** Hook při commitu a test v CI porovnají otisk korpusu, metodu
  a referenci s `audit.json`. Neauditovaný nebo změněný korpus neprojde.

## Uložení

Adresář `resources/synthetic/`:

| Soubor | Obsah |
|---|---|
| `plans.jsonl` | plány zpráv, společné pro oba jazyky |
| `en/<note_id>.txt`, `cs/<note_id>.txt` | jedna zpráva na soubor |
| `labels.csv` | `note_id`, `subject_id` a labely, odvozené z plánů |
| `provenance.json` | záznam podle oddílu [Provenance](#provenance) |
| `audit.json` | záznam čistého auditu |

Vstup smoketestu ze souborů korpusu sestavuje S12a.

## Provenance

`provenance.json` obsahuje:

- seed vzorkovače a otisk jeho configu;
- model (`claude-opus-5-5`) a verzi Claude Code v době generování;
- datum generování a otisk šablony instrukcí;
- commit této specifikace;
- odkaz na `audit.json`.

Stejný seed dává identické plány. Verbalizace je nedeterministická: nový běh dává
jiný text se stejnou strukturou, labely a zmínkami a s délkami v tolerancích.
Rozhodující verzí korpusu je ta uložená v repozitáři.

## Český korpus

Každá anglická zpráva se překládá do češtiny (S11e). Páry sdílejí `note_id`,
pacienta, labely i plán.

- Česká zpráva má stejný počet značek `___` jako anglický protějšek.
- Zmínky diagnóz odpovídají plánu i v češtině; česká klíčová slova a negační
  výrazy definuje S11e.
- Kanonické hlavičky zůstávají anglické, překládá se obsah sekcí (NÁVRH).
- Na český korpus se uplatní tentýž audit.

Otevřená rozhodnutí S11e: překladač a platnost anglických regexů pro zkratky, které
překlad zachová.

## Změny specifikace

Specifikace se mění commitem. Korpus vzniklý podle starší verze se pozná podle
commitu specifikace v `provenance.json`; změna pravidel, která korpus nesplňuje,
znamená jeho přegenerování nebo opravu.
